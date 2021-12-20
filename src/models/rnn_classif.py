import contextlib
import copy
import logging
import re
from typing import List, Tuple

import pandas as pd
import torch
import transformers
from src.utils.json_utils import read_json_lines
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JsonDataset(Dataset):
    def __init__(
        self,
        json_file_path,
        tokenizer: transformers.models.distilbert.tokenization_distilbert.DistilBertTokenizer,
        data_type: str,
    ) -> None:
        json_lines = read_json_lines(json_file_path)
        df = pd.DataFrame(
            {
                "text": [json_dict["text"] for json_dict in json_lines],
                "label": [json_dict["label"] for json_dict in json_lines],
            }
        )
        if data_type == "smokers":
            df["label"] = df["label"].replace({2: 2, 3: 2, 4: 2})
        self.dataset = df
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"text": self.dataset.iloc[idx]["text"], "label": self.dataset.iloc[idx]["label"]}
        return sample

    def yield_samples(self) -> None:
        assert self.dataset is not None
        for sample in self.dataset:
            yield sample["text"]

    def label_pipeline(self, label) -> int:
        return int(label)

    def text_pipeline(self, text) -> List[int]:
        return self.tokenizer(text).input_ids

    def collate_batch(self, batch, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        # Collating function for the dataloader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        label_batch, text_batch = [], []
        for sample in batch:
            _text = sample["text"]
            _label = sample["label"]
            label_batch.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_batch.append(processed_text)
        label_batch = torch.tensor(label_batch, dtype=torch.int64)
        text_batch = pad_sequence(
            text_batch, batch_first=True, padding_value=tokenizer._pad_token_type_id
        )
        return text_batch.to(device), label_batch.to(device)


class RNNClassifier(torch.nn.Module):
    def __init__(
        self,
        model_type="lstm",
        pretrained_word_embedding: torch.nn.modules.sparse.Embedding = None,
        hidden_dim=256,
        num_layers=1,
        output_dim=2,
        bidirectional=True,
        dropout=0.2,
    ):
        super(RNNClassifier, self).__init__()
        assert model_type in ["lstm", "gru"], "rnn_type can be one of: 'lstm', 'gru'."
        assert pretrained_word_embedding is not None, "Must provide pretrained word embedding."
        # Get the rnn type
        rnn_type = nn.LSTM if model_type == "lstm" else nn.GRU
        # Create the NN layers
        self.embedding = pretrained_word_embedding
        self.bidirectional = bidirectional
        self.rnn = rnn_type(
            input_size=pretrained_word_embedding.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_batch):
        # Pass the input batch into the embedding layer
        embedded_input = self.embedding(input_batch)
        # Pass embedded batch into RNN
        _, (hidden, _) = self.rnn(embedded_input)
        # Apply dropout to the hidden state
        hidden = self.dropout(hidden)
        # Use the last hidden state of the RNN as the output
        # (might be 2 if use bidirectional)
        if self.bidirectional:
            hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden_cat = hidden[-1, :, :]
        # Pass the hidden state into the fully connected layer
        output = self.fc(hidden_cat)
        # NOTE: Do NOT pass the output through the softmax layer
        # because CL loss expected un-normalized values
        return output


class RNNTrainer:
    def __init__(
        self,
        model: RNNClassifier,
        optimizer: torch.optim,
        criterion: nn.CrossEntropyLoss,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        logger: logging.Logger,
        num_epochs: int,
        log_interval: int = 50,
        early_stopping_threshold: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.early_stopping_threshold = early_stopping_threshold

    def train(self):
        # Running variables
        best_valid_acc, best_f1_score = 0.0, 0.0
        epoch_counter = 0
        best_model, best_loss, best_optimizer = None, None, None
        model, optimizer, criterion = self.model, self.optimizer, self.criterion
        self.logger.info("=" * 59)
        self.logger.info(f"Training model {self.model}")
        for epoch in range(self.num_epochs):
            if epoch_counter >= self.early_stopping_threshold:
                logger.warning(
                    f"The validation f1 score has not improved for {epoch_counter} epochs. Stopping training early."
                )
                break
            logger.info("*" * 59)
            logger.info(f"Training epoch {epoch}")
            model, loss = self.train_one_epoch(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                dataloader=self.train_dataloader,
                logger=self.logger,
            )
            valid_acc, f1_score = self.evaluate_one_epoch(
                epoch=epoch, model=model, dataloader=self.val_dataloader, logger=self.logger
            )
            if f1_score > best_f1_score:
                with contextlib.redirect_stdout(None):
                    best_model = copy.deepcopy(model)
                    best_loss = loss
                    best_optimizer = copy.deepcopy(optimizer)
                    best_valid_acc = valid_acc
                    best_f1_score = f1_score
                    epoch_counter = 0  # reset counter to 0
            else:
                epoch_counter += 1
        self.best_model = best_model
        self.best_loss = best_loss
        self.best_optimizer = best_optimizer
        self.best_f1_score = best_f1_score
        self.best_valid_acc = best_valid_acc

    def train_one_epoch(self, epoch, model, optimizer, criterion, dataloader, logger):
        model.to(device)
        model.train()
        total_acc, total_count = 0, 0
        for idx, (text_batch, label_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text_batch)
            loss = criterion(predicted_label, label_batch)
            loss.backward()
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label_batch).sum().item()
            total_count += label_batch.size(0)
            if idx % self.log_interval == 0 and idx > 0:
                logger.info(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| training accuracy {:8.3f}".format(
                        epoch, idx, len(dataloader), total_acc / total_count
                    )
                )
                total_acc, total_count = 0, 0
        return model, loss

    def evaluate_one_epoch(self, epoch, model, dataloader, logger):
        model.to(device)
        model.eval()
        y_pred = torch.tensor([]).to(device)
        y_true = torch.tensor([]).to(device)
        with torch.no_grad():
            for text, label in dataloader:
                prediction = model(text)
                predicted_label = prediction.argmax(1)
                y_pred = torch.cat((y_pred, predicted_label))
                y_true = torch.cat((y_true, label))
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        accu_val = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_val = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        logger.info("*" * 59)
        logger.info(
            "| end of epoch {:3d} valid accuracy {:8.3f} and f1 score {:8.3f}".format(
                epoch, accu_val, f1_val
            )
        )
        logger.info("*" * 59)
        return accu_val, f1_val

    def report_metrics(model, dataloader, logger):
        model.to(device)
        model.eval()
        y_pred = torch.tensor([]).to(device)
        y_true = torch.tensor([]).to(device)
        with torch.no_grad():
            for text, label in dataloader:
                prediction = model(text)
                predicted_label = prediction.argmax(1)
                y_pred = torch.cat((y_pred, predicted_label))
                y_true = torch.cat((y_true, label))
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        accu_val = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_val = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        logger.info("*" * 59)
        logger.info("|valid accuracy {:8.3f} and f1 score {:8.3f}".format(accu_val, f1_val))
        logger.info(classification_report(y_true=y_true, y_pred=y_pred))
        logger.info(confusion_matrix(y_true=y_true, y_pred=y_pred))
        logger.info("*" * 59)
        return accu_val, f1_val

    # Code from https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
    def save_checkpoint(save_path, model, optimizer, loss):
        if save_path is None:
            return

        state_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }

        torch.save(state_dict, save_path)
        logger.info(f"Model saved to ==> {save_path}")

    def load_checkpoint(load_path, model, optimizer):
        if load_path is None:
            return

        state_dict = torch.load(load_path, map_location=device)
        logger.info(f"Model loaded from <== {load_path}")

        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])

        return state_dict["valid_loss"]
