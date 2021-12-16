# Goal identify optimal token masking based on classification and demasking loss
import torch
import copy
import logging
import contextlib
from copy import deepcopy
import torch.nn.functional as F
from typing import Tuple, Union

from nltk.text import TokenSearcher
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils import PreTrainedTokenizer

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.models.rnn_model import RNNClassifier
from src.data.dataio import Dataset

logger = logging.getLogger()
SEP_LINE_LEN = 90
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Randomly mask tokens to initialize the model

# 2. Pnmask tokens using a technique such as transformers

# 3. Perform the same classification procedures as on original TokenSearcher

# 4. use the difference in classification performs and reconstruction compared to original text to tweak the weights / learn the ModuleNotFoundError

# 5. Adjust weights and repeat

# Models : Masking, Bert, Classification


# Components

# MaskinEncoder

# DecoderEnconding

# Classification (RNN classifier, for the sequenece 1 target, look fpr those, get inspiration from Pual's RNN models, instead of hidden to token, should self.denseClassifier)


class RNNMasker(torch.nn.Module):
    def __init__(
        self,
        rnn_type : str = "lstm", 
        embeddings : torch.nn.modules.sparse.Embedding = None,
        hidden_dim : int = 768, 
        num_layers : int = 1,
        bidirectional : bool = False,
        dropout : float = 0.1,
    ) -> None:

        super().__init__()
        assert (rnn_type in ["lstm", "gru"]), "rnn_type can be one of: 'lstm', 'gru'."
        rnn_type = torch.nn.LSTM if rnn_type == "lstm" else torch.nn.GRU
        self.embeddings = embeddings
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.rnn = rnn_type(
            input_size=embeddings.embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, # prevents warning for 1 layer
            bidirectional=bidirectional,
            batch_first=True
        )

        self.dropout = torch.nn.Dropout(p=dropout)
        if self.hidden_dim * (2 if bidirectional else 1) != embeddings.embedding_dim:
            self.projection = torch.nn.Linear(
                in_features=self.hidden_dim * (2 if bidirectional else 1),
                out_features=embeddings.embedding_dim)
        else:
            self.projection = None
        self.dense = torch.nn.Linear(in_features=embeddings.embedding_dim, out_features=1)
        
    def forward(
        self, 
        input_ids : torch.Tensor = None, # [Batch, SeqLen]
        inputs_embeds : torch.Tensor = None, # [Batch, SeqLen, EmbDim]
        return_hidden : bool = False
    ) -> torch.Tensor:

        assert (input_ids is None or inputs_embeds is None), "Can take either input_ids or inputs_embeds, not both."
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids) # [Batch, SeqLen, EmbDim]
        hidden, _ = self.rnn(inputs_embeds) # [Batch, SeqLen, HidDim * (2 if bidirectional else 1)]
        if self.projection:
            hidden = self.projection(hidden) # [Batch, SeqLen, EmbDim]
        hidden = self.dropout(hidden) # [Batch, SeqLen, EmbDim]
        output = self.dense(hidden) # [Batch, SeqLen, 1]
        if return_hidden:
            return output, hidden
        else:
            return output


class HighwayAugmenter(torch.nn.Module):
    def __init__(
        self,
        tokenizer : PreTrainedTokenizer,
        masker : torch.nn.Module,
        unmasker : torch.nn.Module,
        classifier : torch.nn.Module,
        max_seq_length : int = 512,

    ) -> None:

        super().__init__()
        self.tokenizer = tokenizer
        self.masker = masker
        self.unmasker = unmasker
        self.max_seq_length = max_seq_length
        # freeze all parameters of masker
        for p in self.unmasker.parameters():
            p.requires_grad = False

        self.classifier = classifier
    
    def get_params(self):
        # get parameters of masker and classifier
        return [
            {"params": self.masker.parameters()},
            {"params": self.classifier.parameters()}
        ]
   
    def forward(
        self,
        input_ids: torch.tensor,
        attention_mask: torch.tensor,
        special_tokens_mask: torch.tensor,
        apply_mask : bool = False,
        return_mask : bool = False
    ) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:

        # Ignore speacial tokens: [CLS], [SEP], [PAD]
        maskable_tokens = attention_mask * ~(special_tokens_mask > 0)

        # Get token embeddings using BERT
        with torch.no_grad():
            embeddings = self.unmasker.embeddings(input_ids) # NON-contextualized
            # embeddings = self.unmasker(
            #     input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"] # contextualized

        # Masker model: RNN
        masker_output, masker_embeddings = self.masker(
            inputs_embeds=embeddings, return_hidden=True)

        # Apply sigmoid to masker_output to get masking probabilities
        mask_probas = torch.sigmoid(masker_output) * maskable_tokens.unsqueeze(dim=-1)
        # If probability of masking > 50%, replace with [MASK] embedding
        tokens_to_mask = mask_probas > 0.5 # [Batch, SeqLen, 1]
        # Get 768-dim embedding of [MASK] token from BERT
        mask_embedding = deepcopy(
            self.unmasker.embeddings.word_embeddings.weight[self.tokenizer.mask_token_id])
        
        if apply_mask:
            # Replace appropriate emebddings with [MASK] embeddings
            embeddings = torch.where(tokens_to_mask, masker_embeddings, mask_embedding)

        # Unmasking model: BERT (NO BACKPROP)
        with torch.no_grad():
            embeddings = self.unmasker(
                inputs_embeds=embeddings, 
                attention_mask=attention_mask
            )["last_hidden_state"]

        # Concat mask_out with unmasked_embeddings as external feature
        embeddings = torch.cat([embeddings, masker_output], dim=-1) # 768 + 1 = 769
        # Classification: take the last output value
        cls_out = self.classifier(inputs_embeds=embeddings)

        if return_mask:
            return cls_out, tokens_to_mask
        else:
            return cls_out
    
    def __str__(self):
        s = ""
        s += str(self.masker) + '\n'
        s += "BERTModel - " + self.unmasker.name_or_path + '\n'
        s += str(self.classifier)
        return s


class WeightedMaskClassificationLoss(torch.nn.Module):
    def __init__(
        self,
        lambda_mask : float = 0.0,
        lambda_cls : float = 1.0,
        ignore_index = 0
    ) -> None:
        super().__init__()
        assert (lambda_mask >= 0 and lambda_cls >= 0)
        self.lambda_mask = lambda_mask
        self.lambda_cls = lambda_cls
        self.mask_loss = torch.nn.CrossEntropyLoss()
        self.cls_loss = torch.nn.CrossEntropyLoss()
    
    def forward(
        self, 
        mask_out: torch.tensor = None, # [B, L, C]
        mask_labels: torch.tensor = None, # [B, L]
        cls_out: torch.tensor = None, # [B, C]
        cls_labels: torch.tensor = None # [B]
    ):
        # print("mask_out.shape", mask_out.shape)
        # print("mask_labels.shape", mask_labels.shape)
        # print("cls_out.shape", cls_out.shape)
        # print("cls_labels.shape", cls_labels.shape)
        # Have to transpose mask out as CrossEntropyLoss requires it to be [B, C, L]
        mask_loss = self.lambda_mask * self.mask_loss(mask_out.transpose(-2, -1), mask_labels)
        cls_loss = self.lambda_cls * self.cls_loss(cls_out, cls_labels)

        return mask_loss + cls_loss


class HighwayAugmenterTrainer:
    def __init__(
        self,
        model: HighwayAugmenter,
        optimizer: torch.optim,
        criterion: WeightedMaskClassificationLoss,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        logger: logging.Logger,
        num_epochs: int,
        log_interval: int = 10,
        early_stopping_threshold: int = 10,
    ) -> None:

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
        self.logger.info("=" * SEP_LINE_LEN)
        self.logger.info(f"Training model:\n{self.model}")
        for epoch in range(self.num_epochs):
            if epoch_counter >= self.early_stopping_threshold:
                logger.warning(
                    f"The validation f1-score has not improved for {epoch_counter} epochs. Stopping training early."
                )
                break
            logger.info("*" * SEP_LINE_LEN)
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
        model.train()
        model = model.to(device)
        total_acc, total_count = 0, 0
        for idx, batch in enumerate(dataloader):
            inputs = self.model.tokenizer(
                text=batch["text"],
                padding=True,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                truncation=True,
                return_tensors="pt",
                max_length=model.max_seq_length
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            special_tokens_mask = inputs["special_tokens_mask"].to(device)

            optimizer.zero_grad()
            cls_out, tokens_to_mask = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                apply_mask=True,
                return_mask=True
            )
            # Compute micro-average of NOT REALLY "masked" token ratio
            maskable_tokens = attention_mask * ~(special_tokens_mask > 0)
            masked_ratio = tokens_to_mask.sum() / maskable_tokens.sum()

            cls_labels = batch["label"].to(device)
            loss = criterion(cls_out, cls_labels)
            loss.backward()
            optimizer.step()
            total_acc += (cls_out.detach().clone().argmax(dim=-1) == cls_labels).sum().item()
            total_count += cls_labels.size(0)
            if idx % self.log_interval == 0 and idx > 0:
                logger.info(
                    "| epoch {:3d} | {:3d}/{:3d} batches "
                    "| training accuracy {:5.3f} "
                    "| masked_ratio {:5.3f}".format(
                        epoch, idx, len(dataloader), total_acc / total_count, masked_ratio
                    )
                )
                total_acc, total_count = 0, 0
        return model, loss

    def evaluate_one_epoch(self, epoch, model, dataloader, logger):
        model.eval()
        model = model.to(device)
        y_pred = []
        y_true = []
        maskable_tokens = 0
        masked_tokens = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = self.model.tokenizer(
                    text=batch["text"],
                    padding=True,
                    return_attention_mask=True,
                    return_special_tokens_mask=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=model.max_seq_length
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                special_tokens_mask = inputs["special_tokens_mask"].to(device)
                # DO NOT APPLY MASK, but return as if it could be maked
                cls_out, tokens_to_mask = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                    apply_mask=False,
                    return_mask=True
                )
                # Compute micro-average masked token ratio
                maskable_tokens += (attention_mask * ~(special_tokens_mask > 0)).sum().item()
                masked_tokens += tokens_to_mask.sum().item()

                predicted_label = cls_out.argmax(dim=-1)
                y_pred.append(predicted_label)
                y_true.append(batch["label"])
        y_pred = torch.cat(y_pred).cpu() # back to cpu for sklearn
        y_true = torch.cat(y_true).cpu() # back to cpu for sklearn

        accu_val = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_val = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        logger.info("*" * SEP_LINE_LEN)
        logger.info(
            "| end of epoch {:3d} | valid accuracy {:5.3f} | f1-score {:5.3f} | masked_ratio {:5.3f}".format(
                epoch, accu_val, f1_val, masked_tokens / maskable_tokens
            )
        )
        logger.info("*" * SEP_LINE_LEN)
        return accu_val, f1_val

    def report_metrics(model, dataloader, logger):
        model.eval()
        model = model.to(device)
        y_pred = []
        y_true = []
        maskable_tokens = 0
        masked_tokens = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = model.tokenizer(
                    text=batch["text"],
                    padding=True,
                    return_attention_mask=True,
                    return_special_tokens_mask=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=model.max_seq_length
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                special_tokens_mask = inputs["special_tokens_mask"].to(device)
                cls_out, tokens_to_mask = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                    apply_mask=False,
                    return_mask=True
                )
                # Compute micro-average masked token ratio
                maskable_tokens += (attention_mask * ~(special_tokens_mask > 0)).sum().item()
                masked_tokens += tokens_to_mask.sum().item()
                # TODO add logging of percentage of masked tokens
                predicted_label = cls_out.argmax(dim=1)
                y_pred.append(predicted_label)
                y_true.append(batch["label"])
        y_pred = torch.cat(y_pred).cpu() # back to cpu for sklearn
        y_true = torch.cat(y_true).cpu() # back to cpu for sklearn
        accu_val = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_val = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        logger.info("*" * SEP_LINE_LEN)
        logger.info(
            "|valid accuracy {:5.3f} | f1-score {:5.3f} | masked_ratio {:5.3f}".format(
                accu_val, f1_val, masked_tokens / maskable_tokens)
        )
        logger.info(classification_report(y_true=y_true, y_pred=y_pred))
        logger.info(confusion_matrix(y_true=y_true, y_pred=y_pred))
        logger.info("*" * SEP_LINE_LEN)
        return accu_val, f1_val

    def mask_data(model, dataloader, logger, n=10):
        model.eval()
        model = model.to(device)
        original_token_ids = []
        class_labels = []
        masked_tokens_bool = []
        n_remaining = n
        special_tokens = list(model.tokenizer.special_tokens_map.values())

        with torch.no_grad():
            for batch in dataloader:
                inputs = model.tokenizer(
                    text=batch["text"],
                    padding=True,
                    return_attention_mask=True,
                    return_special_tokens_mask=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=model.max_seq_length
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                special_tokens_mask = inputs["special_tokens_mask"].to(device)
                _, tokens_to_mask = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask,
                    apply_mask=False,
                    return_mask=True
                ) 
                tokens_to_mask = tokens_to_mask.squeeze(dim=-1) # [Batch, SeqLen]

                original_token_ids.extend(input_ids[:n].tolist())
                masked_tokens_bool.extend(tokens_to_mask[:n].tolist())
                class_labels.extend(batch["label"][:n].tolist())
                n_remaining = n - len(original_token_ids)
                if n_remaining == 0:
                    break

        logger.info("*" * SEP_LINE_LEN)
        logger.info("Masking Examples:")
        logger.info("*" * SEP_LINE_LEN)
        for i in range(len(original_token_ids)):
            original_tokens = []
            masked_tokens = []
            for j in range(len(original_token_ids[i])):
                original_token = model.tokenizer.convert_ids_to_tokens(original_token_ids[i][j])
                if original_token not in special_tokens:
                    original_tokens.append(original_token)
                    if not masked_tokens_bool[i][j]:
                        masked_tokens.append(original_token)
                    else:
                        masked_tokens.append('_' * len(original_token))
            logger.info("# {:2d} | ORIGINAL | CLASS {} | {}".format(i, class_labels[i], ' '.join(original_tokens)))
            logger.info("# {:2d} |  MASKED  | CLASS {} | {}".format(i, class_labels[i], ' '.join(masked_tokens)))
            logger.info("*" * SEP_LINE_LEN)
        logger.info("*" * SEP_LINE_LEN)


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