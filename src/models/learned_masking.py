# Goal identify optimal token masking based on classification and demasking loss
import contextlib
import copy
import logging
import torch
import torch.nn.functional as F
from typing import Tuple

from nltk.text import TokenSearcher
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils import PreTrainedTokenizer

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.models.rnn_model import RNN
from src.data.dataio import Dataset

logger = logging.getLogger()
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


class HighwayAugmenter(torch.nn.Module):
    def __init__(
        self,
        tokenizer : PreTrainedTokenizer,
        masking_model : torch.nn.Module,
        unmasking_model : torch.nn.Module,
        classifier : torch.nn.Module,
        max_seq_length : int = 512
    ) -> None:

        super().__init__()
        self.tokenizer = tokenizer
        self.masking_model = masking_model
        self.unmasking_model = unmasking_model
        self.max_seq_length = max_seq_length
        # freeze all parameters of unmasking_model
        for p in self.unmasking_model.parameters():
            p.requires_grad = False

        self.classifier = classifier
    
    def get_params(self):
        # get parameters of masking_model and classifier
        return [
            {"params": self.masking_model.parameters()},
            {"params": self.classifier.parameters()}
        ]
   
    def forward(
        self,
        input_ids : torch.tensor = None,
        attention_mask : torch.tensor = None,
        special_tokens_mask: torch.tensor = None
    ) -> Tuple[torch.tensor, torch.tensor]:

        maskable_tokens = attention_mask * ~(special_tokens_mask > 0)
        print("maskable_tokens", maskable_tokens)

        # Masking model: RNN
        mask_out, mask_embeddings = self.masking_model(input_ids, ret_pre_dense=True)

        # Decide what tokens to mask and mask them with [MASK] embeddings
        tokens_to_mask = (mask_out.log_softmax(dim=-1).argmax(dim=-1) * maskable_tokens).unsqueeze(dim=-1)
        print("tokens_to_mask", tokens_to_mask)
        mask_emb = self.unmasking_model.embeddings.word_embeddings.weight[self.tokenizer.mask_token_id]
        mask_embeddings = torch.where(tokens_to_mask > 0, mask_embeddings, mask_emb)

        # Unmasking model: BERT
        unmasked_output = self.unmasking_model(inputs_embeds=mask_embeddings, attention_mask=attention_mask)
        unmasked_embeddings = unmasked_output["last_hidden_state"]

        # Classification: take the last output value
        cls_out = self.classifier(inputs_embeds=unmasked_embeddings)[:,-1,:]

        return mask_out, cls_out


class WeightedMaskClassificationLoss(torch.nn.Module):
    def __init__(
        self,
        lambda_mask : float = 1.0,
        lambda_cls : float = 1.0,
        ignore_index = 0
    ) -> None:
        super().__init__()
        assert (lambda_mask >= 0 and lambda_cls >= 0)
        self.lambda_mask = lambda_mask
        self.lambda_cls = lambda_cls
        self.mask_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.cls_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, mask_out, mask_labels, cls_out, cls_labels):
        # need to transpose (-2, -1) to mathc [B, C, H]
        mask_loss = self.lambda_mask * self.mask_loss(mask_out.transpose(-2, -1), mask_labels)
        print("mask_loss", mask_loss)
        cls_loss = self.lambda_cls * self.mask_loss(cls_out, cls_labels)
        print("cls_loss", cls_loss)

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
        log_interval: int = 50,
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
            mask_out, cls_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask
            )
            # print("mask_out.shape", mask_out.shape)
            # print("cls_out.shape", cls_out.shape)
            mask_labels = attention_mask * ~(special_tokens_mask > 0)
            cls_labels = batch["label"].to(device)
            loss = criterion(
                mask_out=mask_out,
                mask_labels=mask_labels,
                cls_out=cls_out,
                cls_labels=cls_labels
            )
            loss.backward()
            optimizer.step()
            total_acc += (cls_out.detach().clone().log_softmax(-1).argmax(-1) == cls_labels).sum().item()
            total_count += cls_labels.size(0)
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
        model.eval()
        model = model.to(device)
        y_pred = []
        y_true = []
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
                mask_out, cls_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask
                )
                print("mask_out.shape", mask_out.shape)
                print("cls_out.shape", cls_out.shape)
                mask_labels = attention_mask * ~(special_tokens_mask > 0)
                
                # TODO add logging of percentage of masked tokens
                predicted_label = cls_out.log_softmax(-1).argmax(-1)
                y_pred.append(predicted_label)
                y_true.append(batch["label"])
        y_pred = torch.cat(y_pred).cpu() # back to cpu for sklearn
        y_true = torch.cat(y_true).cpu() # back to cpu for sklearn

        accu_val = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_val = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        logger.info("*" * 59)
        logger.info(
            "| end of epoch {:3d} valid accuracy {:8.3f} and f1 score {:8.3f}".format(
                epoch, accu_val, f1_val
            )
        )
        logger.info("*" * 59)
        return accu_val, f1_val

    def report_metrics(model, dataloader, logger):
        model.eval()
        model = model.to(device)
        y_pred = []
        y_true = []
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
                mask_log_probas, cls_log_probas = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask
                )
                mask_labels = attention_mask * ~(special_tokens_mask > 0)
                # TODO add logging of percentage of masked tokens
                predicted_label = cls_log_probas.argmax(dim=1)
                y_pred.append(predicted_label)
                y_true.append(batch["label"])
        y_pred = torch.cat(y_pred).cpu() # back to cpu for sklearn
        y_true = torch.cat(y_true).cpu() # back to cpu for sklearn
        accu_val = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_val = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
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