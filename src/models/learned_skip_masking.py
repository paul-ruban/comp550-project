# Goal identify optimal token masking based on classification and demasking loss
import torch

from nltk.text import TokenSearcher
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizer

from src.models.rnn_model import RNN
from src.data.dataio import Dataset

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


class DeepSkipAugmenter(torch.nn.Module):
    def __init__(
        self,
        bert_model : str = "distilbert-base-uncased"
    ) -> None:
        super().__init__()
        self.bert_name = bert_model
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.masking_model = RNN(output_size=2)
        self.unmasking_model = AutoModelForMaskedLM.from_pretrained(bert_model)
        self.classifier = RNN(output_size=1)
    
    def train(
        self,
        dataset : Dataset,
        n_epochs : int = 1, 
        batch_size : int = 32, 
        log_every : int = 1
    ):
        loader = DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=True)
        
        masking_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id)
        optimizer = torch.optim.Adam(self.masking_model.parameters(), lr=0.001)

        for epoch in range(n_epochs):
            for i, batch in enumerate(loader):
                # try masking input where log_probas.detach().clone().argmax(dim=1) is 1
                inputs = self.tokenizer(
                    text=batch["text"],
                    padding=True,
                    return_attention_mask=True,
                    truncation=True,
                    return_tensors="pt",
                    return_special_tokens_mask=True
                )

                # print('inputs is', inputs)

                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                special_tokens_mask = inputs["special_tokens_mask"]

                optimizer.zero_grad()
                # log_probas = self.model(input_ids)
                # print(input_ids.shape)
                log_probas = self.masking_model(input_ids)
                
                #C Compute masking loss
                masked_targets = torch.ones_like(input_ids)
                loss = masking_loss(log_probas, masked_targets)

                # Decide what to mask
                # log_probas.detach().clone().argmax(dim=1) * attention_mask
                special_tokens_mask = torch.gt(special_tokens_mask, 0)
                mask_probas = log_probas.detach().clone().argmax(dim=1) * attention_mask
                print('pre mask_probas is', mask_probas)
                mask_probas = torch.where(special_tokens_mask, mask_probas, 0)
                print('post mask_probas is', mask_probas)

                print('special_tokens_mask is', special_tokens_mask)
                mask_probas = torch.gt(mask_probas, 0)
                # mask_probas = torch.where(special_tokens_mask, mask_probas, False)
                print('input_ids is', input_ids)
                input_ids = torch.where(mask_probas, input_ids, 103)
                print('masked input_ids is', input_ids)

                # if i and i % (log_every) == 0:
                #     print(f"Step {i}: loss = {loss.item()}")
                #     out = (log_probas.detach().clone().argmax(dim=1) * attention_mask).tolist()
                #     print(out)
                
                loss.backward()
                optimizer.step()
        
        return self
