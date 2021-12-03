from abc import abstractmethod
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from src.models.model import Model
from typing import Union, List, Iterator, Tuple
from src.data.dataio import Dataset
from src.models.tokenizing import Tokenizer
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter


class RNN(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim, 
                num_layers, dropout, bidirectional, tie_weights):
        super().__init__()

        assert (rnn_type in ["lstm", "gru"]), "rnn_type can be one of: 'lstm', 'gru'."
        rnn_type = nn.LSTM if rnn_type == "lstm" else nn.GRU

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = rnn_type(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.tie_weights = tie_weights
        if bidirectional:
            # create projection from 2 * hidden_dim to hidden_dim
            self.projection = nn.Linear(2 * hidden_dim, hidden_dim)

        self.hidden2token = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        
        # tie input and output embedding weights
        if tie_weights:
            self.hidden2token.weight = self.embeddings.weight

    def forward(self, x):
        x = self.embeddings(x)
        x, _ = self.rnn(x)
        if self.rnn.bidirectional:
            x = self.projection(x)
        x = self.hidden2token(x)
        log_probas = F.log_softmax(x, dim=-1)
        log_probas = log_probas.transpose(-1, 1)
        return log_probas


class RNNBaseLM(Model):
    def __init__(
        self,
        tokenizer : Tokenizer, 
        rnn_type : str = "lstm",
        embedding_dim : int = 300, 
        hidden_dim : int =  300, 
        num_layers : int = 2, 
        dropout : float = 0.1, 
        bidirectional : bool = False,
        tie_weights : bool = True
    ) -> None:

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tokenizer = tokenizer

        self.model = RNN(
            rnn_type=rnn_type,
            vocab_size=len(tokenizer.vocab),
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            tie_weights=tie_weights
        )
    
    def parameters(self) -> Iterator[Parameter]:
        return self.model.parameters()

    def _get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _pad_batch(
        self, 
        x : List[List[int]], 
        batch_first : bool = True, 
        return_pad_mask : bool = False
    ) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        
        padded_batch = nn.utils.rnn.pad_sequence(
            sequences=[torch.tensor(i) for i in x], 
            batch_first=batch_first,
            padding_value = self.tokenizer.vocab.pad_token_id
        )
        
        if return_pad_mask:
            pad_mask = nn.utils.rnn.pad_sequence(
                sequences=[torch.ones(len(i), dtype=int) for i in x], 
                batch_first=batch_first,
                padding_value = 0
            )
            return padded_batch, pad_mask
        
        return padded_batch
    
    @abstractmethod
    def fit():
        raise NotImplemented("Must implement fit() method.")
    
    @abstractmethod
    def predict():
        raise NotImplemented("Must implement predict() method.")

class RNNMaskedLM(RNNBaseLM):
    def fit(
        self, 
        dataset : Dataset, 
        loss_fn : torch.nn.Module, 
        optim : torch.optim.Optimizer, 
        n_epochs : int = 1, 
        batch_size : int = 32, 
        log_every : int = 50
    ) -> Model:

        device = self._get_device()
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.model = self.model.to(device) 

        for epoch in range(n_epochs):
            for i, batch in enumerate(loader):
                x = self.tokenizer.encode(batch["text"], add_special=True)
                # here we would mask some of the tokens
                # TODO
                x, pad_mask = self._pad_batch(x, batch_first=True, return_pad_mask=True)
                x = x.to(device)
                pad_mask = pad_mask.to(device)
                y = self.tokenizer.encode(batch["text"], add_special=True)
                y = self._pad_batch(y, batch_first=True).to(device)

                self.model.zero_grad()
                log_probas = self.model(x)

                loss = loss_fn(log_probas, y)
                if i and i % (log_every) == 0:
                    print(f"Step {i}: loss = {loss.item()}")
                    out = (log_probas.detach().clone().argmax(dim=1) * pad_mask).tolist()
                    print(self.tokenizer.decode(out, remove_special=True))
                loss.backward()
                optim.step()
        
        return self
    
    def predict(self, dataset : Dataset):
        output = []
        device = self._get_device()
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x = self.tokenizer.encode(batch["text"], add_special=True)
                x, pad_mask = self._pad_batch(x, batch_first=True, return_pad_mask=True)
                x = x.to(device)
                pad_mask = pad_mask.to(device)

                log_probas = self.model(x)
                out = (log_probas.detach().clone().argmax(dim=1) * pad_mask).tolist()
                output.extend(self.tokenizer.decode(out, remove_special=True))
                if i > 10:
                    break

        return output


class RNNLanguageModel(RNNBaseLM):
    def _shift_labels(self, labels : List[List[int]]):
        # shifts labels one position left to predict next word in LM
        labels = [row[1:] + [self.tokenizer.vocab.pad_token_id] for row in labels]

        return labels

    def fit(
        self, 
        dataset : Dataset, 
        loss_fn : torch.nn.Module, 
        optim : torch.optim.Optimizer, 
        n_epochs : int = 1, 
        batch_size : int = 32, 
        log_every : int = 50
    ) -> Model:

        device = self._get_device()
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.model = self.model.to(device) 

        for epoch in range(n_epochs):
            for i, batch in enumerate(loader):
                x = self.tokenizer.encode(batch["text"], add_special=True)
                x, pad_mask = self._pad_batch(x, batch_first=True, return_pad_mask=True)
                x = x.to(device)
                pad_mask = pad_mask.to(device)
                y = self.tokenizer.encode(batch["text"], add_special=True)
                y = self._shift_labels(y)
                y = self._pad_batch(y, batch_first=True).to(device)

                self.model.zero_grad()
                log_probas = self.model(x)

                loss = loss_fn(log_probas, y)
                if i and i % (log_every) == 0:
                    print(f"Step {i}: loss = {loss.item()}")
                    out = (log_probas.detach().clone().argmax(dim=1) * pad_mask).tolist()
                    print(self.tokenizer.decode(out, remove_special=True))
                loss.backward()
                optim.step()
        
        return self