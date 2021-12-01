from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from src.models.model import Model
from typing import List, Iterator
from src.data.dataio import Dataset
from src.models.vocabulary import Vocabulary
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

class RNNMaskedLM(Model):
    def __init__(
        self, 
        vocab : Vocabulary, 
        embedding_dim : int = 300, 
        hidden_dim : int =  300, 
        num_layers : int = 2, 
        dropout : float = 0.1, 
        bidirectional : bool = False
    ) -> None:

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = vocab

        self.lstm = RNN(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    
    def parameters(self) -> Iterator[Parameter]:
        return self.lstm.parameters()

    def _get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

        self.lstm = self.lstm.to(device) 

        for epoch in range(n_epochs):
            for i, batch in enumerate(loader):
                x = self.vocab.encode(batch["text"], add_special=True)
                x = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in x], batch_first=True)
                x = x.to(device)
                y = self.vocab.encode(batch["labels"], add_special=True)
                y = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in y], batch_first=True)
                y = y.to(device)
                self.lstm.zero_grad()
                log_probas = self.lstm(x)


                loss = loss_fn(log_probas, y)
                if i and i % (log_every) == 0:
                    print(f"Step {i}: loss = {loss.item()}")
                    out = log_probas.detach().clone().argmax(dim=1).tolist()
                    print(self.vocab.decode(out))
                loss.backward()
                optim.step()
        
        return self
    
    def predict(
        self, 
        dataset : Dataset
    ):
        device = self._get_device()
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            log_probas = self.lstm(x)


        return y_pred

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                num_layers, dropout, bidirectional):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.hidden2token = nn.Linear(
            in_features=hidden_dim * (2 if bidirectional else 1), 
            out_features=vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x, _ = self.rnn(x)
        x = self.hidden2token(x)
        x = F.log_softmax(x, dim=-1)
        x = x.transpose(-1, 1)
        return x