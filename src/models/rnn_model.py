from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from src.models.model import Model
from typing import List
from src.data.dataio import Dataset
from torch.utils.data import DataLoader


class RNNModel(Model):
    def __init__(self, vocab, embedding_dim=768, hidden_dim=768, num_layers=2, dropout=0.1, bidirectional=False) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = vocab

        self.lstm = RNNLanguageModel(
            vocab=vocab,
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    
    def fit(self, dataset: Dataset, tokenizer, loss_fn, optim, n_epochs, batch_size=32, verbose=False) -> Model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.lstm = self.lstm.to(device) 

        for epoch in range(n_epochs):
            for i, batch in enumerate(loader):
                x = tokenizer(batch["text"], padding=True, return_attention_mask=False, return_tensors="pt")["input_ids"]
                x = x.to(device)
                y = tokenizer(batch["labels"], padding=True, return_attention_mask=False, return_tensors="pt")["input_ids"]
                y = y.to(device)
                self.lstm.zero_grad()
                print(y)
                log_probas = self.lstm(x)

                out = log_probas.detach().clone().argmax(dim=1)
                print(tokenizer.batch_decode(out))

                loss = loss_fn(log_probas, y)
                loss.backward()
                optim.step()
        
        return self
    
    def predict(self, x):
        with torch.no_grad():
            log_probas = self.lstm(x)


        return y_pred

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab, embedding_dim, hidden_dim, 
                num_layers, dropout, bidirectional):
        super().__init__()

        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.hidden2token = nn.Linear(hidden_dim, len(vocab))

    def forward(self, x):
        x = self.embeddings(x)
        x, _ = self.rnn(x)
        x = self.hidden2token(x)
        x = F.log_softmax(x, dim=-1)
        x = x.transpose(-1, 1)
        return x