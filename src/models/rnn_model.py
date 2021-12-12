import torch
from torch import nn
from tqdm import tqdm
from abc import abstractmethod
from src.models.model import Model
from src.data.dataio import Dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from typing import Union, List, Iterator, Tuple
from transformers.tokenization_utils import PreTrainedTokenizer


class RNN(nn.Module):
    def __init__(
        self, 
        input_size : int = 30522, # ignored if embeddings_layer is not None, instead inferred from embeddings_layer
        embedding_dim : int = 768, # ignored if embeddings_layer is not None, instead inferred from embeddings_layer
        output_size : int = 30522,
        embeddings_layer : torch.nn.modules.sparse.Embedding = None,
        rnn_type : str = "lstm", 
        hidden_dim : int = 768, 
        num_layers : int = 1,
        dropout : float = 0.1,
        bidirectional : bool = False
    ) -> None:
        super().__init__()
        if embeddings_layer is not None:
            input_size, embedding_dim = embeddings_layer.weight.shape
            self.embeddings = embeddings_layer
        else:
            assert (input_size > 0 and embedding_dim > 0), "Negative input_size or embedding_dim."
            self.embeddings = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)

        assert (rnn_type in ["lstm", "gru"]), "rnn_type can be one of: 'lstm', 'gru'."

        rnn_type = nn.LSTM if rnn_type == "lstm" else nn.GRU

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        self.rnn = rnn_type(
            input_size=embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, # prevents warning for 1 layer
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dense = nn.Linear(in_features=hidden_dim * (2 if bidirectional else 1), out_features=output_size)
        
    def forward(
        self, 
        input_ids : torch.Tensor = None, # [B, L]
        inputs_embeds : torch.Tensor = None, # [B, L, H]
        ret_pre_dense : bool = False
    ) -> torch.Tensor:

        assert (input_ids is None or inputs_embeds is None), "Can take either input_ids or inputs_embeds, not both."
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        x, _ = self.rnn(inputs_embeds)
        if self.dropout:
            x = self.dropout(x)
        # print("Pre-dense", x.shape)
        out = self.dense(x)
        out = F.log_softmax(out, dim=-1)
        # print("Post-dense", out.shape)
        out = out.transpose(-1, 1)
        # print("Post-dense.T", out.shape)
        if ret_pre_dense:
            return out, x
        else:
            return out


class RNNBaseLM(Model):
    def __init__(
        self,
        tokenizer : PreTrainedTokenizer,
        embeddings_layer : torch.nn.modules.sparse.Embedding = None,
        rnn_type : str = "lstm",
        embedding_dim : int = 768, 
        hidden_dim : int =  768, 
        num_layers : int = 1, 
        dropout : float = 0.1, 
        bidirectional : bool = False,
        tie_weights : bool = True
    ) -> None:

        self.tokenizer = tokenizer
        self.model = RNN(
            input_size=len(tokenizer.vocab),
            embedding_dim=embedding_dim, 
            output_size=len(tokenizer.vocab),
            embeddings_layer=embeddings_layer,
            rnn_type=rnn_type,
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
        masking : str = None, # TODO add masking strategies
        n_epochs : int = 1, 
        batch_size : int = 32, 
        log_every : int = 50
    ) -> Model:

        device = self._get_device()
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.model = self.model.to(device) 

        for epoch in range(n_epochs):
            for i, batch in enumerate(loader):
                inputs = self.tokenizer(
                    text=batch["text"],
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                targets = input_ids.detach().clone()
                # TODO
                # here we would mask some of the tokens

                self.model.zero_grad()
                log_probas = self.model(input_ids)

                loss = loss_fn(log_probas, targets)
                if i and i % (log_every) == 0:
                    print(f"Step {i}: loss = {loss.item()}")
                    out = (log_probas.detach().clone().argmax(dim=1) * attention_mask).tolist()
                    print(self.tokenizer.decode(out, remove_special=True))
                loss.backward()
                optim.step()
        
        return self
    
    def predict(self, dataset : Dataset, batch_size : int = 32):
        output = []
        device = self._get_device()
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                inputs = self.tokenizer(
                    text=batch["text"],
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                log_probas = self.model(input_ids)
                out = (log_probas.detach().clone().argmax(dim=1) * attention_mask).tolist()
                decoded_batch = self.tokenizer.batch_decode(out, skip_special_tokens=True)
                output.extend(decoded_batch)

        return output


class RNNLanguageModel(RNNBaseLM):
    def _shift_labels(self, labels : torch.Tensor) -> torch.Tensor:
        # shifts labels one position left to predict next word in LM
        batch_size = labels.shape[0]
        labels = torch.cat(
            tensors=[labels[:, 1:], torch.ones(size=(batch_size, 1), dtype=labels.dtype)], 
            dim=1
        )
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
                inputs = self.tokenizer(
                    text=batch["text"],
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                targets = input_ids.detach().clone()
                targets = self._shift_labels(labels=targets)

                self.model.zero_grad()
                log_probas = self.model(input_ids)

                loss = loss_fn(log_probas, targets)
                if i and i % (log_every) == 0:
                    print(f"Step {i}: loss = {loss.item()}")
                    out = (log_probas.detach().clone().argmax(dim=1) * attention_mask).tolist()
                    print(self.tokenizer.batch_decode(out, skip_special_tokens=True))
                loss.backward()
                optim.step()
        
        return self

    def predict(self, dataset : Dataset):
        device = self._get_device()
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        output = []
        # since we do prediction sequentially we cannot have a batch size > 1
        with torch.no_grad():
            # TODO implement sequential predictions
            for batch in loader:
                inputs = self.tokenizer(
                    text=batch["text"],
                    return_attention_mask=False,
                    return_tensors="pt"
                )
                input_ids = inputs["input_ids"].to(device)
                mask_token_ids = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[-1]
                for mask_token_id in mask_token_ids:
                    _input_ids = input_ids[:, :mask_token_id]
                    _input_ids = _input_ids.to(device)

                    log_proba = self.model(_input_ids)[:, -1] # take the last output of RNN
                    predicted_token_id = log_proba.argmax().item()
                    # replace mask token with predicted token
                    input_ids[0, mask_token_id] = predicted_token_id

                output.append(
                    self.tokenizer.batch_decode(
                        input_ids.detach().clone().tolist(), skip_special_tokens=True)[0]
                )

        return output