import torch


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


class RNNClassifier(torch.nn.Module):
    def __init__(
        self,
        rnn_type : str = "lstm", 
        embeddings_layer : torch.nn.modules.sparse.Embedding = None,
        hidden_dim : int = 768, 
        num_layers : int = 1,
        output_size : int = 2,
        bidirectional : bool = False,
        dropout : float = 0.1,
        ext_feat_size : int = 0 # comes from outside and is concatenated to embeddings
    ) -> None:

        super().__init__()
        assert (rnn_type in ["lstm", "gru"]), "rnn_type can be one of: 'lstm', 'gru'."
        rnn_type = torch.nn.LSTM if rnn_type == "lstm" else torch.nn.GRU
        self.embeddings = embeddings_layer
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.rnn = rnn_type(
            input_size=self.embeddings.embedding_dim + ext_feat_size, 
            # input_size=self.embeddings.embedding_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, # prevents warning for 1 layer
            bidirectional=bidirectional,
            batch_first=True
        )

        self.dropout = torch.nn.Dropout(p=dropout)
        self.dense = torch.nn.Linear(
            in_features=hidden_dim * (2 if bidirectional else 1), 
            out_features=output_size)
        
    def forward(
        self, 
        input_ids : torch.Tensor = None, # [B, L]
        inputs_embeds : torch.Tensor = None # [B, L, H]
    ) -> torch.Tensor:
        assert (input_ids is None or inputs_embeds is None), "Can take either input_ids or inputs_embeds, not both."
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        
        _, (hidden, _) = self.rnn(inputs_embeds)
        hidden = self.dropout(hidden)

        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            # Only use last hidden state
            hidden = hidden[-1]
        output = self.dense(hidden)
        return output
