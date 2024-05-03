import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=100,
        embedding_dim=None,
        num_layers=1,
        dropout=0.0,
        nonlinearity="tanh",
    ):
        super(RNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if embedding_dim is not None:
            # embedding is learnt
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=self.embedding_dim,
            )
            input_size = embedding_dim
        else:
            # one hot encodding
            input_size = vocab_size

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=dropout,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x) if self.embedding_dim is not None else x
        output, state = self.rnn(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)
