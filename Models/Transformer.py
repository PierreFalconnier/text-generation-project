"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class TransformerModel(nn.Module):
    def __init__(
        self,
        dataset,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        num_layers=1,
    ):
        super(TransformerModel, self).__init__()

        self.embedding_dim = dataset.embedding_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.drop_out = dropout
        self.num_layers = num_layers
        self.activation = activation

        # EMBEDDING (MODEL DIM)
        if self.embedding_dim is not None:
            if dataset.word2vec:
                embedding_weights = torch.zeros(
                    (dataset.vocab_size, self.embedding_dim)
                )
                for idx, word in dataset.index_to_word.items():
                    embedding_weights[idx] = dataset.wv[word]
                self.embedding = nn.Embedding.from_pretrained(
                    torch.FloatTensor(embedding_weights), freeze=True
                )
                input_size = self.embedding_dim
            else:
                # Learned embedding
                self.embedding = nn.Embedding(
                    num_embeddings=dataset.vocab_size,
                    embedding_dim=self.embedding_dim,
                )
                input_size = self.embedding_dim
        else:
            # Assume input is already one-hot encoded
            input_size = dataset.vocab_size

        self.input_size = input_size

        self.decoder_layer = TransformerDecoderLayer(
            d_model=input_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, num_layers)

    def forward(self, x):
        return self.transformer_decoder(x)
