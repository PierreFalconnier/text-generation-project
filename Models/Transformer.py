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
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


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

        # NEED TO ADD A POSITION REPRESENTATION SOMEWHERE, IN THE DATASET CLASS?

        # EMBEDDING (MODEL DIM)
        if self.embedding_dim is not None:
            if dataset.word2vec:
                embedding_weights = np.zeros((dataset.vocab_size, self.embedding_dim))
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

        self.positional_encoding = PositionalEncoding(input_size, dropout)

        self.decoder_layer = TransformerEncoderLayer(
            d_model=input_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_decoder = TransformerEncoder(self.decoder_layer, num_layers)
        self.fc = nn.Linear(input_size, dataset.vocab_size)

    def forward(self, x):
        embed = self.embedding(x) if self.embedding_dim is not None else x
        embed = self.positional_encoding(embed)
        embed = self.transformer_decoder(embed)
        logits = self.fc(embed)
        return logits

    def generate(
        self,
        dataset,
        device,
        text,
        total_length=1000,
        temperature=1.0,
        top_p=0.9,
        nucleus_sampling=False,
    ):
        self.eval()

        # Tokenize input text using BPE if enabled
        if dataset.use_bpe:
            words = dataset.bpe_model.encode(text, out_type=str)
        else:
            if dataset.mode == "word":
                if dataset.word2vec:
                    words = dataset.sentences.custom_tokenizer(text)
                else:
                    words = text.split()
            elif dataset.mode == "character":
                words = list(text)
            else:
                raise NotImplementedError

        for i in range(total_length):
            x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(device)

            if self.embedding_dim is None:
                x = F.one_hot(x, num_classes=dataset.vocab_size).float()

            y_pred = self(x)

            last_word_logits = y_pred[0][-1] / temperature
            probs = torch.nn.functional.softmax(last_word_logits, dim=0)

            if nucleus_sampling:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)

                if torch.all(cumulative_probs > top_p):
                    sorted_probs.fill_(0.0)
                    sorted_probs[0] = 1.0
                else:
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_probs[sorted_indices_to_remove] = 0.0

                sorted_probs /= sorted_probs.sum()
                sampled_index = torch.multinomial(sorted_probs, 1).item()
                word_index = sorted_indices[sampled_index].item()

            else:
                word_index = torch.multinomial(probs, 1).item()

            words.append(dataset.index_to_word[word_index])

        # Decode BPE tokens back to text if BPE was used
        if dataset.use_bpe:
            words = dataset.bpe_model.decode(words)

        return words


if __name__ == "__main__":
    import sys
    from pathlib import Path

    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from Dataset.DatasetText import DatasetText as Dataset
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATASET
    folder_path = ROOT / "Data" / "txt" / "harry_potter.txt"

    dataset = Dataset(
        folder_path=folder_path,
        sequence_length=25,
        mode="word",
        # word2vec=True,
        embedding_dim=384,
        # use_bpe=True,
        # bpe_vocab_size=5000,
    )
    datalaoder = DataLoader(dataset, batch_size=2)

    #   TRAIN
    model = TransformerModel(dataset=dataset, nhead=6, num_layers=6).to(device)

    list_text = model.generate(
        dataset,
        device=device,
        text="This is a test to make sure that",
        total_length=100,
        nucleus_sampling=0.5,
    )
    if dataset.use_bpe:
        print(list_text)
    else:
        print(" ".join(list_text))

    # print(model)
    print("Trainable parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    #

    x, _ = next(iter(datalaoder))
    x = x.to(device)

    y = model(x)

    print(y)
    print(y.shape)
