import math
import inspect
from dataclasses import dataclass
from Dataset.utils import custom_tokenizer

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


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

    def forward(self, size):
        return self.pe[:, :size, :]


class TransformerModel(nn.Module):
    def __init__(
        self,
        dataset,
        nhead=8,
        dropout=0.1,
        activation="gelu",
        num_layers=1,
        pos_encoding="learnt",
    ):
        super(TransformerModel, self).__init__()

        assert pos_encoding in ["learnt", "attention_is_all_you_need"]
        self.pos_encoding = pos_encoding
        self.dataset = dataset
        self.embedding_dim = dataset.embedding_dim
        self.nhead = nhead
        self.drop_out = dropout
        self.num_layers = num_layers
        self.activation = activation

        # EMBEDDING (MODEL DIM)
        if self.embedding_dim is not None:
            if dataset.word2vec:
                embedding_weights = np.zeros((dataset.vocab_size, self.embedding_dim))
                for idx, word in dataset.index_to_word.items():
                    embedding_weights[idx] = dataset.wv[word]
                self.embedding = nn.Embedding.from_pretrained(
                    torch.FloatTensor(embedding_weights), freeze=True
                )
                d_model = self.embedding_dim
            else:
                # Learned embedding
                self.embedding = nn.Embedding(
                    num_embeddings=dataset.vocab_size,
                    embedding_dim=self.embedding_dim,
                )
                d_model = self.embedding_dim
        else:
            # input will be one-hot encoded
            d_model = dataset.vocab_size

        self.d_model = d_model  # embedding size

        if pos_encoding == "learnt":
            self.wpe = nn.Embedding(dataset.sequence_length, d_model)
        elif pos_encoding == "attention_is_all_you_need":
            self.wpe = PositionalEncoding(d_model, dropout, dataset.sequence_length)
        else:
            raise NotImplementedError

        self.decoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

        self.transformer_decoder = TransformerEncoder(self.decoder_layer, num_layers)
        self.ln_f = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, dataset.vocab_size, bias=False)

        self.embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):

        device = x.device
        b, t = x.size()
        assert (
            t <= self.dataset.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.data.sequence_length}"

        # forward

        # word embeddings
        tok_emb = (
            self.embedding(x)
            if self.embedding_dim is not None
            else F.one_hot(x, num_classes=self.dataset.vocab_size).float()
        )  # shape (b, t, n_embd)

        if self.pos_encoding == "attention_is_all_you_need":
            pos_emb = self.wpe(size=t)
        elif self.pos_encoding == "learnt":
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
            pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)

        # Pass the input and causal mask to the TransformerDecoder
        causal_mask = nn.Transformer.generate_square_subsequent_mask(t, device=device)
        embed = self.transformer_decoder(
            src=tok_emb + pos_emb,
            mask=causal_mask,
            is_causal=True,
        )
        embed = self.ln_f(embed)
        logits = self.lm_head(embed)
        return logits

    @torch.no_grad()
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

        # TOKENIZE INPUT TEXT
        if dataset.use_bpe:
            words = dataset.bpe_model.encode(text, out_type=str)
        else:
            if dataset.mode == "word":
                words = custom_tokenizer(text)
            elif dataset.mode == "character":
                words = list(text)
            else:
                raise NotImplementedError

        idx = torch.tensor([[dataset.word_to_index[w] for w in words]], device=device)

        # GENERATION
        for _ in range(total_length):
            x = idx[:, -self.dataset.sequence_length :]

            y_pred = self(x)

            last_word_logits = y_pred[0, -1, :] / temperature
            probs = F.softmax(last_word_logits, dim=-1)

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

            idx = torch.cat((idx, torch.tensor([[word_index]], device=device)), dim=1)
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
        sequence_length=256,
        mode="character",
        embedding_dim=384,
    )
    datalaoder = DataLoader(dataset, batch_size=4)

    #   TRAIN
    model = TransformerModel(
        dataset=dataset, nhead=6, num_layers=6, pos_encoding="attention_is_all_you_need"
    ).to(device)

    # inference
    x, _ = next(iter(datalaoder))
    x = x.to(device)

    y = model(x)
    print(y)
    print(y.shape)
    print(x.shape)
    exit()

    # generation
    list_text = model.generate(
        dataset,
        device=device,
        text="This is a test to make sure that",
        total_length=100,
    )
    if dataset.use_bpe:
        print(list_text)
    else:
        print("".join(list_text))

    # number params
    print("Trainable parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
