import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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

    def generate(self, dataset, device, text, total_length=100):
        self.eval()

        words = text.split(" ")
        state_h = self.init_state(1).to(device)

        for i in range(0, total_length):
            x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]]).to(device)

            if self.embedding_dim is None:
                x = F.one_hot(x, num_classes=self.vocab_size).float()

            y_pred, state_h = self(x, state_h)  # B,  sequence_length, vocabsize

            last_word_logits = y_pred[0][-1]
            p = (
                torch.nn.functional.softmax(last_word_logits, dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(dataset.index_to_word[word_index])

        return words


if __name__ == "__main__":
    import sys
    from pathlib import Path

    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from Dataset.DatasetShakespeare import DatasetShakespeare as Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATASET
    folder_path = ROOT / "Data" / "shakespeare"
    dataset = Dataset(
        folder_path=folder_path,
        sequence_length=5,
    )

    #   TRAIN
    model = RNN(
        vocab_size=dataset.vocab_size,
        hidden_dim=100,
        embedding_dim=None,
        num_layers=1,
        dropout=0.0,
        nonlinearity="tanh",
    ).to(device)

    list_text = model.generate(
        dataset,
        device=device,
        text="This is a test to make sure that",
        total_length=100,
    )
    print(" ".join(list_text))
