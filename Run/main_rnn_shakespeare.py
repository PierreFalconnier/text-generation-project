from pathlib import Path
import torch
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

if __name__ == "__main__":

    # Hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    # IMPORTATIONS
    # include the path of the dataset(s) and the model(s)
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from Dataset.DatasetShakespeare import DatasetShakespeare as Dataset
    from Models.RNN import RNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATASET
    folder_path = ROOT / "Data" / "shakespeare"
    dataset = Dataset(
        folder_path=folder_path,
        sequence_length=args.sequence_length,
    )

    # DATALOADER
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # LOGGER
    name = Path(__file__).name[:-3]
    name += (
        "_"
        + str(args.epochs)
        + "_"
        + str(args.batch_size)
        + "_"
        + str(args.sequence_length)
    )
    LOG_DIR = ROOT / "Run" / "Results" / name
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)

    #   TRAIN
    model = RNN(
        vocab_size=dataset.vocab_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        nonlinearity="tanh",
    ).to(device)
    criterion = nn.CrossEntropyLoss()  # include softmax !
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0.0
        state_h = None

        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)

            if state_h is None or state_h.size(1) != x.size(0):
                # init state with current batch size
                state_h = model.init_state(x.size(0)).to(device)

            if model.embedding_dim is None:
                x = F.one_hot(x, num_classes=model.vocab_size).float()

            optimizer.zero_grad()
            y_pred, state_h = model(x, state_h)

            # y_pred shape = B,  sequence_length, vocabsize
            # y_pred to B, vocabsize, sequence_length
            loss = criterion(y_pred.permute(0, 2, 1), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # avoid backprop across batchs
            state_h = state_h.detach()

        # Validation, generation, logger
        model.eval()
        list_text = model.generate(
            dataset,
            device=device,
            text="What is your name?",
            total_length=50,
        )
        text = " ".join(list_text)
        writer.add_scalars("loss", {"train": train_loss}, epoch)
        writer.add_text("text_generation", text, epoch)
