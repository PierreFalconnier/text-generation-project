from pathlib import Path
import torch
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(42)

if __name__ == "__main__":

    # Hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="shakespeare.txt")
    parser.add_argument("--mode", type=str, default="word")

    args = parser.parse_args()

    # IMPORTATIONS
    # include the path of the dataset(s) and the model(s)
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from Dataset.DatasetText import DatasetText as Dataset
    from Models.LSTM import LSTM
    from Models.RNN import RNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATASET
    folder_path = ROOT / "Data" / "txt" / args.dataset
    dataset = Dataset(
        folder_path=folder_path, sequence_length=args.sequence_length, mode=args.mode
    )

    # split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # DATALOADERS
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # LOGGER
    name = Path(__file__).name[:-3]
    name += (
        "_"
        + str(args.epochs)
        + "_"
        + str(args.batch_size)
        + "_"
        + str(args.sequence_length)
        + "_"
        + str(args.lr)[2:]
        + "_"
        + str(args.num_layers)
        + "_"
        + str(args.hidden_dim)
        + "_"
        + str(args.dataset)
    )
    LOG_DIR = ROOT / "Run" / "Results" / "Logs" / name
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"LOG_DIR: {LOG_DIR}")
    writer = SummaryWriter(log_dir=LOG_DIR)

    #   TRAIN
    model = LSTM(
        vocab_size=dataset.vocab_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()  # include softmax !
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state_dict = model.state_dict()
    SAVED_MODEL_DIR = ROOT / "Run" / "Results" / "Saved_models" / name
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0.0
        state_h, state_h_val, state_c, state_c_val = None, None, None, None

        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            if state_h is None or state_h.size(1) != x.size(0):
                # init state with current batch size
                state_h, state_c = model.init_state(x.size(0))
                state_h, state_c = state_h.to(device), state_c.to(device)

            if model.embedding_dim is None:
                x = F.one_hot(x, num_classes=model.vocab_size).float()

            optimizer.zero_grad()
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))

            # y_pred shape = B,  sequence_length, vocabsize
            # y_pred to B, vocabsize, sequence_length
            loss = criterion(y_pred.permute(0, 2, 1), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # avoid backprop across batchs
            state_h = state_h.detach()
            state_c = state_c.detach()

        # Validation, generation, logger
        model.eval()

        with torch.no_grad():
            val_loss = 0.0
            for x, y in val_dataloader:

                x, y = x.to(device), y.to(device)

                if state_h_val is None or state_h_val.size(1) != x.size(0):
                    state_h_val, state_c_val = model.init_state(x.size(0))
                    state_h_val, state_c_val = state_h_val.to(device), state_c_val.to(
                        device
                    )

                if model.embedding_dim is None:
                    x = F.one_hot(x, num_classes=model.vocab_size).float()

                y_pred, (state_h_val, state_c_val) = model(
                    x, (state_h_val, state_c_val)
                )

                loss = criterion(y_pred.permute(0, 2, 1), y)
                optimizer.step()
                val_loss += loss.item()

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)

        # early stopping
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict().copy()
            torch.save(model.state_dict(), SAVED_MODEL_DIR / "best_model.pt")

        list_text = model.generate(
            dataset,
            device=device,
            text="Where are you?",
            total_length=100,
            temperature=args.temperature,
        )
        text = " ".join(list_text)
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_text("text_generation", text, epoch)

    # eval on the test set

    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()
    state_h_val, state_c_val = None, None
    with torch.no_grad():
        test_loss = 0.0
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            if state_h_val is None or state_h_val.size(1) != x.size(0):
                state_h_val, state_c_val = model.init_state(x.size(0))
                state_h_val, state_c_val = state_h_val.to(device), state_c_val.to(
                    device
                )
            if model.embedding_dim is None:
                x = F.one_hot(x, num_classes=model.vocab_size).float()
            y_pred, (state_h_val, state_c_val) = model(x, (state_h_val, state_c_val))
            loss = criterion(y_pred.permute(0, 2, 1), y)
            optimizer.step()
            test_loss += loss.item()
    test_loss /= len(test_dataloader)

    print(f"Test loss = {test_loss}")
