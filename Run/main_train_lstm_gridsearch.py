from pathlib import Path
import torch
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from misspelling_percentage import calculate_misspelling_percentage

torch.manual_seed(42)

if __name__ == "__main__":

    # Hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="harry_potter.txt")
    parser.add_argument("--mode", type=str, default="character")
    parser.add_argument("--word2vec", type=bool, default=False)
    parser.add_argument("--log-interval", type=int, default=10)

    args = parser.parse_args()
    if args.embedding_dim == 0:
        args.embedding_dim = None
    print(args)

    # IMPORTATIONS
    # include the path of the dataset(s) and the model(s)
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from Dataset.DatasetText import DatasetText as Dataset
    from Models.LSTM import LSTM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATASET
    folder_path = ROOT / "Data" / "txt" / args.dataset
    dataset = Dataset(
        folder_path=folder_path,
        sequence_length=args.sequence_length,
        mode=args.mode,
        word2vec=args.word2vec,
        embedding_dim=args.embedding_dim,
    )
    if args.mode == "word":
        joiner_str = " "  # more post-processing will be needed
    elif args.mode == "character":
        joiner_str = ""

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
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=6,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # GRID SEARCH
    learning_rates = [1e-4, 1e-3, 1e-2]
    batch_sizes = [32, 128, 512, 2048]

    # learning_rates = [1e-3]
    # batch_sizes = [1, 8]

    best_val_loss_list = []

    for lr in learning_rates:
        for batch_size in batch_sizes:

            args.lr = lr
            args.batch_size = batch_size

            # LOGGER
            # name = Path(__file__).name[:-3]
            name = "main_train_lstm"
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
                + str(args.dataset[:-4])
                + "_"
                + str(args.word2vec)
            )
            LOG_DIR = ROOT / "Run" / "Results" / "Logs" / name
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            print(f"LOG_DIR: {LOG_DIR}")
            writer = SummaryWriter(log_dir=LOG_DIR)

            hparams = {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "architecture": "LSTM",
            }

            #   TRAIN
            model = LSTM(
                dataset=dataset,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ).to(device)

            print(model)
            print("Trainable parameters:")
            print(sum(p.numel() for p in model.parameters() if p.requires_grad))

            criterion = nn.CrossEntropyLoss()  # include softmax !
            if args.optim == "adagrad":
                optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
            elif args.optim == "adam":
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
            else:
                raise NotImplementedError

            best_val_loss = float("inf")
            best_state_dict = model.state_dict()
            SAVED_MODEL_DIR = ROOT / "Run" / "Results" / "Saved_models" / name
            SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

            early_stopping_tol = 2  # num of epochs
            prev_val_loss = float("inf")
            counter = 0

            for epoch in tqdm(range(args.epochs)):
                model.train()
                train_loss = 0.0

                for step, (x, y) in enumerate(train_dataloader):
                    x, y = x.to(device), y.to(device)

                    state_h, state_c = model.init_state(x.size(0))
                    state_h, state_c = state_h.to(device), state_c.to(device)

                    if model.embedding_dim is None:
                        x = F.one_hot(x, num_classes=dataset.vocab_size).float()

                    optimizer.zero_grad()
                    y_pred, (state_h, state_c) = model(x, (state_h, state_c))

                    # y_pred shape = B,  sequence_length, vocabsize
                    # y_pred to B, vocabsize, sequence_length
                    loss = criterion(y_pred.permute(0, 2, 1), y)
                    loss.backward()

                    # # clip gradients
                    # max_norm = 5.0
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    optimizer.step()
                    train_loss += loss.item()

                    if step % args.log_interval == 0:
                        writer.add_scalar(
                            "train_loss_step",
                            loss.item(),
                            epoch * len(train_dataloader) + step,
                        )

                # Validation, generation, logger
                model.eval()

                with torch.no_grad():
                    val_loss = 0.0
                    for x, y in val_dataloader:

                        x, y = x.to(device), y.to(device)

                        state_h_val, state_c_val = model.init_state(x.size(0))
                        state_h_val, state_c_val = state_h_val.to(
                            device
                        ), state_c_val.to(device)

                        if model.embedding_dim is None:
                            x = F.one_hot(x, num_classes=dataset.vocab_size).float()

                        y_pred, (state_h_val, state_c_val) = model(
                            x, (state_h_val, state_c_val)
                        )

                        loss = criterion(y_pred.permute(0, 2, 1), y)
                        val_loss += loss.item()

                train_loss /= len(train_dataloader)
                val_loss /= len(val_dataloader)

                init_text = "Where are you?"
                list_text = model.generate(
                    dataset,
                    device=device,
                    text=init_text,
                    total_length=10000,
                    temperature=args.temperature,
                )
                text = joiner_str.join(list_text[len(init_text) :])
                misspelling_percentage = calculate_misspelling_percentage(text)

                writer.add_scalars(
                    "loss", {"train": train_loss, "val": val_loss}, epoch
                )
                writer.add_scalars(
                    "misspelling_percentage",
                    {"misspelling_percentage": misspelling_percentage},
                    epoch,
                )
                writer.add_text("text_generation", text, epoch)

                # early stopping
                # stop if no amelioration for early_stopping_tol epochs
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1

                if counter == early_stopping_tol:
                    print("stopped early")
                    break

                prev_val_loss = val_loss

            # once training is over
            best_misspelling_percentage = misspelling_percentage
            best_state_dict = model.state_dict().copy()
            torch.save(model.state_dict(), SAVED_MODEL_DIR / "best_model.pt")

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
                        state_h_val, state_c_val = state_h_val.to(
                            device
                        ), state_c_val.to(device)
                    if model.embedding_dim is None:
                        x = F.one_hot(x, num_classes=dataset.vocab_size).float()
                    y_pred, (state_h_val, state_c_val) = model(
                        x, (state_h_val, state_c_val)
                    )
                    loss = criterion(y_pred.permute(0, 2, 1), y)
                    test_loss += loss.item()
            test_loss /= len(test_dataloader)

            with open(
                str(ROOT / "Run" / "Results" / "Saved_results" / name), "w"
            ) as file:
                print(f"Best val loss:", file=file)
                print(best_val_loss, file=file)
                print(f"Test loss:", file=file)
                print(test_loss, file=file)

            # Log hyperparameters and metrics
            writer.add_hparams(
                hparam_dict=hparams,
                metric_dict={
                    "hparam/best_val_loss": best_val_loss,
                    "hparam/test_loss": test_loss,
                    "hparam/best_misspelling_percentage": best_misspelling_percentage,
                },
            )

            best_val_loss_list.append(best_val_loss)

            writer.close()

    print(best_val_loss_list)
    print(best_val_loss_list.index(max(best_val_loss_list)))
