from pathlib import Path
import torch
import sys
import argparse
import re
import string
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from misspelling_percentage import calculate_misspelling_percentage
from utils import WarmupCosineDecayScheduler, configure_optimizers
import math

torch.manual_seed(42)


if __name__ == "__main__":

    # Hyperparams
    # default used by nanoGPT for baby gpt on shakespeare
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="harry_potter.txt")
    parser.add_argument("--mode", type=str, default="character")
    parser.add_argument("--word2vec", type=bool, default=False)
    parser.add_argument("--use_bpe", type=bool, default=False)
    parser.add_argument("--bpe_vocab_size", type=int, default=10000)

    parser.add_argument("--log-interval-train", type=int, default=10)
    parser.add_argument("--log-interval-val", type=int, default=250)

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
    from Models.Transformer import TransformerModel

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
        use_bpe=args.use_bpe,
        bpe_vocab_size=args.bpe_vocab_size,
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
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # LOGGER
    name = Path(__file__).name[:-3]
    name += (
        "_"
        + str(args.max_iters)
        + "_"
        + str(args.batch_size)
        + "_"
        + str(args.sequence_length)
        + "_"
        + str(args.lr)[2:]
        + "_"
        + str(args.num_layers)
        + "_"
        + str(args.num_heads)
        + "_"
        + str(args.dataset[:-4])
        + "_"
        + str(args.mode)
        + "_"
        + str(args.word2vec)
        + "_"
        + str(args.use_bpe)
    )
    LOG_DIR = ROOT / "Run" / "Results" / "Logs" / name
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"LOG_DIR: {LOG_DIR}")
    writer = SummaryWriter(log_dir=LOG_DIR)

    hparams = {
        "lr": args.lr,
        "max_iters": args.max_iters,
        "batch_size": args.batch_size,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "architecture": "Transformer",
    }

    #   TRAIN
    model = TransformerModel(
        dataset=dataset,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # print(model)
    print("Trainable parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss()  # softmax already included !

    # OPTIMIZER
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.99
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

    optimizer = configure_optimizers(
        model=model,
        weight_decay=weight_decay,
        learning_rate=args.lr,
        betas=(beta1, beta2),
        device_type=device,
    )

    # SCHEDULER
    # learning rate decay settings
    warmup_iters = 100  # how many steps to warm up for
    lr_decay_iters = args.max_iters
    min_lr = args.lr / 10

    scheduler = WarmupCosineDecayScheduler(
        optimizer=optimizer,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
        min_lr=min_lr,
    )

    # TRAINING

    best_val_loss = float("inf")
    best_state_dict = model.state_dict()
    SAVED_MODEL_DIR = ROOT / "Run" / "Results" / "Saved_models" / name
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    early_stopping_tol = 750  # num of steps
    prev_val_loss = float("inf")
    counter = 0

    step = 0
    model.train()

    first_val_batch = next(iter(val_dataloader))

    with tqdm(total=args.max_iters) as pbar:

        while step < args.max_iters:

            for x, y in train_dataloader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                y_pred = model(x)

                # y_pred shape = B,  sequence_length, vocabsize
                # y_pred to B, vocabsize, sequence_length
                loss = criterion(y_pred.permute(0, 2, 1), y)
                loss.backward()

                # clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                scheduler.step()

                # LOG TRAIN LOSS
                if step % args.log_interval_train == 0:
                    writer.add_scalar(
                        "train_loss_step",
                        loss.item(),
                        step,
                    )
                    writer.add_scalar(
                        "Learning-rate", optimizer.param_groups[0]["lr"], step
                    )

                # VALIDATION, GENERATION, LOG
                if step % args.log_interval_val == 0:

                    model.eval()

                    # val loss on the first val batch

                    with torch.no_grad():
                        x, y = first_val_batch
                        x, y = x.to(device), y.to(device)
                        if model.embedding_dim is None:
                            x = F.one_hot(x, num_classes=dataset.vocab_size).float()
                        y_pred = model(x)
                        loss = criterion(y_pred.permute(0, 2, 1), y)
                        val_loss = loss.item()

                    # generation
                    init_text = '"Where are you?"'
                    list_text = model.generate(
                        dataset,
                        device=device,
                        text=init_text,
                        total_length=10000,
                        temperature=args.temperature,
                    )
                    if args.use_bpe:
                        text = list_text
                    else:
                        text = joiner_str.join(list_text[len(init_text) :])
                    misspelling_percentage = calculate_misspelling_percentage(text)

                    # logs
                    writer.add_scalar(
                        "val_loss_step",
                        val_loss,
                        step,
                    )
                    writer.add_scalars(
                        "misspelling_percentage",
                        {"misspelling_percentage": misspelling_percentage},
                        step,
                    )
                    writer.add_text("text_generation", text, step)

                    # early stopping
                    if best_val_loss > val_loss:
                        best_val_loss = val_loss
                        counter = 0
                        best_misspelling_percentage = misspelling_percentage
                        best_state_dict = model.state_dict().copy()
                        torch.save(
                            model.state_dict(), SAVED_MODEL_DIR / "best_model.pt"
                        )
                    else:
                        counter += 1

                    # if counter == early_stopping_tol:
                    #     print("stopped early")
                    #     break

                    prev_val_loss = val_loss

                    model.train()

                step += 1
                pbar.update(1)

    # once training is over
    torch.save(model.state_dict(), SAVED_MODEL_DIR / "last_model.pt")

    # eval on the test set
    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()
    state_h_val, state_c_val = None, None
    with torch.no_grad():
        test_loss = 0.0
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)

            if model.embedding_dim is None:
                x = F.one_hot(x, num_classes=dataset.vocab_size).float()
            y_pred = model(x)
            loss = criterion(y_pred.permute(0, 2, 1), y)
            test_loss += loss.item()
    test_loss /= len(test_dataloader)

    with open(str(ROOT / "Run" / "Results" / "Saved_results" / name), "w") as file:
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
