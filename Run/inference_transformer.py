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
from utils import WarmupCosineDecayScheduler, configure_optimizers
import math


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
    parser.add_argument("--dataset", type=str, default="harry_potter.txt")
    parser.add_argument("--mode", type=str, default="character")
    parser.add_argument("--word2vec", type=bool, default=False)
    parser.add_argument("--use_bpe", type=bool, default=False)
    parser.add_argument("--bpe_vocab_size", type=int, default=10000)
    parser.add_argument("--pos-encoding", type=str, default="learnt")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--nucleous", type=bool, default=False)
    parser.add_argument("--top-p", type=float, default=0.0)

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

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=6
    )

    # LOGGER
    # find the model path
    name = ROOT / "Run" / "Results" / "Saved_models" / "main_train_transformer"
    name = str(name)
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
        # + "_"
        # + str(args.pos_encoding)
    )

    #   TRAIN
    model = TransformerModel(
        dataset=dataset,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pos_encoding=args.pos_encoding,
    ).to(device)

    SAVED_MODEL_DIR = ROOT / "Run" / "Results" / "Saved_models" / name
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_state_dict = torch.load(SAVED_MODEL_DIR / "last_model.pt")

    # eval on the test set
    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()

    # INFERENCE

    init_text = '"What the hell?"'
    list_text = model.generate(
        dataset,
        device=device,
        text=init_text,
        total_length=10000,
        temperature=1,
        # nucleus_sampling=True,
        # top_p=0.6,
        # temperature=args.temperature,
        # nucleus_sampling=args.nucleous,
        # top_p=args.top_p,
    )
    text = "".join(list_text)
    misspelling_percentage = calculate_misspelling_percentage(text)

    print(text)
    print(f"\nMisspelling percentage: {misspelling_percentage}")
