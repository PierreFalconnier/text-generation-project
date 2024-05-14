if __name__ == "__main__":
    import torch
    import sys
    import argparse
    import torch.nn.functional as F
    from pathlib import Path
    from misspelling_percentage import calculate_misspelling_percentage

    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--nucleous", type=bool, default=False)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="harry_potter.txt")
    parser.add_argument("--dropout", type=float, default=0)

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
    from Models.RNN import RNN

    # find the model path
    name = ROOT / "Run" / "Results" / "Saved_models" / "main_train_rnn"
    name = str(name)
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

    SAVED_MODEL_DIR = Path(name) / "best_model.pt"

    if not SAVED_MODEL_DIR.exists():
        raise NotImplementedError(f"The file {SAVED_MODEL_DIR} does not exist.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATASET
    folder_path = ROOT / "Data" / "txt" / args.dataset
    mode = "character"
    dataset = Dataset(folder_path=folder_path, sequence_length=100, mode=mode)

    # MODEL
    model = RNN(
        vocab_size=dataset.vocab_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        nonlinearity="tanh",
    ).to(device)

    state_dict = torch.load(SAVED_MODEL_DIR)
    model.load_state_dict(state_dict)

    # INFERENCE

    init_text = "Where are you?"
    list_text = model.generate(
        dataset,
        device=device,
        text=init_text,
        total_length=10000,
        temperature=args.temperature,
        nucleus_sampling=args.nucleous,
        top_p=args.top_p,
    )
    text = "".join(list_text[len(init_text) :])
    misspelling_percentage = calculate_misspelling_percentage(text)

    print(list_text)
    print(f"\nMisspelling percentage: {misspelling_percentage}")
