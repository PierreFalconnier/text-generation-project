from pathlib import Path
import torch
import sys

if __name__ == "__main__":

    # IMPORTATIONS
    # include the path of the dataset(s) and the model(s)
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from Dataset.DatasetShakespeare import DatasetShakespeare as Dataset
    from Models.RNN import RNN
    from torch.utils.data import DataLoader
    from torch import nn, optim
    import torch.nn.functional as F
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DATASET
    folder_path = ROOT / "Data" / "shakespeare"
    sequence_length = 25
    dataset = Dataset(
        folder_path=folder_path,
        sequence_length=sequence_length,
    )

    # DATALOADER
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size)

    #   TRAIN
    model = RNN(
        vocab_size=dataset.vocab_size,
        hidden_dim=100,
        embedding_dim=None,
        num_layers=1,
        dropout=0.0,
        nonlinearity="tanh",
    ).to(device)
    criterion = nn.CrossEntropyLoss()  # include softmax !
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 1

    for epoch in tqdm(range(epochs)):
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

            # y_pred to batch_size, num_classes, sequence_length
            loss = criterion(y_pred.permute(0, 2, 1), y)
            loss.backward()
            optimizer.step()

            # avoid backprop across batchs
            state_h = state_h.detach()
