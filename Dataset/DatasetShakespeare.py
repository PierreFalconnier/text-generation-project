import torch
from collections import Counter


class DatasetShakespeare(torch.utils.data.Dataset):
    def __init__(self, folder_path, sequence_length):
        self.sequence_length = sequence_length
        self.folder_path = folder_path
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]
        self.vocab_size = len(self.uniq_words)

    def load_words(self):
        # with open(self.folder_path / "goblet_book.txt", "r") as file:
        with open(self.folder_path / "shakespeare.txt", "r") as file:
            text = file.read()
        return text.split()

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        # account for non-overlapping sequences, best approach?
        return len(self.words_indexes) // self.sequence_length
        # return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        start_index = index * self.sequence_length
        end_index = start_index + self.sequence_length

        input_indices = self.words_indexes[start_index:end_index]
        target_indices = self.words_indexes[start_index + 1 : end_index + 1]

        # input_indices = self.words_indexes[index : index + self.sequence_length]
        # target_indices = self.words_indexes[
        #     index + 1 : index + self.sequence_length + 1
        # ]

        return torch.tensor(input_indices), torch.tensor(target_indices)


if __name__ == "__main__":

    from pathlib import Path
    import sys

    # IMPORTATIONS
    # include the path of the dataset(s) and the model(s)
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    # DATASET
    folder_path = ROOT / "Data" / "shakespeare"

    dataset = DatasetShakespeare(folder_path=folder_path, sequence_length=10)

    print(dataset[0][0])
    print(dataset[0][0].shape)
    print(dataset[0][1])
    print(dataset[0][1].shape)

    print(len(dataset))

    for k in range(10):
        print(dataset[k][0])

    print(dataset.word_to_index)
