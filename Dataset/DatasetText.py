import numpy as np
import torch
from collections import Counter
from gensim.models.word2vec import Word2Vec
from utils import MySentences


class DatasetText(torch.utils.data.Dataset):
    def __init__(self, folder_path, sequence_length, mode="word", word2vec=False, embedding_dim=None):
        self.sequence_length = sequence_length
        self.folder_path = folder_path
        self.mode = mode
        self.word2vec = word2vec
        self.embedding_dim = embedding_dim

        if word2vec:
            self.sentences = MySentences(folder_path)
            self.embedding_dim = 100 if embedding_dim is None else embedding_dim
            model = Word2Vec(sentences=self.sentences, vector_size=self.embedding_dim, window=5,  min_count=1, sg=1, negative=5, ns_exponent=0.75)
            self.wv = model.wv

        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.words_indexes = [self.word_to_index[w] for w in self.words]
        self.vocab_size = len(self.uniq_words)

    def load_words(self):
        with open(self.folder_path, "rb") as file:
            text = file.read()
            text = text.decode('utf-8')
        if self.mode == "word":
            if self.word2vec:
                return self.sentences.custom_tokenizer(text)
            else:
                return text.split()
        elif self.mode == "character":
            return list(text)
        else:
            raise ValueError("wrong mode")

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        # account for non-overlapping sequences
        return len(self.words_indexes) // (self.sequence_length + 1)

    def __getitem__(self, index):
        start_index = index * self.sequence_length
        end_index = start_index + self.sequence_length

        input_indices = self.words_indexes[start_index:end_index]
        target_indices = self.words_indexes[start_index + 1 : end_index + 1]

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
    # folder_path = ROOT / "Data" / "txt" / "goblet_book.txt"
    folder_path = ROOT / "Data" / "txt" / "shakespeare.txt"
    # folder_path = ROOT / "Data" / "txt" / "harry_potter.txt"

    dataset = DatasetText(
        folder_path=folder_path, sequence_length=100, mode="word", word2vec=True
    )

    print(dataset[0][0])
    print(" ".join([dataset.index_to_word[i.item()] for i in dataset[0][0]]))
    print(" ".join([dataset.index_to_word[i.item()] for i in dataset[0][1]]))
