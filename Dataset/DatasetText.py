import numpy as np
import torch
from collections import Counter
from gensim.models.word2vec import Word2Vec
from Dataset.utils import MySentences, custom_tokenizer
import sentencepiece as spm
import copy


class DatasetText(torch.utils.data.Dataset):
    def __init__(
        self,
        folder_path,
        sequence_length,
        mode="word",
        word2vec=False,
        embedding_dim=None,
        use_bpe=False,
        bpe_vocab_size=10000,
    ):
        self.sequence_length = sequence_length
        self.folder_path = folder_path
        self.mode = mode
        self.word2vec = word2vec
        self.embedding_dim = embedding_dim
        self.use_bpe = use_bpe  
        self.bpe_vocab_size = bpe_vocab_size


        if self.use_bpe: # Handle BPE separately because of minor differences in the method
            self.words = self.load_words()
            self.bpe_model = self.train_bpe()
            text = copy.copy(self.words)
            self.words = self.apply_bpe(text)
            self.vocab_size = self.bpe_model.get_piece_size()
            self.word_to_index = {self.bpe_model.id_to_piece(i): i for i in range(self.vocab_size)}
            self.index_to_word = {i: self.bpe_model.id_to_piece(i) for i in range(self.vocab_size)}
            self.uniq_words = list(self.word_to_index.keys())
            self.words_indexes = self.bpe_model.encode(text, out_type=int)
        
        else :
            if word2vec:
                self.sentences = MySentences(folder_path)
                self.embedding_dim = 100 if embedding_dim is None else embedding_dim
                model = Word2Vec(
                    sentences=self.sentences,
                    vector_size=self.embedding_dim,
                    window=5,
                    min_count=1,
                    sg=1,
                    negative=5,
                    ns_exponent=0.75,
                )
                self.wv = model.wv
            
            self.words = self.load_words()

            self.uniq_words = self.get_uniq_words()
            self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
            self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
            self.words_indexes = [self.word_to_index[w] for w in self.words]
            self.vocab_size = len(self.uniq_words)

    def load_words(self):
        with open(self.folder_path, "r", encoding="utf-8") as file:
            text = file.read()
        if self.use_bpe:
            return text
        elif self.mode == "word":
            return custom_tokenizer(text)
        elif self.mode == "character":
            return list(text)
        else:
            raise ValueError("wrong mode")

    def train_bpe(self):
        spm.SentencePieceTrainer.train(
            input=str(self.folder_path),
            model_prefix="bpe",
            vocab_size=self.bpe_vocab_size,
        )
        sp = spm.SentencePieceProcessor()
        sp.load("bpe.model")
        return sp

    def apply_bpe(self, words):
        return self.bpe_model.encode(words, out_type=str)

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
    # folder_path = ROOT / "Data" / "txt" / "shakespeare.txt"
    folder_path = ROOT / "Data" / "txt" / "harry_potter.txt"

    dataset = DatasetText(
        folder_path=folder_path,
        sequence_length=100,
        mode="word",
        word2vec=True,
        use_bpe=False,
        bpe_vocab_size=5000,
    )

    print(dataset[0][0])
    print(" ".join([dataset.index_to_word[i.item()] for i in dataset[0][0]]))
    print(" ".join([dataset.index_to_word[i.item()] for i in dataset[0][1]]))

    # If use_bpe is true : 
    #print(dataset.bpe_model.decode([dataset.index_to_word[i.item()] for i in dataset[0][0]]))
    #print(dataset.bpe_model.decode([dataset.index_to_word[i.item()] for i in dataset[0][1]]))

    # print("".join(dataset.words))
    # print(dataset.uniq_words)
