from pathlib import Path
from nltk.tokenize import RegexpTokenizer

class MySentences(object):
    def __init__(self, folder_path):
        self.folder_path = folder_path
 
    def __iter__(self):
        for line in open(self.folder_path, 'rb'):
            line = line.decode('utf-8')
            yield self.custom_tokenizer(line)
    
    def custom_tokenizer(self, input_text):
        tokenizer = RegexpTokenizer(r'\w+|[^\w\s]+|\n')
        tokens = tokenizer.tokenize(input_text)

        new_tokens = []
        for token in tokens:
            if any(char in token for char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
                new_tokens.extend(list(token))
            else:
                new_tokens.append(token)
        
        return new_tokens
    
if __name__ == "__main__":
    CUR_PATH_FILE = Path(__file__)
    ROOT = CUR_PATH_FILE.parents[1]
    folder_path = str(ROOT / 'Data' / 'txt' / 'goblet_book_1000.txt')

    sentences = MySentences(folder_path)
    for sentence in sentences:
        print(sentence)
    
    test_sequence = '"Frank!" cried several people.  "Never!"'
    print(f"With split method: {test_sequence.split()}") # problem is that words and punctuation are not separated during tokenization
    print(f"With custom tokenizer: {sentences.custom_tokenizer(test_sequence)}")