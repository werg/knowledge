import os
from io import open
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.paths = {'train': os.path.join(path.format('train')),
                      'valid': os.path.join(path.format('valid')),
                      'test': os.path.join(path.format('test'))}

        for path in self.paths.values():
            self.build_dict(path)

        self.dictionary.add_word('<sos>')
        self.dictionary.add_word('<eos>')

    def get_docs(self, split, device):
        path = self.paths[split]
        sos = self.dictionary.word2idx['<sos>']
        eos = self.dictionary.word2idx['<eos>']
        with open(path, 'r', encoding="utf8") as f:
            doc = [sos]
            for line in f:
                if line.startswith(' = '):
                    if not line.startswith(' = =') and len(doc) > 1:
                        # new document
                        yield torch.tensor(doc + [eos], device=device).type(torch.int64)
                        doc = [eos]
                else:
                    doc.extend([self.dictionary.word2idx[word]
                                for word in line.split()])

    def build_dict(self, path):
         # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.dictionary.add_word(word)
                
        
