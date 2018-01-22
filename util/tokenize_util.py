import numpy as np
import torch
import os


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class CharCorpus(object):
    def __init__(self, path='./data', mini_batch = 100):
        self.dictionary = Dictionary()
        self.train = os.path.join(path, 'train.txt')
        self.test = os.path.join(path, 'test.txt')
        self.dir = path
        self.size = 0
        self.mini_batch = mini_batch
        self.tokenize = []

    def pre_process(self):
        for f in os.listdir(self.dir):
            path = os.path.join(self.dir, f)
            print 'read file: ', path
            self._read_file(path)
        print 'read finished'

    def _read_file(self, path):
        data = open(path, 'r').read()
        char_set = set(data)
        self.size = len(data)
        for x in char_set:
            self.dictionary.add_word(x)

    def data_loader(self, test=None):
        # all in memory:
        mini_batch = self.mini_batch
        if test is None:
            path = self.train
        else:
            path = self.test
        # return the whole torch tensor
        temp_set = [0]*mini_batch
        for i, x in enumerate(open(path, 'r').read()):
            index = i%mini_batch
            if i>0 and (index == 0 or i==self.size-1):
                self.tokenize.append(temp_set)
                temp_set = [0] * mini_batch
            temp_set[index] = self.dictionary.word2idx[x]

    def data_iter(self, seed=None):
        if len(self.tokenize) == 0:
            self.data_loader()
        token_chunk_size = len(self.tokenize)
        random_seq = np.random.permutation(range(token_chunk_size))
        for index in random_seq:
            yield self.tokenize[index]


class Corpus(object):
    def __init__(self, path='./data'):
        self.dictionary = Dictionary()
        self.train = os.path.join(path, 'train.txt')
        self.test = os.path.join(path, 'test.txt')
        self.dir = path

    def pre_process(self):
        for f in os.listdir(self.dir):
            print 'read file: ', f
            self._read_file(f)
        print 'read finished'

    def _read_file(self, path):
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def data_loader(self, test=None):
        # all in memory:
        if test is None:
            path = self.train
        else:
            path = self.test


if __name__ == '__main__':
    path = './test_data'
    char_corpus = CharCorpus(path, mini_batch=3)
    char_corpus.pre_process()
    char_corpus.data_loader()
    print char_corpus.dictionary.word2idx
    for x in char_corpus.tokenize:
        print ''.join([char_corpus.dictionary.idx2word[z] for z in x])