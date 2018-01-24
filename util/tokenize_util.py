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
    def __init__(self, path='./data', seq_size = 100):
        self.dictionary = Dictionary()
        self.train = os.path.join(path, 'train.txt')
        self.test = os.path.join(path, 'test.txt')
        self.dir = path
        self.size = 0
        self.seq_size = seq_size
        self.tokenize = []
        print '>>> start to read word dictionary:'
        self.pre_process()
        print '>>> start to tokenization data:'
        self.data_loader()

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
        seq_size = self.seq_size
        if test is None:
            path = self.train
        else:
            path = self.test
        # return the whole torch tensor
        temp_set = [0]*seq_size
        for i, x in enumerate(open(path, 'r').read()):
            index = i%seq_size
            if i>0 and (index == 0 or i==self.size-1):
                self.tokenize.append(temp_set)
                temp_set = [0] * seq_size
            temp_set[index] = self.dictionary.word2idx[x]

    def data_iter(self, seed=None, batch_size=1):
        if len(self.tokenize) == 0:
            self.data_loader()
        token_chunk_size = len(self.tokenize)
        for x, y in batch_iter(self.tokenize, self.seq_size,
                               batch_size, seed):
            yield x, y


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


def one_hot_helper(ids, out_tensor):
    if not isinstance(ids, (list, np.ndarray)):
        raise ValueError("ids must be 1-D list or array")
    ids = torch.LongTensor(ids).view(-1,1)
    out_tensor.zero_()
    out_tensor.scatter_(dim=1, index=ids, src=1.)

def one_hot_batch(ids_list, out_tensor_batch, batch_size):
    out_tensor_batch.zero_()
    ids_batch = torch.LongTensor(ids_list).view(batch_size, -1, 1)
    out_tensor_batch.zero_()
    out_tensor_batch.scatter_(dim=2, index=ids_batch, value=1.0)

def batch_iter(id_list, seq_size, batch_size, seed):
    # loop until end: when end is less then batch_size, cut
    size = len(id_list)
    max_index = size/batch_size
    if seed is not None:
        np.random.seed(seed)
    random_seq = np.random.permutation(range(size))
    for i in range(max_index):
        batch_index = random_seq[i*batch_size:(i+1)*batch_size]
        yield ([id_list[x][:seq_size-1] for x in batch_index],
               [id_list[x][1:] for x in batch_index])

if __name__ == '__main__':
    path = './test_data'
    char_corpus = CharCorpus(path, seq_size=5)
    char_corpus.pre_process()
    char_corpus.data_loader()
    print char_corpus.dictionary.word2idx
    for x in char_corpus.tokenize:
        print ','.join([char_corpus.dictionary.idx2word[z] for z in x]) + '$'

    for x in char_corpus.data_iter(batch_size=2):
        print x