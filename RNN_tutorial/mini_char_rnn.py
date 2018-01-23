# coding=utf-8
# RNN本质上是一个从序列到序列的问题
# 而在一个状态到另一个状态之间的跃迁，是共享权重的
# 这个权重，就是我们训练的目标

# 理论上说，这是一个非常长的展开序列：BPTT过程，会消耗很多内存
# 但是我们可以做有限窗口阶段

# 输入原则上是用embedding，但是对于char，可以用字典
# 损失函数用cross熵即可

# 预测的时候，可以取最大，或者sampling？
# 记住测试的时候也是一个序列测试：即从 start token 开始，依次sampling，
# sample出来的字符，是下一次的输入

import torch
from torch import nn, autograd
import numpy as np
from util.tokenize_util import *


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        # self.input_layer = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                          batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0):
        batch, seq, input_size = x.shape
        x_hidden, hn = self.rnn(x, h0)
        x_out = self.output_layer(x_hidden.contiguous().view(batch*seq, -1))
        return x_out, hn


def predict(model, seq_len, char_size, hidden_size, h0=None):
    # how to start predict?
    # or how to generate a char sequense
    def int_to_torch(n, x):
        x.zero_()
        x[0][0][n] = 1.0
        return x

    model.test()
    start = np.random.randint(0, char_size-1)
    if h0 is None:
        h0 = initial_state(1, hidden_size, zero=True)
    x = torch.zeros(1, 1, char_size)
    y_list = [0]*seq_len
    for i in range(seq_len):
        int_to_torch(start, x)
        x_out, h_out = model(x, h0)
        x_sample = sample(x_out.view(-1), char_size)
        y_list[i] = x_sample
        start = x_sample
        h0 = h_out
    return y_list


def predict_helper(model, h, x_in):
    x_out, hout = model(x_in, h)
    return h, x_out


def sample(p, size):
    return np.random.choice(range(size), p=p)


def initial_state(batch_size, hidden_size, zero=None):
    # uniform inital for ReLU hidden state
    if zero is None:
        return torch.zeros(1, batch_size, hidden_size)
    h = np.random.randn(batch_size, hidden_size)
    s = np.sqrt((h*h).sum())
    h0 = torch.FloatTensor(h/s).view(1, batch_size, hidden_size)
    return h0


def test_rnn():
    path = './test_data'
    seq_size = 5
    char_corpus = CharCorpus(path, seq_size=seq_size)

    char_size = len(char_corpus.dictionary.word2idx)
    input_size = char_size
    output_size = char_size
    hidden_size = 5
    batch_size = 1
    model = CharRNN(input_size, hidden_size, output_size)

    x_tensor = torch.zeros(batch_size, seq_size, input_size)
    for x in char_corpus.data_iter(batch_size=batch_size):
        one_hot_batch(x, x_tensor, batch_size=batch_size)
        print x
        print x_tensor


if __name__ == '__main__':
    test_rnn()

