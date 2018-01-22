# coding=utf-8
# 实现对词或者句子的序列modeling

import torch
from torch import nn
from torch.autograd import Variable, Function


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

    def forward(self, *input):
        pass


def data_loader(path):
    pass


def train(model, data_iter):
    pass

if __name__ == '__main__':
    pass
