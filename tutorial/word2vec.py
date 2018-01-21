# coding=utf-8
# 实验word embedding
# url
import torch
from torch import autograd
from torch.autograd import Variable, Function
from torch import nn


class WordEmbedding(nn.Module):
    def __init__(self):
        super(WordEmbedding, self).__init__()
        self.linear = nn.Linear(100, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 100)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    word_embedding = WordEmbedding()
    x = Variable(torch.rand(10, 100))
    loss_func = nn.MSELoss()

    x_pred = word_embedding.forward(x)
    loss = loss_func(x, x_pred)
    print loss
