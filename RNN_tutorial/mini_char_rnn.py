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


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        # self.input_layer = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                          batch_first=True)
        # self.output_layer = nn.Softmax()

    def forward(self, x, h0):
        x_out, hn = self.rnn(x, h0)
        return x_out, hn


def predict(size):
    # how to start predict?
    # or how to generate a char sequense
