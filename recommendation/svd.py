# coding=utf-8
# 输入数据格式： user_id_index, doc_id_index, label, 其中的id已经转化为index
# 网络结构：单纯的输入输出层
# 做demo，不考虑效率问题
# 实现SVD版本的分解

import torch
from torch import nn
from torch.autograd import Variable, Function
from torch.nn import LogSigmoid

class SVDModel(nn.Module):
    def __init__(self, user_size, doc_size, hidden_size):
        pass

    def forward(self, *input):
        pass


if __name__ == '__main__':
    pass

