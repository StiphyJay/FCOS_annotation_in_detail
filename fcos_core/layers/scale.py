import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value])) #将init_value这个不可训练的tensor转换成可以训练的类型parameter

    def forward(self, input):
        return input * self.scale
