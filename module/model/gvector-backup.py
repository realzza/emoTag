import sys
import torch
import torch.nn as nn 
from .resnet import *


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


class Gvector(nn.Module):
    def __init__(self, channels=16, block='BasicBlock', num_blocks=[2,2,2,2], 
            embd_dim=128, drop=0.5, n_class=1211):
        super(Gvector, self).__init__()
        block = str_to_class(block) 
        self.resnet = ResNet(channels, block, num_blocks)
        self.fc1 = nn.Linear(channels*8*2, embd_dim)
        self.dropout = nn.Dropout(drop)
        self.fc2 = nn.Linear(embd_dim, n_class)

    def extractor(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x_mean = x.mean(dim=2)
        x_std  = x.std(dim=2)
        x = torch.cat((x_mean, x_std), dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.extractor(x)
        x = self.fc2(x)
        return x 
