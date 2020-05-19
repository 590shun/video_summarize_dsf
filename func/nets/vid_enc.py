#-*- coding:utf-8 -*-

#モジュールのインポート#
import sys, os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from os.path import dirname
from numpy.random import RandomState
import json

#クラス定義#
class Model(nn.Module):
    def __init__(self, batch_size={'video':5}):
        self.batch_size = batch_size
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4096, 1000)
        self.fc2 = nn.Linear(1000, 300)

    def __call__(self, x_seg):
        batch_size = self.batch_size['video'] #5
        y0 = F.tanh(self.fc1(x_seg)) #torch.Size([510, 1000])
        y1 = F.tanh(self.fc2(y0)) #torch.Size([510, 300])
        h = torch.reshape(y1, (int(y1.shape[0] / batch_size), batch_size, 300))
        return torch.sum(h, axis=1) / batch_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
