#-*- coding:utf-8 -*-

#モジュールのインポート#
import torch.nn.functional as F
import torch.nn as nn

#クラス定義#
class Model(nn.Module):
    def __init__(self):
        self.batch_size = {'video':5}
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4096, 1000)
        self.fc2 = nn.Linear(1000, 300)
    
    def __call__(self, x_seg):
        # 入力:np.array(5*N, 4096)
        batch_size = self.batch_size['video']
        with cuda.get_device(x_seg.data):
            y0 = F.tanh(self.fc1(x_seg))
            # print(y0.shape)
            y1 = F.tanh(self.fc2(y0))
            h = F.reshape(y1, (y1.shape[0] / batch_size, batch_size, 300))
            return F.sum(h, axis=1) / batch_size
