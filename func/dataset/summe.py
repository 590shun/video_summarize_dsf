#-*- coding:utf-8 -*-

##SumMeデータセットのロード##

#モジュールのインポート#
import numpy as np
import json

#データセットが存在するフォルダの指定#
data_root = './data/summe'

#クラスの定義#
class summe():
    def __init__(self, video_id, feat_type='vgg'):
        #データのロード
        dataset = json.load(open(data_root + 'dataset.json'))
        print('ロードした動画のID:', video_id)
        #リストdatasetからvideo_idと一致する要素'videoID'を抽出(lambda x:無名関数)
        data = filter(lambda x: x['videoID'] == video_id, dataset)
        self.data = data[0]
        self.feat = np.load(data_root + 'feat/' + feat_type + '/' + video_id + '.npy').astype(np.float32)
    
    def SamplingFrame(self):
        #frame rate
        fps = self.data['fps']
        fnum = self.data['fnum']

        idx = np.arange(fps, fnum, fps)
        #np.floor():値の切り捨て
        idx = np.floor(idx)
        #list化
        idx = idx.tolist()
        #idxをmap関数によりint型に変換
        idx = map(idx, int)

        img = [self.data['image'][i] for i in idx]
        img_id = [self.data['imgID'][i] for i in idx]
        score = [self.data['score'][i] for i in idx]

        return img, img_id, score
