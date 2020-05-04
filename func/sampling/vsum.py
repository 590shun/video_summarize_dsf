#-*- coding:utf-8 -*-

#モジュールのインポート#
import gm_submoduler
import gm_submoduler.example_objectives as ex
from gm_submoduler import leskovec_maximize
from func.dataset.summe import summe
import numpy as np
import scipy.spatial.distance as dist

#クラス定義#
class VSUM(gm_submoduler.DataElement):
    def __init__(self, videoID, model, dataset='summe', featType='vgg', seg_l=5):
        
        #データセットの読み込み
        self.dataset = summe(videoID)
        
        #生成する動画の長さ？(オリジナルの動画に対する割合)
        self.budget = int(0.15 * self.dataset.data['length'] / seg_l)
        print('budget:', self.budget)

        #動画のsegmentを埋め込む
        seg_feat = encodeSeg(self.dataset, model, seg_size=seg_l)

        #特徴量の保管先？
        self.x = seg_feat
        #np.ones:配列の要素を全て1で初期化
        self.Y = np.ones(self.x.shape[0])
        
        #segment間の距離を計算
        self.dist_e = dist.squareform(dist.pdist(self.x, 'sqeuclidean'))

        #時系列方向の距離を算出
        self.frame, img_id, self.score = self.dataset.sampleFrame()
        fno_arr = np.expand_dims(np.array(img_id), axis=1)
        self.dist_c = dist.pdist(fno_arr, 'sqeulidean')
    
    def getCosts(self):
        return np.ones(self.x.shape[0])
    
    def getRelevance(self):
        return np.multiply(self.rel, self.rel)
    
    def getChrDistances(self):
        d = dist.squareform(self.dist_c)
        return np.multiply(d, d)
    
    def getDistances(self):
        return np.multiply(self.dist_e, self.dist_e)
    
    def summarizeRep(self, weight=[1.0, 0.0], seg_l=5):
        objectives = [representativeness(self), uniformity(self)]

        selected, score, minoux_bound = leskovec_maximize(self, weights, objcetives, budget=self.budget)
        selected.sort()

        frames = []
        gt_score = []
        for i in selected:
            frames.append(self.frame[i:i+seg_l])
            gt_score.append(self.score[i:i+seg_l])
        return selected, frames, gt_score

    def encodeSeg(data, model, seg_size=5):
        feat = data.feat

        img, img_id, score = data.sampleFrame()
        segs = [img_id[i:i+seg_size] for i in range(len(img_id) - seg_size + 1)]
        segs = reduce(lambda x, y: x + y, segs)

        x = feat[segs]

        #embedding
        enc_x = model(x)
        
        return enc_x.data

    def uniformity(S):
        #入力:S(getChrDistance()のデータ要素)#
        #return: uniformity objective#

        tempDMat.mean()
        return (lambda X: (1 - ex.kmedoid_loss(X, tempDMat, float(norm))))
    
    def representativeness(S):
        #入力: getDistances()のデータ要素#
        #return representativeness objectiveness#

        tempDMat = S.getDistances()
        norm = tempDMat.mean()
        return (lambda X: (1 - ex.kmedoid_loss(X, tempDMat, float(norm))))
