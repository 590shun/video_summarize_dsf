#-*- coding:utf-8 -*-

#モジュールのインポート#
from scipy.io import loadmat
import numpy as np
import json
from sklearn.metrics import f1_score


header = '''|                         | Min. HUMAN | Avg. HUMAN | Max. HUMAN |  Summarization  |
|------------------------:|:----------:|:----------:|:----------:|:---------------:|'''

entry = '|{:>25}|{:^12.3}|{:^12.3}|{:^12.3}|{:^17.3}|'

res_sum_mean = '|                     mF1 |{:^12.3}|{:^12.3}|{:^12.3}|{:^17.3}|'
res_sum_rel_to_avr = '|   relative to avg.human |{:^12.4}|{:^12.4}|{:^12.4}|{:^17.4}|'
res_sum_rel_to_max = '|   relative to best.human|{:^12.4}|{:^12.4}|{:^12.4}|{:^17.4}|'

#f1 scoreの算出#
def eval_f1(pred_summary, gt_summary):
    pred_summary = (pred_summary > 0).astype(np.int)
    gt_summary = (gt_summary > 0).astype(np.int)
    f1 = map(lambda y_true: f1_score(y_true, pred_summary), gt_summary)
    return sum(f1) / len(f1)

#summaryの評価#
def eval_summary(dataset_name, res_base, gt_base):
    res_base = res_base

    #データセットのロード
    data = json.load(open('data/{}/dataset.json'.format(dataset_name)))
    res = {}

    for d in data:
        gt_data = loadmat(gt_base +'/%s.mat' % v_id)
        #人間の手によるスコアを読み込む
        user_score = gt_data.get('user_score')
        user_score = user_score.T

        fscore_all = []

        for u_id in range(len(user_score)):
            pred_summary = user_score[u_id]
            gt_summary = np.delete(user_score, u_id, 0)
            f1 = eval_f1(pred_summary, gt_summary)
            fscore_all.append(f1)

        res_min[v_id] = min(fscore_all) #最小
        res_avg[v_id] = sum(fscore_all) / len(fscore_all) #平均
        res_max[v_id] = max(fscore_all) #最大
    
    return res_min, res_avg, res_max
