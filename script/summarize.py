#-*- coding:utf-8 -*-

#モジュールのインポート#
import sys, os
import json
import numpy as np
from scipy.io import savemat


def get_flabel(frames, fnum, fps, seg_l):
    s_i = [int(seg_fn[0][:-4]) for seg_fn in frames]
    e_i = [s + fps * seg_l for s in s_i]
    e_i = map(round, e_i)
    e_i = map(int, e_i)

    label = np.zeros((fnum, 1))
    for s, e in zip(s_i, e_i):
        label[s:e] = 1
    return label

if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from func.sampling.vsum import VSUM
    from func.nets import vid_enc
    import torch
    import argparse
    #from chainer import serializers: モデルの保存(torch.saveで代用可能)
    #from chainer import configuration(chainer.using_config): 検証データに対する予測値の計算

    parser = argparse.ArgumentParser()
    parser.add_argument('--dname', '-d', type=str, default='summe', help='dataset name')
    parser.add_argument('--feat_type', '-f', type=str, default='smt_feat', help='feat_type: smt_feat or vgg')
    args = parser.parse_args()

    #設定
    seg_l = 5
    feat_type = args.feat_type
    d_name = args.dname
    dataset_root = 'data/{}/'.format(d_name)
    out_dir = 'results/{:}/{:}/'.format(d_name, feat_type)
    print('resultの保存先:', out_dir)

    #保存先が存在しない場合
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    dataset = json.load(open(dataset_root + 'dataset.json'))
    video_id = [d['videoID'] for d in dataset]

    #embeddingしたモデルの読み込み
    if feat_type == 'smt_feat':
        model = vid_enc.Model()
        torch.load_npz('./data/trained_model/model_par', model)
    elif feat_type == 'vgg':
        from func.nets.Seg_vgg_19 import Model
        model = Model()
    else:
        raise RunTimeError('[invalid feat_type] use smt_feat or vgg')

    for v__id in video_id:

        with torch.no_grad:
            vsum = VSUM(v_id, model, dataset=d_name, seg_l=seg_l)

        _, frames = vsum.summarizeRep(seg_l=seg_l, weights=[1.0, 0.0])

        #0か1のラベルを各フレームに付与
        fps = vsum.dataset.data['fps']
        fnum = vsum.dataset.data['fnum']
        label = get_flabel(frames, fnum, fps, seg_l)

        np.save(out_dir + v_id, label)
