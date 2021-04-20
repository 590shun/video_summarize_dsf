# Video Summarization using Deep Semantic Features
これは"Video Summarization using Deep Semantic Features" in ACCV'16 [[arXiv](arxiv.org/abs/1609.08758)]を書き直したものになります。  
[実装元のリンク](http://github.com/mayu-ot/vsum_dsf),[備忘録](https://github.com/590shun/paper_challenge/issues/7)

## 実験手順

	git clone https://github.com/590shun/vsum_dsf.git

### オプション
このコードでは以下のM. Gygli *et al.* [1]を参考にしています。
環境構築は次のように行います。

	cd vsum_dsf
	git clone https://github.com/gyglim/gm_submodular.git
	cd gm_submodular
	python setup.py install --user

[1] Gygli, Grabner & Van Gool. Video Summarization by Learning Submodular Mixtures of Objectives. CVPR 2015.

### データセットのダウンロードとモデルの使用について

この実験で使うデータ(SumMeデータセット)は [**data.zip**](https://www.dropbox.com/s/zxp8dq18t0tqlk2/data.zip?dl=0)として保管。
データセットについては([こちら](https://people.ee.ethz.ch/~gyglim/vsum/index.php))を参照。  
 

	cd data/summe
	wget https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip
	unzip SumMe.zip

## 実験

	python script/summarize.py
	python script/evaluate.py results/summe/smt_feat
