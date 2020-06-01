# Video Summarization using Deep Semantic Features
これは"Video Summarization using Deep Semantic Features" in ACCV'16 [[arXiv](arxiv.org/abs/1609.08758)]でのを書き直したものになります。詳しくは[[こちら](http://github.com/mayu-ot/vsum_dsf)]

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

この実験で使うデータ(SumMeデータセット)は `data.zip` [**HERE**](https://www.dropbox.com/s/zxp8dq18t0tqlk2/data.zip?dl=0)として保管してあります。
データセットについては([こちら](https://people.ee.ethz.ch/~gyglim/vsum/index.php))を参照してください。  
以下のようにしてデータを解凍してください。  

	cd data/summe
	wget https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip
	unzip SumMe.zip

## 実験
以下の通りに実行してください。

	python script/summarize.py
	python script/evaluate.py results/summe/smt_feat
