# [`第 1 回大規模言語モデル分散学習ハッカソン`](https://abci.ai/event/2023/06/13/ja_event.html) 作業レポジトリ

## レポジトリのセットアップ

```shell
git clone https://github.com/shunk031/abci-llm-distributed-training-hackathon-01
cd /path/to/abci-llm-distributed-training-hackathon-01
```

## Python 環境の構築

- ABCI プリインストールモジュールの読み込み

```shell
module load python/3.10 cuda/11.7 cudnn/8.6

module list
# Currently Loaded Modulefiles:
#  1) python/3.10/3.10.10   2) cuda/11.7/11.7.1   3) cudnn/8.6/8.6.0
```

- python 環境の構築

```shell
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip wheel setuptools
pip install ruff black mypy
```

## [`mosaicml/llm-foundry`](https://github.com/mosaicml/llm-foundry) のインストール

- mosaicml/llm-foundry を clone

```shell
git clone https://github.com/mosaicml/llm-foundry
cd llm-foundry

# Clone したときの commit hash を確認
git show --format="%H" --no-patch
# ef350d9e64d13cb1db35ab7941bf9039b1b499fd
```

- mosaicml/llm-foundry をインストール

```shell
pip install cmake packaging torch
pip install -e ".[gpu]" # 結構時間かかります
```

- (2023/07/10 現在） composer の dev 版をインストール
  - https://github.com/mosaicml/llm-foundry/issues/221
```shell
pip install git+https://github.com/mosaicml/composer.git@dev
```

## ジョブを投入

```shell
export GROUP=XXXXXXXXXX
export WANDB_API_KEY=XXXXXXXXXX

cd /path/to/abci-llm-distributed-training-hackathon-01

qsub -g $GROUP scripts/exp03.sh
```

## モデルの種類

- `exp02.sh`: MPT-7B 用
- `exp03.sh`: MPT-30B 用

## 学習結果
wandb から確認できます：
- [wandb.ai/shunk031/abci-llm-distributed-training-hackathon-01](https://wandb.ai/shunk031/abci-llm-distributed-training-hackathon-01)
