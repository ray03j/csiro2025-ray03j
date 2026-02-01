# csiro2025

本リポジトリは、Kaggle コンペティション
**[CSIRO Biomass Estimation](https://www.kaggle.com/competitions/csiro-biomass)**
に取り組むための学習・実験用コードをまとめたものです。

実装やディレクトリ構成は、以下のリポジトリを参考にしています。

* [https://github.com/yu4u/kaggle-czii-4th](https://github.com/yu4u/kaggle-czii-4th)

---

## セットアップ（Docker を使用する場合）

Docker を用いて開発環境を構築する手順は以下の通りです。

まず、Docker イメージをビルドしてコンテナを起動します。

```bash
docker compose up --build
```

起動後、開発用コンテナに入ります。

```bash
docker compose exec dev /bin/bash
```

Weights & Biases（wandb）にログインします。

```bash
wandb login
```

---

## セットアップ（uv を使用する場合・代替手段）

Docker を使用しない場合は、`uv` を用いて仮想環境を構築できます。

### uv のインストール（未インストールの場合）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 仮想環境の作成と依存関係のインストール

```bash
uv venv
source .venv/bin/activate  # Linux / macOS の場合
# .venv\Scripts\activate  # Windows の場合
uv pip install -e .
```

### wandb へのログイン

```bash
wandb login
```

---

## 学習（Training）

基本的な学習の実行方法は以下の通りです。

```bash
python 02_train.py
```

Hydra のオプションを指定して学習設定を変更することも可能です。
以下は、エポック数を 5 に設定し、wandb 上の実験名を指定する例です。

```bash
python 02_train.py opts \
       trainer.max_epochs=5 \
       wandb.name=exp_1
```
