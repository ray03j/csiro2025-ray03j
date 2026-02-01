# csiro2025

## Setup with Docker

```bash
docker compose up --build
```

```bash
docker compose exec dev /bin/bash
```

```bash
wandb login
```

## Setup with uv (alternative)

Install uv if not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate  # On Windows
uv pip install -e .
```

Login to wandb:

```bash
wandb login
```

## Training

```bash
python 02_train.py
```

```bash
python 02_train.py opts \
       trainer.max_epochs=5 \
       wandb.name=exp_1
```
