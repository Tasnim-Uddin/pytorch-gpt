# pytorch-gpt

A GPT language model in PyTorch, built up step by step. It started as a rewrite of
[numpy-gpt](https://github.com/Tasnim-Uddin/numpy-gpt) and extends it with modern components.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). PyTorch comes from the CUDA 12.6 index.

```sh
uv sync                     # create .venv and install the package with dev tools
uv run pre-commit install   # lint and format on every commit
```

## Data

Training uses Tiny Shakespeare (about 1 MB of text):

```sh
mkdir -p data && curl -o data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Tests

```sh
uv run pytest
```

## Layout

```
src/pytorch_gpt/
  tokenizer.py   text <-> token IDs
  data.py        train/validation split and random batches
  model.py       transformer layers and the GPT model
tests/           pytest tests, one file per module
docs/            notes, and the naming conventions in docs/conventions.md
data/            datasets (not tracked)
checkpoints/     saved weights (not tracked)
```
