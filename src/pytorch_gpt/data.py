"""Data pipeline: train/validation split and random batches."""

from pathlib import Path

import torch

from pytorch_gpt.tokenizer import CharTokenizer


def load_data(
    path: str | Path, train_fraction: float = 0.9
) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    """Read a text file, build its tokenizer, and split the encoded text.

    Args:
        path: Text file to load.
        train_fraction: Share of the text used for training; the rest is validation.

    Returns:
        The tokenizer, then the training IDs of shape (N_train,) and the validation
        IDs of shape (N_val,), both long tensors on the CPU.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download Tiny Shakespeare with:\n"
            "  mkdir -p data && curl -o data/input.txt "
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    text = path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)

    # The whole text as one long tensor of token IDs (long: required by nn.Embedding)
    all_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)  # (total characters,)

    # One contiguous cut: first part for training, last part held out for validation
    split = int(train_fraction * len(all_ids))
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    return tokenizer, train_ids, val_ids


def get_batch(
    ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random chunks of tokens as inputs, with the next token of each as targets.

    Args:
        ids: Token IDs of shape (N,) on the CPU, e.g. `train_ids` or `val_ids`.
        batch_size: Number of chunks, B.
        context_length: Tokens per chunk, T.
        device: Device to put the batch on, e.g. "cuda" or "cpu".
        generator: Random generator for reproducible batches; None uses the global one.

    Returns:
        Inputs and targets, both of shape (B, T) on `device`, where targets[b, t] is the
        token that follows inputs[b, t].
    """
    # B random start positions; high is excluded and leaves room for the +1 shift in targets
    starts = torch.randint(0, len(ids) - context_length, (batch_size,), generator=generator)  # (B,)

    # Index grid: row b is starts[b], starts[b]+1, ..., starts[b]+T-1
    # (B, 1) + (T,) broadcasts to (B, T)
    offsets = torch.arange(context_length)  # (T,)
    index_grid = starts[:, None] + offsets  # (B, T)

    # Indexing with a (B, T) grid gathers a (B, T) tensor of token IDs
    inputs = ids[index_grid]  # (B, T)
    targets = ids[index_grid + 1]  # (B, T): every index shifted one later

    # The dataset stays on the CPU; only this small batch is sent to the GPU
    return inputs.to(device), targets.to(device)
