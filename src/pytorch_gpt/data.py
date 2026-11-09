"""
Data pipeline: character-level tokenizer, train/validation split, and random batches.
"""

from pathlib import Path

import torch


class CharTokenizer:
    """
    Character-level tokenizer: every distinct character in the text is one token.
    IDs are 0 .. V-1, assigned in sorted order so they are the same on every run.
    """

    def __init__(self, text):
        # Vocabulary: every distinct character, sorted (a set has no fixed order)
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)  # V

        # Lookup tables in both directions
        self.char_to_id = {char: i for i, char in enumerate(self.chars)}  # for encode
        self.id_to_char = {i: char for i, char in enumerate(self.chars)}  # for decode

    def encode(self, text):
        """
        text:    string
        returns: list of token IDs, one per character
        """
        return [self.char_to_id[char] for char in text]

    def decode(self, ids):
        """
        ids:     sequence of token IDs (list, or a 1D tensor converted with .tolist())
        returns: the string they represent
        """
        return "".join(self.id_to_char[i] for i in ids)


def load_data(path, train_fraction=0.9):
    """
    Reads a text file, builds the tokenizer, and splits the encoded text.
    path:           text file to load (str or Path)
    train_fraction: share of the text used for training (the rest is validation)
    returns:        tokenizer, train_ids (N_train,), val_ids (N_val,)  1D long tensors on the CPU
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


def get_batch(ids, batch_size, context_length, device, generator=None):
    """
    Random chunks of the data as inputs; the same chunks shifted by one as targets.
    ids:            (N_ids,) 1D long tensor on the CPU (train_ids or val_ids)
    batch_size:     B, number of chunks
    context_length: T, tokens per chunk
    device:         where to put the batch ("cuda" or "cpu")
    generator:      torch.Generator for reproducible batches (None = global random state)
    returns:        X (B, T), Y (B, T) on device, with Y[b, t] = the token after X[b, t]
    """
    # B random start positions; high is excluded and leaves room for the +1 shift in Y
    starts = torch.randint(0, len(ids) - context_length, (batch_size,), generator=generator)  # (B,)

    # Index grid: row b is starts[b], starts[b]+1, ..., starts[b]+T-1
    offsets = torch.arange(context_length)  # (T,)
    index_grid = starts[:, None] + offsets  # (B, T)

    X = ids[index_grid]  # (B, T) inputs
    Y = ids[index_grid + 1]  # (B, T) targets: every index shifted one later

    # The dataset stays on the CPU; only this small batch is sent to the GPU
    return X.to(device), Y.to(device)
