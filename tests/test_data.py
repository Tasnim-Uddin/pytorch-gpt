"""Tests for loading text data and sampling training batches."""

import pytest
import torch

from pytorch_gpt.data import get_batch, load_data

BATCH_SIZE = 4
CONTEXT_LENGTH = 8
TEXT = "First Citizen:\nBefore we proceed any further, hear me speak."

# Each token ID equals its own position, so a batch's values show exactly which windows it took
SEQUENTIAL_IDS = torch.arange(100)


@pytest.fixture
def text_file(tmp_path):
    """Write TEXT to a file in a fresh temporary directory and return its path."""
    path = tmp_path / "input.txt"
    path.write_text(TEXT, encoding="utf-8")
    return path


def test_load_data_splits_all_tokens(text_file):
    tokenizer, train_ids, val_ids = load_data(text_file, train_fraction=0.9)
    assert len(train_ids) == int(0.9 * len(TEXT))

    # Joined back together, the two splits must be exactly the original text:
    # nothing lost, duplicated or reordered at the cut
    assert tokenizer.decode(torch.cat([train_ids, val_ids]).tolist()) == TEXT


def test_load_data_returns_int64_ids(text_file):
    _, train_ids, val_ids = load_data(text_file)
    assert train_ids.dtype == val_ids.dtype == torch.long


def test_load_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.txt")


def test_get_batch_shapes():
    inputs, targets = get_batch(SEQUENTIAL_IDS, BATCH_SIZE, CONTEXT_LENGTH, "cpu")
    assert inputs.shape == targets.shape == (BATCH_SIZE, CONTEXT_LENGTH)


def test_get_batch_windows_are_contiguous_and_shifted():
    inputs, targets = get_batch(SEQUENTIAL_IDS, BATCH_SIZE, CONTEXT_LENGTH, "cpu")

    # Each row is a run of consecutive positions
    assert torch.equal(inputs[:, 1:], inputs[:, :-1] + 1)
    # Each target is the token one position after its input
    assert torch.equal(targets, inputs + 1)


def test_get_batch_reaches_last_token():
    # Data one token longer than a window fits exactly one window, starting at 0,
    # so an off-by-one in the start positions would crash or pick an impossible window
    ids = torch.arange(CONTEXT_LENGTH + 1)
    inputs, targets = get_batch(ids, BATCH_SIZE, CONTEXT_LENGTH, "cpu")

    # (B, T) == (T,) broadcasts: every row must equal the one possible window
    assert (inputs == ids[:-1]).all()
    assert (targets == ids[1:]).all()


def test_get_batch_is_reproducible_with_generator():
    def sample(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        return get_batch(SEQUENTIAL_IDS, BATCH_SIZE, CONTEXT_LENGTH, "cpu", generator)

    inputs, targets = sample(seed=0)
    inputs_again, targets_again = sample(seed=0)
    assert torch.equal(inputs, inputs_again)
    assert torch.equal(targets, targets_again)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_get_batch_moves_batch_to_device():
    inputs, targets = get_batch(SEQUENTIAL_IDS, BATCH_SIZE, CONTEXT_LENGTH, "cuda")
    assert inputs.device.type == targets.device.type == "cuda"
