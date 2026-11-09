"""Tests for saving and loading checkpoints."""

import torch

from pytorch_gpt.checkpoint import load_checkpoint, save_checkpoint
from pytorch_gpt.config import GPTConfig
from pytorch_gpt.model import GPT
from pytorch_gpt.tokenizer import CharTokenizer

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak."


def test_checkpoint_round_trip(tmp_path):
    tokenizer = CharTokenizer(TEXT)
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=8,
        d_model=16,
        num_heads=2,
        num_layers=1,
        d_ff=64,
    )
    model = GPT(config).eval()

    # A folder that doesn't exist yet, which save_checkpoint must create
    path = tmp_path / "checkpoints" / "model.pt"
    save_checkpoint(path, model, tokenizer, step=10, val_loss=1.5)
    loaded_model, loaded_tokenizer = load_checkpoint(path, "cpu")

    assert loaded_model.config == config
    assert not loaded_model.training
    assert loaded_tokenizer.chars == tokenizer.chars

    # Same architecture and same weights must give exactly the same logits
    token_ids = torch.tensor([tokenizer.encode("First Ci")])  # (1, 8)
    with torch.no_grad():
        logits, _ = model(token_ids)
        loaded_logits, _ = loaded_model(token_ids)
    assert torch.equal(loaded_logits, logits)
