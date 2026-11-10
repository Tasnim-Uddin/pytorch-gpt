"""Tests for saving and loading checkpoints."""

import torch

from pytorch_gpt.checkpoint import load_checkpoint, load_training_state, save_checkpoint
from pytorch_gpt.config import GPTConfig, TrainConfig
from pytorch_gpt.data import get_batch
from pytorch_gpt.model import GPT
from pytorch_gpt.tokenizer import CharTokenizer

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak."
TOKENIZER = CharTokenizer(TEXT)
IDS = torch.tensor(TOKENIZER.encode(TEXT))
CONTEXT_LENGTH = 8
GPT_CONFIG = GPTConfig(
    vocab_size=TOKENIZER.vocab_size,
    context_length=CONTEXT_LENGTH,
    d_model=16,
    num_heads=2,
    num_layers=1,
    d_ff=64,
)
TRAIN_CONFIG = TrainConfig(batch_size=4, lr=1e-2)


def make_optimizer(model: GPT) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=TRAIN_CONFIG.lr, weight_decay=TRAIN_CONFIG.weight_decay
    )


def train_step(model: GPT, optimizer: torch.optim.Optimizer) -> None:
    """Take one optimizer step on a batch drawn from the global random generator."""
    inputs, targets = get_batch(IDS, TRAIN_CONFIG.batch_size, CONTEXT_LENGTH, "cpu")
    _, loss = model(inputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = GPT(GPT_CONFIG).eval()

    # A folder that doesn't exist yet, which save_checkpoint must create
    path = tmp_path / "checkpoints" / "model.pt"
    save_checkpoint(path, model, TOKENIZER, make_optimizer(model), TRAIN_CONFIG, 10, 1.5)
    loaded_model, loaded_tokenizer = load_checkpoint(path, "cpu")

    assert loaded_model.config == GPT_CONFIG
    assert not loaded_model.training
    assert loaded_tokenizer.chars == TOKENIZER.chars

    # Same architecture and same weights must give exactly the same logits
    token_ids = IDS[:CONTEXT_LENGTH][None]  # (1, T)
    with torch.no_grad():
        logits, _ = model(token_ids)
        loaded_logits, _ = loaded_model(token_ids)
    assert torch.equal(loaded_logits, logits)


def test_resume_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(0)
    model = GPT(GPT_CONFIG)
    optimizer = make_optimizer(model)
    for _ in range(3):
        train_step(model, optimizer)

    path = tmp_path / "last.pt"
    save_checkpoint(path, model, TOKENIZER, optimizer, TRAIN_CONFIG, 3, 2.0)

    # Uninterrupted: carry on in memory
    train_step(model, optimizer)

    # Resumed: a fresh model and optimizer, restored from the checkpoint
    resumed_model, _ = load_checkpoint(path, "cpu")
    resumed_model.train()
    resumed_optimizer = make_optimizer(resumed_model)
    step, best_val_loss = load_training_state(path, resumed_optimizer)
    train_step(resumed_model, resumed_optimizer)

    assert (step, best_val_loss) == (3, 2.0)

    # Identical weights need both the same batch (random state) and the same AdamW state
    for param, resumed_param in zip(model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.equal(param, resumed_param)
