"""Saving training progress to disk and loading it back."""

from dataclasses import asdict
from pathlib import Path

import torch

from pytorch_gpt.config import GPTConfig, TrainConfig
from pytorch_gpt.model import GPT
from pytorch_gpt.tokenizer import CharTokenizer


def save_checkpoint(
    path: str | Path,
    model: GPT,
    tokenizer: CharTokenizer,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    step: int,
    best_val_loss: float,
) -> None:
    """Save everything needed to use the model or to resume training it.

    The checkpoint holds only tensors and plain Python values, so it loads with
    `weights_only=True`.

    Args:
        path: File to write, usually ending in .pt; missing folders are created.
        model: Model whose architecture and weights are saved.
        tokenizer: Tokenizer whose vocabulary is saved.
        optimizer: Optimizer whose state is saved.
        train_config: Training settings, kept as a record of the run.
        step: Training step the weights come from.
        best_val_loss: Lowest validation loss reached up to `step`.
    """
    checkpoint = {
        # To use the model
        "gpt_config": asdict(model.config),  # architecture, to build a model of the right shape
        "model": model.state_dict(),  # learned weights, keyed by parameter name
        "vocab": "".join(tokenizer.chars),  # each token's character, in ID order
        # To resume training
        "optimizer": optimizer.state_dict(),  # AdamW's running averages, per parameter
        "train_config": asdict(train_config),
        "step": step,
        "best_val_loss": best_val_loss,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: torch.device | str) -> tuple[GPT, CharTokenizer]:
    """Rebuild a model and its tokenizer from a checkpoint.

    Args:
        path: Checkpoint written by `save_checkpoint`.
        device: Device to put the model on, e.g. "cuda" or "cpu".

    Returns:
        The model in eval mode, and its tokenizer.
    """
    # weights_only refuses arbitrary Python objects, which could run code when loaded;
    # map_location moves tensors saved on one device onto the one we load to
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # Build a model of the saved shape, then copy the saved weights into it
    model = GPT(GPTConfig(**checkpoint["gpt_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # CharTokenizer sorts the distinct characters, so the vocabulary rebuilds the same IDs
    tokenizer = CharTokenizer(checkpoint["vocab"])
    return model, tokenizer


def load_training_state(path: str | Path, optimizer: torch.optim.Optimizer) -> tuple[int, float]:
    """Restore the optimizer and the random generators from a checkpoint.

    Args:
        path: Checkpoint written by `save_checkpoint`.
        optimizer: Optimizer over the parameters of the model loaded from `path`.

    Returns:
        The step the checkpoint was saved at, and the lowest validation loss up to it.
    """
    # Load onto the CPU, where set_rng_state requires its tensors;
    # the optimizer moves its own state to its parameters' device
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(checkpoint["optimizer"])

    # The random state at the save, so batches continue as if training never stopped
    torch.set_rng_state(checkpoint["cpu_rng_state"])
    if checkpoint["cuda_rng_states"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])

    return checkpoint["step"], checkpoint["best_val_loss"]
