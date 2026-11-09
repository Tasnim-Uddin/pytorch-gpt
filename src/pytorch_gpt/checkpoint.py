"""Saving a trained model to disk and loading it back."""

from dataclasses import asdict
from pathlib import Path

import torch

from pytorch_gpt.config import GPTConfig
from pytorch_gpt.model import GPT
from pytorch_gpt.tokenizer import CharTokenizer


def save_checkpoint(
    path: str | Path, model: GPT, tokenizer: CharTokenizer, step: int, val_loss: float
) -> None:
    """Save everything needed to rebuild the model and its tokenizer.

    The checkpoint holds only tensors and plain Python values, so it loads with
    `weights_only=True`.

    Args:
        path: File to write, usually ending in .pt; missing folders are created.
        model: Model whose architecture and weights are saved.
        tokenizer: Tokenizer whose vocabulary is saved.
        step: Training step the weights come from.
        val_loss: Validation loss at that step.
    """
    checkpoint = {
        "gpt_config": asdict(model.config),  # architecture, to build a model of the right shape
        "model": model.state_dict(),  # learned weights, keyed by parameter name
        "vocab": "".join(tokenizer.chars),  # each token's character, in ID order
        "step": step,
        "val_loss": val_loss,
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
