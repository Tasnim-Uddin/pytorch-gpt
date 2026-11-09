"""Generate text from the best checkpoint saved by training.

Run with `uv run python -m pytorch_gpt.sample` after `pytorch_gpt.train`.
"""

import torch

from pytorch_gpt.checkpoint import load_checkpoint
from pytorch_gpt.train import CHECKPOINT_PATH

# Sampling settings; None switches a filter off
PROMPT = "\n"
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
TOP_K: int | None = None
TOP_P: float | None = 0.9


def main() -> None:
    """Load the best checkpoint and print a continuation of `PROMPT`."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_checkpoint(CHECKPOINT_PATH, device)

    prompt_ids = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long, device=device)  # (1, T)
    output_ids = model.generate(
        prompt_ids, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
