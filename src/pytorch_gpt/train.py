"""Train a GPT on Tiny Shakespeare, reporting its loss as it learns.

Run with `uv run python -m pytorch_gpt.train`.
"""

from pathlib import Path

import torch

from pytorch_gpt.config import GPTConfig, TrainConfig
from pytorch_gpt.data import get_batch, load_data
from pytorch_gpt.model import GPT

DATA_PATH = Path("data/input.txt")

# Sample printed once training ends
PROMPT = "\n"
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.9


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    train_config: TrainConfig,
    device: torch.device | str,
) -> dict[str, float]:
    """Average the model's loss over many batches from each split.

    A single batch gives a noisy loss, so each split averages `eval_steps` batches.
    The batches come from a freshly seeded generator, so every evaluation uses the
    same ones and a change in loss reflects the model, not the sample.

    Args:
        model: The model to evaluate.
        train_ids: Training token IDs of shape (N_train,).
        val_ids: Validation token IDs of shape (N_val,).
        train_config: Supplies `batch_size`, `eval_steps` and `seed`.
        device: Device the model is on.

    Returns:
        The mean loss for each split, keyed "train" and "val".
    """
    model.eval()  # switch layers such as dropout to inference behaviour
    losses = {}
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        generator = torch.Generator().manual_seed(train_config.seed)
        total = 0.0
        for _ in range(train_config.eval_steps):
            inputs, targets = get_batch(
                ids, train_config.batch_size, model.config.context_length, device, generator
            )
            _, loss = model(inputs, targets)
            total += loss.item()
        losses[split] = total / train_config.eval_steps
    model.train()
    return losses


def main() -> None:
    """Train a GPT with the default configs, printing both losses as it goes."""
    train_config = TrainConfig()
    torch.manual_seed(train_config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, train_ids, val_ids = load_data(DATA_PATH)
    gpt_config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPT(gpt_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{num_params:,} parameters on {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )

    # One extra pass so the finished model is evaluated too, after max_steps updates
    for step in range(train_config.max_steps + 1):
        is_last_step = step == train_config.max_steps
        if step % train_config.eval_interval == 0 or is_last_step:
            losses = estimate_loss(model, train_ids, val_ids, train_config, device)
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        if is_last_step:
            break

        inputs, targets = get_batch(
            train_ids, train_config.batch_size, gpt_config.context_length, device
        )
        _, loss = model(inputs, targets)

        optimizer.zero_grad()  # backward() adds to .grad, so clear the previous step's
        loss.backward()  # set each parameter's .grad to d(loss)/d(parameter)
        optimizer.step()  # update each parameter using its .grad

    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long, device=device)  # (1, T)
    output_ids = model.generate(prompt_ids, MAX_NEW_TOKENS, temperature=TEMPERATURE)
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
