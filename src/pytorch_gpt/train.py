"""Train a GPT on Tiny Shakespeare, saving the model with the lowest validation loss.

Run with `uv run python -m pytorch_gpt.train`; add `--resume` to continue the last run.
"""

import argparse
from pathlib import Path

import torch

from pytorch_gpt.checkpoint import load_checkpoint, load_training_state, save_checkpoint
from pytorch_gpt.config import GPTConfig, TrainConfig
from pytorch_gpt.data import get_batch, load_data
from pytorch_gpt.model import GPT

DATA_PATH = Path("data/input.txt")
BEST_CHECKPOINT_PATH = Path("checkpoints/best.pt")
LAST_CHECKPOINT_PATH = Path("checkpoints/last.pt")


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
    parser = argparse.ArgumentParser(description="Train a GPT on Tiny Shakespeare.")
    parser.add_argument(
        "--resume", action="store_true", help=f"continue from {LAST_CHECKPOINT_PATH}"
    )
    args = parser.parse_args()

    train_config = TrainConfig()
    torch.manual_seed(train_config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, train_ids, val_ids = load_data(DATA_PATH)

    if args.resume:
        model, _ = load_checkpoint(LAST_CHECKPOINT_PATH, device)
        model.train()  # load_checkpoint returns the model in eval mode, ready for generation
    else:
        model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size)).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{num_params:,} parameters on {device}")

    # Built over the parameters of the model being trained, before any saved state is loaded
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )

    # Keep the model that generalises best, since later steps may overfit and do worse
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss = load_training_state(LAST_CHECKPOINT_PATH, optimizer)
        print(f"resuming from step {start_step}")

    # One extra pass so the finished model is evaluated too, after max_steps updates
    for step in range(start_step, train_config.max_steps + 1):
        is_last_step = step == train_config.max_steps
        if step % train_config.eval_interval == 0 or is_last_step:
            losses = estimate_loss(model, train_ids, val_ids, train_config, device)
            report = f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}"
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    BEST_CHECKPOINT_PATH,
                    model,
                    tokenizer,
                    optimizer,
                    train_config,
                    step,
                    best_val_loss,
                )
                report += " | new best"
            # Saved before this step's update, so resuming repeats this evaluation exactly
            save_checkpoint(
                LAST_CHECKPOINT_PATH, model, tokenizer, optimizer, train_config, step, best_val_loss
            )
            print(report)
        if is_last_step:
            break

        inputs, targets = get_batch(
            train_ids, train_config.batch_size, model.config.context_length, device
        )
        _, loss = model(inputs, targets)

        optimizer.zero_grad()  # backward() adds to .grad, so clear the previous step's
        loss.backward()  # set each parameter's .grad to d(loss)/d(parameter)
        optimizer.step()  # update each parameter using its .grad

    print(f"best val {best_val_loss:.4f}, saved to {BEST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
