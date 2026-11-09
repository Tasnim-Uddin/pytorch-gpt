"""Configuration dataclasses that group related settings."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class GPTConfig:
    """Architecture of a GPT model; the defaults suit character-level Tiny Shakespeare.

    Attributes:
        vocab_size: Number of distinct tokens, V; taken from the tokenizer.
        context_length: Longest sequence the model accepts, in tokens.
        d_model: Width of each token's vector, D.
        num_heads: Attention heads per block, H; must divide `d_model`.
        num_layers: Number of transformer blocks.
        d_ff: Width of the feed-forward hidden layer, usually 4 * d_model.
    """

    vocab_size: int
    context_length: int = 256
    d_model: int = 384
    num_heads: int = 6
    num_layers: int = 6
    d_ff: int = 1536


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """How a model is trained; the defaults suit the default `GPTConfig` on one GPU.

    Attributes:
        batch_size: Sequences per training step, B.
        lr: Learning rate for AdamW.
        weight_decay: How strongly AdamW shrinks weights towards zero each step.
        max_steps: Total number of optimizer steps.
        eval_interval: Training steps between loss evaluations.
        eval_steps: Batches averaged per split in each evaluation.
        seed: Random seed for weight initialisation and batch sampling.
    """

    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.1
    max_steps: int = 5000
    eval_interval: int = 250
    eval_steps: int = 200
    seed: int = 0
