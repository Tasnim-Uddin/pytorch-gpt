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
