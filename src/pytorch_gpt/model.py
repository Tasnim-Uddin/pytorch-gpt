"""Transformer layers for the GPT model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_gpt.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention in which each token sees only itself and earlier tokens.

    Args:
        d_model: Width of each token's vector, D.
        num_heads: Number of attention heads, H; must divide `d_model`.
        context_length: Longest sequence the layer accepts; sets the causal mask size.

    Raises:
        ValueError: If `d_model` is not divisible by `num_heads`.
    """

    def __init__(self, d_model: int, num_heads: int, context_length: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # width of each head

        # Each maps (B, T, D) -> (B, T, D); the heads are split out of the result afterwards
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)  # mixes the merged heads back together

        # True where query i may attend to key j (j <= i): the lower triangle, diagonal included
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))

        # A buffer follows the module across devices but is not trained;
        # non-persistent keeps it out of the state dict, since it is rebuilt on init
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix information across positions, looking only backwards.

        Args:
            x: Input of shape (B, T, D), with T <= context_length.

        Returns:
            Output of shape (B, T, D).
        """
        B, T, D = x.shape

        # Project, then split heads: (B, T, D) -> (B, T, H, d_k) -> (B, H, T, d_k)
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product scores: (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
        # Dividing by sqrt(d_k) keeps their variance near 1, so softmax does not saturate
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        # Hide future keys: -inf becomes 0 after softmax
        # Slice the mask in case T < context_length; (T, T) broadcasts over (B, H)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))

        # Row i holds how much query i takes from each key; every row sums to 1
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)

        # Weighted sum of values: (B, H, T, T) @ (B, H, T, d_k) -> (B, H, T, d_k)
        out = attn_weights @ v

        # Merge heads: (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, D)
        # transpose only changes strides, and view needs contiguous memory
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out)  # (B, T, D)


class FeedForward(nn.Module):
    """Position-wise feed-forward network: expand, apply GELU, project back.

    Args:
        d_model: Width of each token's vector, D.
        d_ff: Width of the hidden layer, usually 4 * d_model.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform each position independently, with the same weights at every position.

        Args:
            x: Input of shape (B, T, D).

        Returns:
            Output of shape (B, T, D).
        """
        # nn.Linear acts on the last dimension only; B and T are treated as batch dimensions
        x = self.up_proj(x)  # (B, T, D) -> (B, T, d_ff)
        x = F.gelu(x)  # elementwise, no learned parameters
        return self.down_proj(x)  # (B, T, d_ff) -> (B, T, D)


class TransformerBlock(nn.Module):
    """One transformer layer: causal self-attention, then a feed-forward network.

    Uses pre-norm residuals: each sublayer reads a normalised copy of x and adds its
    output back onto x, so x itself is never normalised inside the block.

    Args:
        d_model: Width of each token's vector, D.
        num_heads: Number of attention heads, H.
        context_length: Longest sequence the block accepts.
        d_ff: Width of the feed-forward hidden layer.
    """

    def __init__(self, d_model: int, num_heads: int, context_length: int, d_ff: int) -> None:
        super().__init__()
        # Each LayerNorm owns its own learned gain and bias, both of shape (D,)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, context_length)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and then the feed-forward network, each as a residual update.

        Args:
            x: Input of shape (B, T, D), with T <= context_length.

        Returns:
            Output of shape (B, T, D).
        """
        x = x + self.attn(self.attn_norm(x))  # tokens exchange information
        x = x + self.ffn(self.ffn_norm(x))  # each token is processed on its own
        return x  # (B, T, D)


class GPT(nn.Module):
    """Decoder-only transformer that predicts the next token at every position.

    Args:
        config: Architecture settings.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Lookup tables: one row per token ID, and one row per position
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)

        # Separate blocks, each with its own weights; ModuleList registers their parameters
        self.blocks = nn.ModuleList(
            TransformerBlock(config.d_model, config.num_heads, config.context_length, config.d_ff)
            for _ in range(config.num_layers)
        )

        # Pre-norm never normalises the residual stream, so it is normalised once at the end
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute next-token logits, and the loss if targets are given.

        Args:
            token_ids: Input token IDs of shape (B, T), with T <= context_length.
            targets: Next-token IDs of shape (B, T), or None to skip the loss.

        Returns:
            Logits of shape (B, T, V), and the mean cross-entropy loss as a scalar
            tensor, or None when `targets` is None.

        Raises:
            ValueError: If T is longer than `context_length`.
        """
        T = token_ids.shape[1]
        if T > self.config.context_length:
            raise ValueError(
                f"sequence length {T} exceeds context_length ({self.config.context_length})"
            )

        # What each token is plus where it is: (B, T, D) + (T, D) broadcasts to (B, T, D)
        positions = torch.arange(T, device=token_ids.device)  # (T,)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)  # (B, T, D)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, D) -> (B, T, V)

        if targets is None:
            return logits, None

        # cross_entropy takes (N, V) scores and (N,) IDs, so merge B and T into N = B * T
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss
