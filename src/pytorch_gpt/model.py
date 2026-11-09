"""Transformer layers for the GPT model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
