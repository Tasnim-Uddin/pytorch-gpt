import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, context_length: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # True where query i may attend to key j (j <= i)
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # C == d_model

        # Project, then split heads: (B, T, C) -> (B, T, H, d_k) -> (B, H, T, d_k)
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product scores: (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        # Slice the mask in case T < context_length; (T, T) broadcasts over (B, H)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, H, T, T)
        out = attn @ v  # (B, H, T, d_k)

        # Merge heads: (B, H, T, d_k) -> (B, T, H, d_k) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)
