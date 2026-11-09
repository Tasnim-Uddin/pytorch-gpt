# Conventions

Dimensions take the paper's `d_` names, counts start with `num_`, a short list of standard
abbreviations is allowed, and everything else is spelled out.

## Shape letters

Used only in shape comments, docstrings and `x.shape` unpacking.

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `T` | sequence length of the current batch (≤ `context_length`) |
| `D` | `d_model` |
| `H` | `num_heads` |
| `V` | `vocab_size` |
| `N` | tokens in a dataset split |
| `d_k`, `d_ff` | written out in full |

```python
B, T, D = x.shape
# (B, T, D) -> (B, H, T, d_k)
```

## Hyperparameters

Constructor arguments and config fields: `vocab_size`, `context_length`, `d_model`, `num_heads`,
`d_ff`, `num_layers`, `dropout`. Later: `num_kv_heads`, `norm_eps`, `rope_theta`.

Counts use `num_`, never `n_`. Derived values such as `d_k = d_model // num_heads` are not arguments.

## Classes and attributes

Learned linear layers end in `_proj`. Norms are named after the sublayer they feed, so the names
still hold when LayerNorm is swapped for RMSNorm.

| Class | Attributes |
|---|---|
| `CausalSelfAttention` | `q_proj`, `k_proj`, `v_proj`, `out_proj`, buffer `causal_mask` |
| `FeedForward` | `up_proj` (D → d_ff), `down_proj` (d_ff → D); later `gate_proj` |
| `TransformerBlock` | `attn_norm`, `attn`, `ffn_norm`, `ffn` |
| `GPT` | `token_embedding`, `position_embedding`, `blocks`, `final_norm`, `lm_head` |
| `GPTConfig`, `TrainConfig` | settings dataclasses |

## Variables

- **Token IDs** end in `_ids`: `train_ids`, `val_ids`, `token_ids`, `next_ids`. Bare `ids` only
  when nothing more specific fits.
- **Batches** are `inputs, targets`. `x` is reserved for float activations inside the model.
- **Inside the model:** `x` is the hidden state throughout; then `q, k, v` → `scores` (before
  softmax) → `attn_weights` (after softmax) → `out`. At the end: `logits` (B, T, V) and `loss`.
  Also `positions` and `probs`.
- **Training:** `step` (optimizer steps, not epochs), `max_steps`, `lr`, `weight_decay`,
  `warmup_steps`, `grad_clip`, `eval_interval`, `eval_steps`, `train_loss`, `val_loss`.
  Splits are `train` and `val`, never `valid` or `dev`.
- **Generation:** `max_new_tokens`, `temperature`, `top_k`, `top_p`.

Allowed abbreviations: `q k v`, `attn`, `ffn`, `proj`, `norm`, `lr`, `ids`, `val`, `num_`, `d_`,
and the shape letters.

## General Python

- Functions are verbs (`load_data`, `get_batch`, `generate`); classes are CamelCase nouns.
- Constants are `UPPER_CASE`; boolean flags start with `use_`.
- Tests live in `tests/test_<module>.py` as `test_<unit>_<behaviour>`.

## Spelling and comments

- Identifiers use American spelling (`optimizer`, `normalize`) to match PyTorch.
- Prose (comments, docstrings, commits, docs) uses British spelling.
- Docstrings follow Google style: full sentences ending in a full stop, with shapes as
  "of shape (B, T, D)".
- Inline comments are short, start with a capital and have no full stop. Shapes trail the line:
  `# (B, T, D)`.
