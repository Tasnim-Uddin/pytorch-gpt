"""Tests for the transformer layers and the GPT model."""

import math

import pytest
import torch

from pytorch_gpt.config import GPTConfig
from pytorch_gpt.model import GPT, CausalSelfAttention, FeedForward, TransformerBlock

BATCH_SIZE = 2
CONTEXT_LENGTH = 16
D_MODEL = 64
NUM_HEADS = 4
D_FF = 4 * D_MODEL
NUM_LAYERS = 2
VOCAB_SIZE = 10
GPT_CONFIG = GPTConfig(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
)


def make_attention() -> CausalSelfAttention:
    torch.manual_seed(0)
    return CausalSelfAttention(D_MODEL, NUM_HEADS, CONTEXT_LENGTH)


def make_feed_forward() -> FeedForward:
    torch.manual_seed(0)
    return FeedForward(D_MODEL, D_FF)


def make_transformer_block() -> TransformerBlock:
    torch.manual_seed(0)
    return TransformerBlock(D_MODEL, NUM_HEADS, CONTEXT_LENGTH, D_FF)


def make_gpt() -> GPT:
    torch.manual_seed(0)
    return GPT(GPT_CONFIG)


def make_token_ids(length: int = CONTEXT_LENGTH) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, length))


def assert_causal(y: torch.Tensor, y_changed: torch.Tensor, t: int) -> None:
    """Check outputs for inputs that differ only after position t."""
    # Outputs up to position t must not change
    assert torch.allclose(y[:, : t + 1], y_changed[:, : t + 1], atol=1e-6)
    # Outputs after t should change (sanity check that the test does something)
    assert not torch.allclose(y[:, t + 1 :], y_changed[:, t + 1 :])


# Layers that map (B, T, D) to (B, T, D), and the subset that must not look ahead
ALL_LAYERS = [
    pytest.param(make_attention, id="attention"),
    pytest.param(make_feed_forward, id="feed_forward"),
    pytest.param(make_transformer_block, id="transformer_block"),
]
CAUSAL_LAYERS = [
    pytest.param(make_attention, id="attention"),
    pytest.param(make_transformer_block, id="transformer_block"),
]


@pytest.mark.parametrize("make_layer", ALL_LAYERS)
@pytest.mark.parametrize("length", [CONTEXT_LENGTH, CONTEXT_LENGTH // 2])
def test_layer_preserves_shape(make_layer, length):
    layer = make_layer()
    x = torch.randn(BATCH_SIZE, length, D_MODEL)
    assert layer(x).shape == (BATCH_SIZE, length, D_MODEL)


@pytest.mark.parametrize("make_layer", CAUSAL_LAYERS)
def test_layer_is_causal(make_layer):
    layer = make_layer()
    x = torch.randn(BATCH_SIZE, CONTEXT_LENGTH, D_MODEL)

    # Change every token after position t
    t = 7
    x_changed = x.clone()
    x_changed[:, t + 1 :] = torch.randn(BATCH_SIZE, CONTEXT_LENGTH - t - 1, D_MODEL)

    with torch.no_grad():
        assert_causal(layer(x), layer(x_changed), t)


def test_attention_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        CausalSelfAttention(D_MODEL, num_heads=5, context_length=CONTEXT_LENGTH)


def test_feed_forward_is_position_wise():
    ffn = make_feed_forward()
    x = torch.randn(BATCH_SIZE, CONTEXT_LENGTH, D_MODEL)

    # Change only the token at position t; x_changed[:, t] has shape (B, D_MODEL)
    t = 7
    x_changed = x.clone()
    x_changed[:, t] = torch.randn(BATCH_SIZE, D_MODEL)

    with torch.no_grad():
        y = ffn(x)
        y_changed = ffn(x_changed)

    # Outputs at every other position must not change
    others = torch.arange(CONTEXT_LENGTH) != t  # (T,) bool, False only at t
    assert torch.allclose(y[:, others], y_changed[:, others], atol=1e-6)
    # Output at t should change (sanity check that the test does something)
    assert not torch.allclose(y[:, t], y_changed[:, t])


@pytest.mark.parametrize("length", [CONTEXT_LENGTH, CONTEXT_LENGTH // 2])
def test_gpt_logits_shape(length):
    gpt = make_gpt()
    logits, _ = gpt(make_token_ids(length))
    assert logits.shape == (BATCH_SIZE, length, VOCAB_SIZE)


def test_gpt_loss_is_none_without_targets():
    gpt = make_gpt()
    _, loss = gpt(make_token_ids())
    assert loss is None


def test_gpt_initial_loss_is_near_uniform():
    gpt = make_gpt()

    # Random targets leave nothing to learn, so an untrained model should spread its
    # probability evenly: p = 1/V for every token, giving a loss of -log(1/V) = log(V)
    _, loss = gpt(make_token_ids(), targets=make_token_ids())
    assert abs(loss.item() - math.log(VOCAB_SIZE)) < 0.5


def test_gpt_is_causal():
    gpt = make_gpt()
    token_ids = make_token_ids()

    # Change every token after position t
    t = 7
    token_ids_changed = token_ids.clone()
    token_ids_changed[:, t + 1 :] = make_token_ids(CONTEXT_LENGTH - t - 1)

    with torch.no_grad():
        logits, _ = gpt(token_ids)
        logits_changed, _ = gpt(token_ids_changed)

    assert_causal(logits, logits_changed, t)


def test_gpt_rejects_too_long_sequence():
    gpt = make_gpt()
    with pytest.raises(ValueError):
        gpt(make_token_ids(CONTEXT_LENGTH + 1))
