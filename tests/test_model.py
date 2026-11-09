import torch

from pytorch_gpt.model import CausalSelfAttention

D_MODEL = 64
NUM_HEADS = 4
CONTEXT_LENGTH = 16


def make_attention() -> CausalSelfAttention:
    torch.manual_seed(0)
    return CausalSelfAttention(D_MODEL, NUM_HEADS, CONTEXT_LENGTH)


def test_attention_output_shape():
    attn = make_attention()
    x = torch.randn(2, CONTEXT_LENGTH, D_MODEL)
    assert attn(x).shape == (2, CONTEXT_LENGTH, D_MODEL)


def test_attention_shorter_sequence():
    attn = make_attention()
    x = torch.randn(2, 10, D_MODEL)
    assert attn(x).shape == (2, 10, D_MODEL)


def test_attention_is_causal():
    attn = make_attention()
    x = torch.randn(2, CONTEXT_LENGTH, D_MODEL)

    # Change every token after position t
    t = 7
    x_changed = x.clone()
    x_changed[:, t + 1 :] = torch.randn(2, CONTEXT_LENGTH - t - 1, D_MODEL)

    with torch.no_grad():
        y = attn(x)
        y_changed = attn(x_changed)

    # Outputs up to position t must not change
    assert torch.allclose(y[:, : t + 1], y_changed[:, : t + 1], atol=1e-6)
    # Outputs after t should change (sanity check that the test does something)
    assert not torch.allclose(y[:, t + 1 :], y_changed[:, t + 1 :])
