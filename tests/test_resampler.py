"""Stage B gate: FusionResampler shapes, frame masking, and gradient flow."""

import torch

from fusion_embedding.config import FusionConfig
from fusion_embedding.model import FusionResampler


def test_output_shape_1280_to_2048():
    cfg = FusionConfig()  # production dims: 1280 -> 2048, N=64
    rs = FusionResampler(cfg)
    frames = torch.randn(2, 137, cfg.d_audio)
    out = rs(frames)
    assert out.shape == (2, cfg.n_query, cfg.d_llm) == (2, 64, 2048)


def test_param_count_single_digit_millions():
    cfg = FusionConfig()
    rs = FusionResampler(cfg)
    n = sum(p.numel() for p in rs.parameters())
    assert 1e6 < n < 1e7, f"expected single-digit millions, got {n:,}"


def test_variable_length_same_n():
    cfg = FusionConfig.tiny()
    rs = FusionResampler(cfg)
    for T in (1, 5, 50):
        out = rs(torch.randn(3, T, cfg.d_audio))
        assert out.shape == (3, cfg.n_query, cfg.d_llm)


def test_padding_invariance():
    """Tokens for the real (unmasked) frames must not depend on masked padding frames."""
    cfg = FusionConfig.tiny()
    rs = FusionResampler(cfg).eval()
    real = torch.randn(1, 6, cfg.d_audio)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :6] = True

    a = torch.cat([real, torch.randn(1, 4, cfg.d_audio)], dim=1)   # garbage in padding
    b = torch.cat([real, torch.zeros(1, 4, cfg.d_audio)], dim=1)   # zeros in padding
    with torch.no_grad():
        oa = rs(a, mask)
        ob = rs(b, mask)
    assert torch.allclose(oa, ob, atol=1e-5)


def test_fully_masked_row_is_finite():
    cfg = FusionConfig.tiny()
    rs = FusionResampler(cfg)
    frames = torch.randn(2, 8, cfg.d_audio)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[1] = False                                                # row 1 fully masked
    out = rs(frames, mask)
    assert torch.isfinite(out).all()


def test_gradients_flow():
    cfg = FusionConfig.tiny()
    rs = FusionResampler(cfg)
    out = rs(torch.randn(2, 9, cfg.d_audio))
    out.sum().backward()
    assert rs.queries.grad is not None and rs.queries.grad.abs().sum() > 0
    assert rs.in_proj.weight.grad is not None
