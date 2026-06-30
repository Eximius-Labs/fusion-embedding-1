"""Stage D gate: InfoNCE-over-MRL + CORAL — correctness, symmetry, debias reduction, grads."""

import math

import pytest
import torch
import torch.nn.functional as F

from fusion_embedding.config import FusionConfig
from fusion_embedding.losses import FusionContrastiveLoss, coral_penalty, _infonce_directional


def _two_views(B=8, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    audio = torch.randn(B, d, generator=g)
    text = torch.randn(B, d, generator=g)
    return audio, text


def test_debias_zero_equals_plain_infonce():
    """γ⁺=0 with no hard negatives must reduce exactly to softmax cross-entropy InfoNCE."""
    B, d = 8, 16
    a = F.normalize(torch.randn(B, d), dim=-1)
    t = F.normalize(torch.randn(B, d), dim=-1)
    scale = torch.tensor(14.0)
    got = _infonce_directional(a, t, scale, tau_plus=0.0).mean()
    logits = scale * (a @ t.T)
    want = F.cross_entropy(logits, torch.arange(B))
    assert torch.allclose(got, want, atol=1e-5)


def test_perfect_alignment_low_loss():
    cfg = FusionConfig.tiny()
    loss_fn = FusionContrastiveLoss(cfg)
    a, _ = _two_views(B=8, d=cfg.d_llm)
    # text == audio scaled: diagonal dominates -> near-zero InfoNCE at high scale
    aligned = a.clone()
    ls = torch.tensor(math.log(50.0))
    loss, m = loss_fn(a, aligned, ls)
    assert m["acc_a2t"].item() == 1.0
    assert m["infonce"].item() < 0.1


def test_symmetry_invariance():
    """Swapping audio<->text leaves the symmetric InfoNCE unchanged (CORAL off)."""
    cfg = FusionConfig.tiny(lambda_coral=0.0)
    loss_fn = FusionContrastiveLoss(cfg)
    a, t = _two_views(B=8, d=cfg.d_llm)
    ls = torch.tensor(2.0)
    l1, _ = loss_fn(a, t, ls)
    l2, _ = loss_fn(t, a, ls)
    assert torch.allclose(l1, l2, atol=1e-5)


def test_mrl_tiling_uses_all_rungs():
    """Loss must reflect every rung: zeroing a non-prefix rung's weight changes the value."""
    cfg_all = FusionConfig.tiny()
    cfg_one = FusionConfig.tiny(mrl_weights=(1.0, 0.0, 0.0))  # only the 32-dim rung
    a, t = _two_views(B=6, d=cfg_all.d_llm)
    ls = torch.tensor(2.0)
    l_all, _ = FusionContrastiveLoss(cfg_all)(a, t, ls)
    l_one, _ = FusionContrastiveLoss(cfg_one)(a, t, ls)
    assert not torch.allclose(l_all, l_one)


def test_coral_zero_when_identical():
    x = torch.randn(10, 16)
    assert coral_penalty(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_coral_contributes():
    cfg_on = FusionConfig.tiny(lambda_coral=1.0)
    cfg_off = FusionConfig.tiny(lambda_coral=0.0)
    a, t = _two_views(B=8, d=cfg_on.d_llm)
    ls = torch.tensor(2.0)
    l_on, m_on = FusionContrastiveLoss(cfg_on)(a, t, ls)
    l_off, _ = FusionContrastiveLoss(cfg_off)(a, t, ls)
    assert m_on["coral"].item() > 0
    assert l_on.item() > l_off.item()


def test_hard_negatives_increase_loss():
    """Adding confusable texts to the A→T denominator can only raise (never lower) the loss."""
    cfg = FusionConfig.tiny(lambda_coral=0.0)
    loss_fn = FusionContrastiveLoss(cfg)
    a, t = _two_views(B=8, d=cfg.d_llm)
    ls = torch.tensor(3.0)
    base, _ = loss_fn(a, t, ls)
    hard = torch.randn(8, 4, cfg.d_llm)               # K=4 hard negs per anchor
    withhn, _ = loss_fn(a, t, ls, hard_neg_text=hard)
    assert withhn.item() >= base.item() - 1e-6
    assert withhn.item() > base.item()


def test_debias_changes_and_stays_finite():
    cfg = FusionConfig.tiny(debias_gamma=0.1, lambda_coral=0.0)
    cfg0 = FusionConfig.tiny(debias_gamma=0.0, lambda_coral=0.0)
    a, t = _two_views(B=8, d=cfg.d_llm)
    ls = torch.tensor(2.0)
    ld, _ = FusionContrastiveLoss(cfg)(a, t, ls)
    l0, _ = FusionContrastiveLoss(cfg0)(a, t, ls)
    assert torch.isfinite(ld)
    assert not torch.allclose(ld, l0)


def test_gradients_flow_to_inputs_and_scale():
    cfg = FusionConfig.tiny()
    loss_fn = FusionContrastiveLoss(cfg)
    a = torch.randn(8, cfg.d_llm, requires_grad=True)
    t = torch.randn(8, cfg.d_llm, requires_grad=True)
    ls = torch.tensor(2.0, requires_grad=True)
    loss, _ = loss_fn(a, t, ls)
    loss.backward()
    assert a.grad is not None and a.grad.abs().sum() > 0
    assert t.grad is not None and t.grad.abs().sum() > 0
    assert ls.grad is not None and ls.grad.abs() > 0


def test_high_scale_is_stable():
    """Even at the clamp ceiling (scale up to 100) the loss must not overflow to inf/nan."""
    cfg = FusionConfig.tiny()
    loss_fn = FusionContrastiveLoss(cfg)
    a, t = _two_views(B=8, d=cfg.d_llm)
    ls = torch.tensor(math.log(100.0))               # logit_scale_max
    loss, m = loss_fn(a, t, ls)
    assert torch.isfinite(loss)
    assert torch.isfinite(m["infonce"])
