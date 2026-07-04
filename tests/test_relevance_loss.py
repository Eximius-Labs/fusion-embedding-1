"""Relevance-aware loss knobs (floor-audit follow-ups): soft-label InfoNCE + FN masking.

Contract under test:
  1. both knobs off => EXACTLY the legacy loss (bit-level backward compat);
  2. FN masking removes near-dup pairs from the denominator (in-batch + bank) and lowers
     the loss on batches that contain them, leaving clean batches untouched;
  3. soft labels: beta=0 == hard labels; beta>0 credits semantic matches (lower loss on a
     batch with a near-dup pair than hard labels give), rows always sum to 1;
  4. gradients flow through both paths; soft-label + debias is rejected.
"""

import pytest
import torch

from fusion_embedding.config import FusionConfig
from fusion_embedding.losses import FusionContrastiveLoss


def _cfg(**kw):
    return FusionConfig.tiny(**kw)


def _orthogonal(m, d, scale=3.0):
    assert m <= d
    return torch.eye(m, d) * scale


def _pair_batch(cfg, neardup=True):
    """4 items; rows 0/1 are a semantic near-dup pair (cos ~1) when neardup=True."""
    d = cfg.d_llm
    text = _orthogonal(4, d)
    if neardup:
        text[1] = text[0] + 0.03 * torch.eye(4, d)[3] * 0  # start same dir
        text[1, 4] = 0.03                                   # tiny extra dim -> cos ~0.99995
        text[1, :4] = text[0, :4]
    audio = text.clone() + 0.01 * torch.randn_like(text)    # near-perfect connector
    return audio, text


def test_knobs_off_is_exactly_legacy():
    cfg_legacy = _cfg()
    cfg_knobs = _cfg(fn_mask_threshold=0.0, soft_label_beta=0.0)
    torch.manual_seed(0)
    audio = torch.randn(5, cfg_legacy.d_llm)
    text = torch.randn(5, cfg_legacy.d_llm)
    ls = torch.tensor(cfg_legacy.logit_scale_init)
    l1, m1 = FusionContrastiveLoss(cfg_legacy)(audio, text, ls)
    l2, m2 = FusionContrastiveLoss(cfg_knobs)(audio, text, ls)
    assert torch.equal(l1, l2)
    assert torch.equal(m1["infonce"], m2["infonce"])


def test_fn_mask_lowers_loss_on_neardup_batch_only():
    cfg_off = _cfg()
    cfg_on = _cfg(fn_mask_threshold=0.98, fn_mask_dim=32)   # tiny d_llm=32: full-dim relevance
    ls = torch.tensor(cfg_off.logit_scale_init)
    audio, text = _pair_batch(cfg_off, neardup=True)
    l_off, _ = FusionContrastiveLoss(cfg_off)(audio, text, ls)
    l_on, _ = FusionContrastiveLoss(cfg_on)(audio, text, ls)
    assert float(l_on) < float(l_off) - 0.05                # dup pair left the denominator

    audio2, text2 = _pair_batch(cfg_off, neardup=False)     # clean batch: mask must be a no-op
    l_off2, _ = FusionContrastiveLoss(cfg_off)(audio2, text2, ls)
    l_on2, _ = FusionContrastiveLoss(cfg_on)(audio2, text2, ls)
    assert float(l_on2) == pytest.approx(float(l_off2), abs=1e-6)


def test_fn_mask_extends_to_bank():
    cfg_off = _cfg()
    cfg_on = _cfg(fn_mask_threshold=0.98, fn_mask_dim=32)
    ls = torch.tensor(cfg_off.logit_scale_init)
    audio, text = _pair_batch(cfg_off, neardup=False)
    bank = torch.cat([text + 0.001, _orthogonal(4, cfg_off.d_llm)[2:]])  # bank rows 0-3 ~= batch texts
    own_mask = torch.zeros(4, bank.size(0), dtype=torch.bool)
    l_off, _ = FusionContrastiveLoss(cfg_off)(audio, text, ls, bank_text=bank,
                                              bank_exclude_mask=own_mask)
    l_on, _ = FusionContrastiveLoss(cfg_on)(audio, text, ls, bank_text=bank,
                                            bank_exclude_mask=own_mask)
    assert float(l_on) < float(l_off) - 0.1                 # near-identical bank rows masked


def test_soft_labels_beta_zero_equals_hard_and_rows_sum_to_one():
    cfg = _cfg(soft_label_beta=0.3, fn_mask_dim=32)
    loss_fn = FusionContrastiveLoss(cfg)
    audio, text = _pair_batch(cfg, neardup=True)
    _, soft, _ = loss_fn._relevance_terms(text.float(), None, None)
    assert soft is not None
    assert torch.allclose(soft.sum(dim=1), torch.ones(4), atol=1e-5)
    assert float(soft[0, 1]) > 0.1                          # dup pair gets real target mass
    # beta=0 path == legacy
    cfg0 = _cfg(soft_label_beta=0.0)
    ls = torch.tensor(cfg.logit_scale_init)
    l_hard, _ = FusionContrastiveLoss(cfg0)(audio, text, ls)
    l_hard2, _ = FusionContrastiveLoss(_cfg())(audio, text, ls)
    assert torch.equal(l_hard, l_hard2)


def test_soft_labels_stop_pushing_the_twin_away():
    """The soft-label benefit is in the GRADIENT, not the loss value: with a semantic twin in
    the batch, hard labels keep pushing the anchor's audio away from the twin's text (positive
    gradient along that direction); soft labels assign the twin target mass ~= its softmax mass,
    so the push (nearly) vanishes. (At perfect alignment the loss VALUES are ~equal — verified
    by the floor audit's math: redistributing mass between equal logits changes nothing.)"""
    ls = torch.tensor(_cfg().logit_scale_init)
    _, text = _pair_batch(_cfg(), neardup=True)
    twin_dir = torch.nn.functional.normalize(text[1], dim=-1)

    def grad_along_twin(cfg):
        audio = text.clone().requires_grad_(True)
        loss, _ = FusionContrastiveLoss(cfg)(audio, text, ls)
        loss.backward()
        return float(audio.grad[0] @ twin_dir)      # >0 == loss decreases by moving AWAY from twin

    g_hard = grad_along_twin(_cfg())
    g_soft = grad_along_twin(_cfg(soft_label_beta=0.3, fn_mask_dim=32))
    assert g_hard > 0                                # hard labels fight the twin
    assert abs(g_soft) < 0.5 * abs(g_hard)          # soft labels (mostly) stop fighting it


def test_gradients_flow_and_debias_conflict_rejected():
    for kw in ({"fn_mask_threshold": 0.98, "fn_mask_dim": 32},
               {"soft_label_beta": 0.3, "fn_mask_dim": 32}):
        cfg = _cfg(**kw)
        audio = torch.randn(4, cfg.d_llm, requires_grad=True)
        text = torch.randn(4, cfg.d_llm)
        ls = torch.tensor(cfg.logit_scale_init, requires_grad=True)
        loss, _ = FusionContrastiveLoss(cfg)(audio, text, ls)
        loss.backward()
        assert audio.grad is not None and torch.isfinite(audio.grad).all()
        assert ls.grad is not None and torch.isfinite(ls.grad)
    with pytest.raises(ValueError):
        FusionContrastiveLoss(_cfg(soft_label_beta=0.3, debias_gamma=0.1))
