"""Stage #3 gate: the frozen-text memory bank — FIFO mechanics, loss math, and the
small-micro-batch payoff (the lever that makes 8GB contrastive training viable)."""

import pytest
import torch
import torch.nn.functional as F

from fusion_embedding.config import FusionConfig
from fusion_embedding.losses import FusionContrastiveLoss, _infonce_directional
from fusion_embedding.memory_bank import TextMemoryBank, precompute_text_bank
from fusion_embedding.data import make_synthetic_dataset
from fusion_embedding._tiny import build_tiny_model


# ----------------------------- FIFO mechanics ------------------------------ #
def test_bank_starts_empty():
    bank = TextMemoryBank(dim=8, capacity=5)
    assert bank.is_empty and len(bank) == 0 and bank.get() is None


def test_bank_enqueue_and_get():
    bank = TextMemoryBank(dim=4, capacity=6)
    bank.enqueue(torch.ones(3, 4))
    assert len(bank) == 3 and bank.get().shape == (3, 4)


def test_bank_wraparound_keeps_most_recent():
    bank = TextMemoryBank(dim=1, capacity=4)
    for v in range(6):                       # enqueue scalars 0..5 one at a time
        bank.enqueue(torch.full((1, 1), float(v)))
    assert len(bank) == 4
    got = set(bank.get().flatten().tolist())
    assert got == {2.0, 3.0, 4.0, 5.0}       # oldest (0,1) evicted


def test_bank_batch_larger_than_capacity():
    bank = TextMemoryBank(dim=2, capacity=3)
    emb = torch.arange(10 * 2, dtype=torch.float32).reshape(10, 2)
    bank.enqueue(emb)
    assert len(bank) == 3
    assert torch.equal(bank.get(), emb[-3:])  # keeps the last `capacity`


def test_bank_stores_detached():
    bank = TextMemoryBank(dim=3, capacity=4)
    x = torch.randn(2, 3, requires_grad=True)
    bank.enqueue(x * 2)
    assert not bank.get().requires_grad


# ------------------------------- loss math --------------------------------- #
def test_bank_negatives_increase_loss():
    cfg = FusionConfig.tiny(lambda_coral=0.0)
    loss_fn = FusionContrastiveLoss(cfg)
    a = torch.randn(4, cfg.d_llm)
    t = torch.randn(4, cfg.d_llm)
    ls = torch.tensor(3.0)
    base, _ = loss_fn(a, t, ls)
    bank = torch.randn(32, cfg.d_llm)                 # 32 shared negatives
    withbank, _ = loss_fn(a, t, ls, bank_text=bank)
    assert withbank.item() > base.item()              # more true negatives -> higher loss


def test_bank_gradients_flow_to_audio_and_scale():
    cfg = FusionConfig.tiny(lambda_coral=0.0)
    loss_fn = FusionContrastiveLoss(cfg)
    a = torch.randn(2, cfg.d_llm, requires_grad=True)
    t = torch.randn(2, cfg.d_llm, requires_grad=True)
    ls = torch.tensor(2.0, requires_grad=True)
    bank = torch.randn(50, cfg.d_llm)
    loss, _ = loss_fn(a, t, ls, bank_text=bank)
    loss.backward()
    assert a.grad.abs().sum() > 0 and ls.grad.abs() > 0
    assert torch.isfinite(loss)


# ---------- the small-micro-batch payoff (deterministic mechanism) --------- #
def test_bank_provides_signal_when_inbatch_has_none():
    """The core value proposition for 8GB: at micro-batch 1 (B=1) there are NO in-batch
    negatives, so plain InfoNCE has zero loss and zero gradient — nothing to learn from.
    The frozen-text bank supplies negatives, producing a real loss and a real gradient.
    """
    torch.manual_seed(0)
    d = 16
    scale = torch.tensor(20.0)
    a = F.normalize(torch.randn(1, d), dim=-1)          # one audio anchor
    t = F.normalize(torch.randn(1, d), dim=-1)          # its positive text
    bank = F.normalize(torch.randn(32, d), dim=-1)      # frozen-text negatives

    # B=1, no bank -> loss is identically 0 (positive is its own only "candidate")
    loss_nobank = _infonce_directional(a, t, scale, tau_plus=0.0).mean()
    assert loss_nobank.item() == pytest.approx(0.0, abs=1e-6)

    # B=1, with bank -> real positive loss
    loss_bank = _infonce_directional(a, t, scale, tau_plus=0.0, shared_neg=bank).mean()
    assert loss_bank.item() > 0.1

    # ...and a real gradient where there was none
    a_no = a.clone().requires_grad_(True)
    _infonce_directional(a_no, t, scale, tau_plus=0.0).mean().backward()
    assert a_no.grad is None or a_no.grad.abs().sum().item() == pytest.approx(0.0, abs=1e-6)

    a_bk = a.clone().requires_grad_(True)
    _infonce_directional(a_bk, t, scale, tau_plus=0.0, shared_neg=bank).mean().backward()
    assert a_bk.grad.abs().sum().item() > 0


def test_precompute_text_bank_shape_and_frozen():
    """precompute_text_bank embeds every item once (frozen tower) -> [M, d_llm], reusable."""
    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg)
    manifest, collator = make_synthetic_dataset(cfg, n=10)
    bank = precompute_text_bank(model, manifest, collator)
    assert bank.shape == (10, cfg.d_llm)
    # frozen text tower => deterministic across calls
    bank2 = precompute_text_bank(model, manifest, collator)
    assert torch.allclose(bank, bank2)
    # normalized variant truncates to a rung
    bank_n = precompute_text_bank(model, manifest, collator, normalize_dim=cfg.mrl_default)
    assert bank_n.shape == (10, cfg.mrl_default)
    assert torch.allclose(bank_n.norm(dim=-1), torch.ones(10), atol=1e-5)


def test_train_loop_accepts_live_memory_bank():
    """Integration: train_stage1 with a live TextMemoryBank runs, stays finite, base frozen."""
    from fusion_embedding.train_stage1 import build_tiny_training_setup, train_stage1

    cfg = FusionConfig.tiny(max_steps=40, d_resampler=32)
    s = build_tiny_training_setup(cfg, n_train=8, batch_size=4, seed=0)
    bank = TextMemoryBank(dim=cfg.d_llm, capacity=16)

    state = train_stage1(
        s.model, s.train_loader, s.loss_fn, cfg,
        steps=40, eval_fn=s.eval_fn, device="cpu", log_every=20, memory_bank=bank,
    )
    assert len(bank) > 0                                   # batches were enqueued
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in state.history)
    assert state.final_eval["regression_ok"] is True       # base never moved
