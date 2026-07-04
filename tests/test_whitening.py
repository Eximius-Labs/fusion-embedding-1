"""Text-whitening (anisotropy fix): unit coverage of TextWhitening + an integration test that
the tiny P1 pipeline still trains (loss drops, base stays frozen) with whitening fitted."""

import torch

from fusion_embedding.config import FusionConfig
from fusion_embedding.model import TextWhitening
from fusion_embedding._tiny import build_tiny_model
from fusion_embedding.train_stage1 import (
    RegressionGuard,
    build_tiny_training_setup,
    fit_text_whitening,
    train_stage1,
)
from .conftest import make_batch


# ------------------------------- unit: the transform ------------------------------- #
def test_identity_until_fit():
    w = TextWhitening(8)
    x = torch.randn(5, 8)
    assert torch.equal(w(x), x)
    assert int(w.fitted) == 0


def test_fit_standardizes_per_dim():
    w = TextWhitening(8)
    embs = torch.randn(500, 8) * torch.arange(1, 9).float() + 3.0   # per-dim scale + offset
    w.fit(embs)
    out = w(embs)
    assert torch.allclose(out.mean(0), torch.zeros(8), atol=1e-4)
    assert torch.allclose(out.std(0), torch.ones(8), atol=1e-2)
    assert int(w.fitted) == 1


def test_mrl_safe_diagonal():
    """Truncate-then-whiten(first-d stats) == whiten-then-truncate — the MRL nesting invariant."""
    full = TextWhitening(8)
    full.fit(torch.randn(200, 8) * 2 + 1)
    x = torch.randn(4, 8)
    d = 4
    whiten_then_trunc = full(x)[:, :d]
    sub = TextWhitening(d)
    sub.mean.copy_(full.mean[:d]); sub.std.copy_(full.std[:d]); sub.fitted.fill_(1)
    trunc_then_whiten = sub(x[:, :d])
    assert torch.allclose(whiten_then_trunc, trunc_then_whiten, atol=1e-6)


def test_whitening_has_no_parameters():
    w = TextWhitening(16)
    assert list(w.parameters()) == []        # buffers only -> never trained, never in optimizer


def test_whitening_follows_input_dtype():
    # buffers are fp32; a fp16 input must not raise (transform follows x's dtype/device).
    # Regression for the frozen-frames path where buffers and text live on different devices.
    w = TextWhitening(8)
    w.fit(torch.randn(100, 8))
    x = torch.randn(3, 8, dtype=torch.float16)
    out = w(x)
    assert out.dtype == torch.float16 and torch.isfinite(out).all()


# ------------------------------- model wiring ------------------------------- #
def test_model_forward_applies_whitening_to_text_only():
    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg, seed=0)
    batch = make_batch(cfg, batch_size=6, seed=2)

    before = model(batch)
    text_before, audio_before = before["text"].clone(), before["audio"].clone()

    raw = model.encode_text(batch["text_input_ids"], batch["text_attention_mask"])
    model.text_whitening.fit(raw)
    after = model(batch)

    assert int(model.text_whitening.fitted) == 1
    assert not torch.allclose(text_before, after["text"])      # text branch changed
    assert torch.allclose(audio_before, after["audio"])        # audio branch untouched
    # whitening stays out of the optimizer set
    tp = {id(p) for p in model.trainable_parameters()}
    assert all(id(p) not in tp for p in model.text_whitening.buffers())


# ------------------------------- integration ------------------------------- #
def test_pipeline_trains_with_whitening_fitted():
    cfg = FusionConfig.tiny(max_steps=400, d_resampler=32)
    s = build_tiny_training_setup(cfg, n_train=8, batch_size=8, seed=0)

    stats = fit_text_whitening(s.model, s.manifest, s.collator, device="cpu", max_samples=64)
    assert int(s.model.text_whitening.fitted) == 1
    assert stats["n_samples"] > 0
    assert -1.0 <= stats["raw_mean_pairwise_cos"] <= 1.0

    guard = RegressionGuard(s.model)
    state = train_stage1(s.model, s.train_loader, s.loss_fn, cfg, steps=400, device="cpu",
                         eval_fn=s.eval_fn, log_every=10**9)

    assert state.history[-1]["loss"] < state.history[0]["loss"]     # still learns
    assert state.final_eval["a2t_R@1"] > 1.0 / 8                    # above random floor
    assert guard.max_drift(s.model) == 0.0                         # base still byte-frozen
