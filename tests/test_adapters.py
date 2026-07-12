"""Modality-gated deep adapters (docs/adapter_experiment_plan.md, Stage 0 gate).

The load-bearing claims under test:
1. text path BITWISE identical with adapters attached (the frozen-space guarantee);
2. audio path changes once adapter weights are nonzero;
3. gradients reach only resampler+adapters+logit_scale — never the base;
4. zero-init adapters are the exact identity (step 0 == current architecture);
5. checkpoint round-trips, including the hard-fail on adapter/арch mismatch;
6. the gate must be HELD OPEN through backward under gradient checkpointing
   (the recompute hazard) — and doing so yields gradients equal to no-checkpointing.
"""

import dataclasses

import pytest
import torch
import torch.nn as nn

from fusion_embedding.config import FusionConfig
from fusion_embedding.model import FusionEmbeddingModel
from fusion_embedding.train_stage1 import (build_optimizer, build_scheduler,
                                           init_trainables_from_ckpt, load_resume_ckpt,
                                           save_resume_ckpt, RegressionGuard)
from fusion_embedding._tiny import build_tiny_model

ADAPTER_RANK = 8


def _cfg(rank=ADAPTER_RANK):
    return dataclasses.replace(FusionConfig.tiny(), adapter_rank=rank)


def _pair(seed=0):
    """Two models with IDENTICAL shared weights: one without adapters, one with.

    Adapter modules are constructed after the shared components, so the RNG stream for
    base/resampler/whitening is the same in both builds.
    """
    plain = build_tiny_model(FusionConfig.tiny(), seed=seed)
    adapted = build_tiny_model(_cfg(), seed=seed)
    return plain, adapted


def _text_batch(cfg, B=3, S=7, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 60, (B, S), generator=g)
    ids[:, -1] = cfg.eos_id
    return ids, torch.ones(B, S, dtype=torch.long)


def _audio_pooled(model, cfg, B=3, T=11, seed=2):
    g = torch.Generator().manual_seed(seed)
    frames = torch.randn(B, T, cfg.d_audio, generator=g)
    tok = model.audio_tokens_from_frames(frames)
    ids = torch.tensor([[cfg.audio_pad_id] * cfg.n_query + [cfg.eos_id]] * B)
    return model.encode_audio(ids, torch.ones_like(ids), tok)


def _randomize_up(model, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for ad in model.audio_adapters:
            ad.up.weight.normal_(0, 0.05, generator=g)


def test_text_path_bitwise_identical_and_audio_identity_at_init():
    plain, adapted = _pair()
    cfg = plain.cfg
    ids, am = _text_batch(cfg)
    with torch.no_grad():
        assert torch.equal(plain.encode_text(ids, am), adapted.encode_text(ids, am))
        # zero-init adapters: even the GATED audio path is exactly the current arch
        assert torch.equal(_audio_pooled(plain, cfg), _audio_pooled(adapted, cfg))


def test_audio_changes_but_text_stays_identical_with_nonzero_adapters():
    plain, adapted = _pair()
    cfg = plain.cfg
    _randomize_up(adapted)
    ids, am = _text_batch(cfg)
    with torch.no_grad():
        assert torch.equal(plain.encode_text(ids, am), adapted.encode_text(ids, am)), \
            "text path must stay BITWISE identical regardless of adapter weights"
        pa, pb = _audio_pooled(plain, cfg), _audio_pooled(adapted, cfg)
        assert not torch.allclose(pa, pb), "nonzero adapters must change the audio path"


def test_gradient_isolation_and_guard():
    _, model = _pair()
    cfg = model.cfg
    _randomize_up(model)
    guard = RegressionGuard(model)
    ids, am = _text_batch(cfg)
    pooled_audio = _audio_pooled(model, cfg)
    with torch.no_grad():
        pooled_text = model.encode_text(ids, am)
    loss = -(torch.nn.functional.normalize(pooled_audio, dim=-1)
             * torch.nn.functional.normalize(pooled_text, dim=-1)).sum()
    loss.backward()
    for _, p in model.base_lm.named_parameters():
        assert p.grad is None, "base params must never receive grads"
    assert all(ad.up.weight.grad is not None and ad.up.weight.grad.abs().sum() > 0
               for ad in model.audio_adapters), "adapter grads must flow"
    opt = build_optimizer(model, cfg)
    opt.step()
    assert guard.max_drift(model) == 0.0, "adapters must not count as base drift"
    n_ad = sum(p.numel() for p in model.audio_adapters.parameters())
    assert n_ad > 0 and any(p2 is p for p2 in model.trainable_parameters()
                            for p in [model.audio_adapters[0].up.weight])


def test_resume_ckpt_roundtrip_and_presence_mismatch(tmp_path):
    _, model = _pair()
    cfg = model.cfg
    _randomize_up(model)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, max_steps=10)
    p = str(tmp_path / "resume.pt")
    save_resume_ckpt(p, model, opt, sched, step=4, total_steps=10, config_key="k")

    fresh = build_tiny_model(_cfg(), seed=0)                    # same frozen base…
    _randomize_up(fresh, seed=7)                                # …different trainables
    opt2 = build_optimizer(fresh, cfg)
    sched2 = build_scheduler(opt2, cfg, max_steps=10)
    assert load_resume_ckpt(p, fresh, opt2, sched2, total_steps=10, config_key="k") == 5
    with torch.no_grad():
        assert torch.equal(_audio_pooled(model, cfg), _audio_pooled(fresh, cfg))

    plain = build_tiny_model(FusionConfig.tiny(), seed=0)       # no adapters
    opt3 = build_optimizer(plain, plain.cfg)
    sched3 = build_scheduler(opt3, plain.cfg, max_steps=10)
    assert load_resume_ckpt(p, plain, opt3, sched3, total_steps=10, config_key="k") == 0, \
        "adapter ckpt into adapter-less model must refuse to resume"


def test_init_trainables_adapter_key_semantics():
    _, model = _pair()
    _randomize_up(model)
    ck = {"config": dataclasses.asdict(model.cfg), "resampler": model.resampler.state_dict(),
          "adapters": model.audio_adapters.state_dict(),
          "text_whitening": model.text_whitening.state_dict(), "logit_scale": 20.0}
    # adapters ckpt -> adapter-less model: hard fail (silent unadapted scoring hazard)
    plain = build_tiny_model(FusionConfig.tiny(), seed=1)
    with pytest.raises(ValueError, match="adapter_rank=0"):
        init_trainables_from_ckpt(plain, ck)
    # adapters ckpt -> adapter model (same frozen base, perturbed trainables): loads
    fresh = build_tiny_model(_cfg(), seed=0)
    _randomize_up(fresh, seed=7)
    info = init_trainables_from_ckpt(fresh, ck)
    assert "adapters" in info["loaded"]
    with torch.no_grad():
        assert torch.equal(_audio_pooled(model, model.cfg), _audio_pooled(fresh, fresh.cfg))
    # adapter-LESS ckpt -> adapter model: warm start, fresh identity adapters (Stage-3 arm)
    ck2 = {k: v for k, v in ck.items() if k != "adapters"}
    fresh2 = build_tiny_model(_cfg(), seed=0)
    info2 = init_trainables_from_ckpt(fresh2, ck2)
    assert "adapters(fresh-identity)" in info2["loaded"]


class _CheckpointedLM(nn.Module):
    """Wrap a base so its forward runs under torch.utils.checkpoint (GC emulation)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, inputs_embeds, attention_mask):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(lambda e, m: self.inner(inputs_embeds=e, attention_mask=m),
                          inputs_embeds, attention_mask, use_reentrant=False)


def _gc_grads(hold_scope: bool, seed=0):
    model = build_tiny_model(_cfg(), seed=seed)
    _randomize_up(model)
    model.base_lm = _CheckpointedLM(model.base_lm)              # hooks stay on inner layers
    # gate discovery happened at __init__ against the raw base; hooks are already attached
    scope = model.adapter_scope() if hold_scope else None
    if scope is not None:
        scope.__enter__()
    try:
        pooled = _audio_pooled(model, model.cfg)
        pooled.sum().backward()
    finally:
        if scope is not None:
            scope.__exit__(None, None, None)
    return model


def test_gate_must_span_backward_under_gradient_checkpointing():
    # Reference: no checkpointing, encode_audio's own scope is enough.
    ref = build_tiny_model(_cfg(), seed=0)
    _randomize_up(ref)
    _audio_pooled(ref, ref.cfg).sum().backward()
    ref_grad = ref.audio_adapters[0].up.weight.grad.clone()
    assert ref_grad.abs().sum() > 0

    held = _gc_grads(hold_scope=True, seed=0)
    assert torch.allclose(held.audio_adapters[0].up.weight.grad, ref_grad, atol=1e-6), \
        "with the scope held through backward, GC grads must match no-GC grads"

    # Without the outer scope, the recompute runs gate-closed: adapters drop out of the
    # recomputed graph. use_reentrant=False DETECTS the saved-tensor-count mismatch and
    # raises — the hazard is loud, not silent (still: always hold the scope).
    from torch.utils.checkpoint import CheckpointError
    with pytest.raises(CheckpointError):
        _gc_grads(hold_scope=False, seed=0)
