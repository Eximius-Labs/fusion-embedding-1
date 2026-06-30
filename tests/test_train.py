"""Stage F gate: optimizer/schedule/guard primitives + the connector-only loop mechanics."""

import torch

from fusion_embedding.config import FusionConfig
from fusion_embedding.train_stage1 import (
    RegressionGuard,
    build_optimizer,
    cosine_warmup,
    recall_at_k,
    retrieval_report,
)
from fusion_embedding._tiny import build_tiny_model
from .conftest import make_batch


def test_cosine_warmup_shape():
    warm, total = 10, 100
    assert cosine_warmup(0, warm, total) < cosine_warmup(5, warm, total)   # warming up
    assert abs(cosine_warmup(warm - 1, warm, total) - 1.0) < 1e-6          # peak at end of warmup
    assert cosine_warmup(total, warm, total) < 1e-6                        # decays to ~0


def test_optimizer_only_holds_trainable_params(tiny_model):
    opt = build_optimizer(tiny_model, tiny_model.cfg)
    opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert opt_param_ids == {id(p) for p in tiny_model.trainable_parameters()}
    # no frozen base param is in the optimizer
    frozen_ids = {id(p) for comp in tiny_model.frozen_modules() for p in comp.parameters()}
    assert opt_param_ids.isdisjoint(frozen_ids)


def test_regression_guard_detects_drift(tiny_model):
    guard = RegressionGuard(tiny_model)
    assert guard.max_drift(tiny_model) == 0.0
    # mutate a frozen base param
    with torch.no_grad():
        next(tiny_model.base_lm.parameters()).add_(1.0)
    assert guard.max_drift(tiny_model) > 0
    import pytest
    with pytest.raises(RuntimeError):
        guard.check(tiny_model)


def test_recall_at_k_perfect_and_worst():
    perfect = torch.eye(5) * 10
    r = recall_at_k(perfect, ks=(1,))
    assert r["R@1"] == 1.0
    # anti-diagonal of EVEN size is a derangement -> positive never top-1
    bad = torch.eye(4).flip(1) * 10
    assert recall_at_k(bad, ks=(1,))["R@1"] == 0.0


def test_retrieval_report_keys():
    a = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    rep = retrieval_report(a, a)            # identical -> perfect both directions
    assert rep["a2t_R@1"] == 1.0 and rep["t2a_R@1"] == 1.0


def test_single_train_step_updates_only_connector():
    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg)
    from fusion_embedding.losses import FusionContrastiveLoss
    loss_fn = FusionContrastiveLoss(cfg)
    opt = build_optimizer(model, cfg)

    before_resampler = [p.detach().clone() for p in model.resampler.parameters()]
    before_scale = model.logit_scale.detach().clone()

    batch = make_batch(cfg, batch_size=6, seed=3)
    out = model(batch)
    loss, _ = loss_fn(out["audio"], out["text"], out["logit_scale"])
    loss.backward()
    opt.step()

    # connector + temp moved
    assert any(not torch.equal(b, a) for b, a in zip(before_resampler, model.resampler.parameters()))
    assert not torch.equal(before_scale, model.logit_scale.detach())
    # frozen base received no grad
    for comp in model.frozen_modules():
        for p in comp.parameters():
            assert p.grad is None
