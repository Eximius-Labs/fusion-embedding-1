"""Stage G — the capstone end-to-end integration test.

Drives the entire P1 pipeline on tiny CPU stand-ins exactly as production will run it:
synthetic manifest -> DataLoader -> FusionCollator -> FusionEmbeddingModel (frozen
base + trained connector) -> FusionContrastiveLoss (InfoNCE/MRL + CORAL) -> AdamW
step -> retrieval eval. The exit-gate assertions mirror HLD §7's P1 gate:

    * audio lands in the space: A→T / T→A R@1 climbs from the random floor toward 1;
    * training actually reduces the loss;
    * MMEB-V2 stays unchanged: the frozen base does not move by a single bit.

Random synthetic data has no generalizable cross-modal structure, so the meaningful
signal is memorization of the trained pairs (see build_tiny_training_setup) — the
proof that inject -> pool -> loss -> optimizer is correctly wired end to end.
"""

import torch

from fusion_embedding.config import FusionConfig
from fusion_embedding.train_stage1 import (
    RegressionGuard,
    build_tiny_training_setup,
    encode_dataset,
    retrieval_report,
    train_stage1,
)


def test_p1_exit_gate_end_to_end():
    # d_resampler=32 gives the connector enough capacity to memorize the 8 random pairs
    # (the bottleneck width 16 is exercised by the resampler/model/data tests).
    cfg = FusionConfig.tiny(max_steps=400, d_resampler=32)
    s = build_tiny_training_setup(cfg, n_train=8, batch_size=8, seed=0)

    pre = s.eval_fn(s.model)                                # audio-blind floor
    guard = RegressionGuard(s.model)

    state = train_stage1(
        s.model, s.train_loader, s.loss_fn, cfg,
        steps=400, eval_fn=s.eval_fn, device="cpu", log_every=25, guard_every=25,
    )

    first_loss = state.history[0]["loss"]
    last_loss = state.history[-1]["loss"]
    post = state.final_eval

    # 1) training reduced the loss
    assert last_loss < first_loss, f"loss did not drop: {first_loss:.3f} -> {last_loss:.3f}"

    # 2) audio landed in the space — retrieval climbs far above the random floor (1/8)
    random_floor = 1.0 / 8
    assert post["a2t_R@1"] > pre["a2t_R@1"]
    assert post["a2t_R@1"] >= 0.875 > random_floor, post
    assert post["t2a_R@1"] >= 0.875, post

    # 3) regression guard: the frozen base did not move at all
    assert post["regression_ok"] is True
    assert post["base_drift"] == 0.0
    assert guard.max_drift(s.model) == 0.0


def test_mrl_rungs_all_retrieve_above_floor():
    """After P1, every MRL rung (not just the default) gives strongly-above-floor retrieval."""
    cfg = FusionConfig.tiny(max_steps=400, d_resampler=32)
    s = build_tiny_training_setup(cfg, n_train=8, batch_size=8, seed=0)
    train_stage1(s.model, s.train_loader, s.loss_fn, cfg, steps=400, device="cpu", log_every=10**9)

    floor = 1.0 / 8
    for dim in cfg.mrl_dims:
        a, t = encode_dataset(s.model, s.manifest, s.collator, dim=dim)
        rep = retrieval_report(a, t)
        assert rep["a2t_R@1"] > floor, f"rung {dim} at/below random floor: {rep['a2t_R@1']}"


def test_temperature_stays_in_clamp():
    cfg = FusionConfig.tiny(max_steps=100)
    s = build_tiny_training_setup(cfg, seed=1)
    train_stage1(s.model, s.train_loader, s.loss_fn, cfg, steps=100, device="cpu", log_every=100)
    assert s.model.clamped_logit_scale().item() <= cfg.logit_scale_max + 1e-6
    assert torch.isfinite(s.model.logit_scale)
