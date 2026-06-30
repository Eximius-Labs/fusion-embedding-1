"""Runnable P1 demo on tiny CPU stand-ins — the whole pipeline end to end.

    python -m fusion_embedding.demo_stage1

Trains the connector to align synthetic audio↔text pairs, prints the loss curve and
retrieval, and asserts the frozen base never moved. This is the same code path the
real pipeline runs once ``load_components`` wires the Qwen towers in.
"""

from __future__ import annotations

from .config import FusionConfig
from .train_stage1 import build_tiny_training_setup, train_stage1


def main() -> None:
    cfg = FusionConfig.tiny(max_steps=400, d_resampler=32)
    s = build_tiny_training_setup(cfg, n_train=8, batch_size=8, seed=0)

    print(f"trainable params : {sum(p.numel() for p in s.model.trainable_parameters()):,}")
    print(f"frozen params    : {sum(p.numel() for c in s.model.frozen_modules() for p in c.parameters()):,}")
    print(f"MRL ladder       : {cfg.mrl_dims}  (default {cfg.mrl_default})\n")

    pre = s.eval_fn(s.model)
    print(f"before training  : A->T R@1={pre['a2t_R@1']:.3f}  T->A R@1={pre['t2a_R@1']:.3f}")

    state = train_stage1(
        s.model, s.train_loader, s.loss_fn, cfg,
        steps=cfg.max_steps, eval_fn=s.eval_fn, device="cpu", log_every=50, guard_every=50,
    )

    print("\nstep    loss    infonce  acc_a2t  lr")
    for h in state.history:
        print(f"{h['step']:>4}  {h['loss']:6.3f}  {h['infonce']:6.3f}   {h['acc_a2t']:.3f}   {h['lr']:.2e}")

    post = state.final_eval
    print(f"\nafter training   : A->T R@1={post['a2t_R@1']:.3f}  T->A R@1={post['t2a_R@1']:.3f}")
    print(f"regression guard : base_drift={post['base_drift']:g}  ok={post['regression_ok']}")


if __name__ == "__main__":
    main()
