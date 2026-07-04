"""Floor audit (Step-3 saturation diagnosis) — predict_loss_floor + bank_neardup_stats.

The floor is the training loss under a PERFECT connector (audio == whitened text). These
tests pin the three properties the audit's conclusions rest on:

  1. plumbing exactness — with the whole corpus in the batch, the union exclude-mask must
     kill the entire bank, so the bank floor equals the in-batch-only floor exactly;
  2. semantic near-dups raise the floor, and FN-masking them removes exactly that cost;
  3. the near-dup census counts different-string neighbors only (self + exact-string dup
     groups excluded — training already masks those).
"""

import pytest
import torch

from fusion_embedding.config import FusionConfig
from fusion_embedding.losses import FusionContrastiveLoss
from fusion_embedding.train_stage1 import bank_neardup_stats, predict_loss_floor


def _tiny_loss():
    cfg = FusionConfig.tiny()
    return cfg, FusionContrastiveLoss(cfg), torch.tensor(cfg.logit_scale_init)


def _orthogonal_bank(m: int, d: int, scale: float = 3.0) -> torch.Tensor:
    """m exactly-orthogonal rows (scaled basis vectors) — cosine 0 between all pairs."""
    assert m <= d
    return torch.eye(m, d) * scale


def test_bank_fully_masked_equals_inbatch_floor():
    # batch == corpus and all captions unique -> the union mask covers every bank row,
    # so the bank must contribute NOTHING: bank floor == in-batch-only floor.
    cfg, loss_fn, ls = _tiny_loss()
    m = 8
    embs = torch.randn(m, cfg.d_llm)
    caps = [f"caption {i}" for i in range(m)]
    with_bank = predict_loss_floor(embs, caps, loss_fn, ls, batch_size=m, n_batches=2, use_bank=True)
    no_bank = predict_loss_floor(embs, caps, loss_fn, ls, batch_size=m, n_batches=2, use_bank=False)
    assert with_bank["floor_mean"] == pytest.approx(no_bank["floor_mean"], abs=1e-5)
    assert with_bank["mean_masked_bank_rows_per_anchor"] == pytest.approx(m)


def test_exact_dup_rows_are_union_masked():
    # Two rows share the SAME caption string (identical embedding). The union mask must
    # cover BOTH whenever either caption is in the batch. NOTE (real training semantics):
    # only BANK rows are masked — an exact-dup pair landing in the same batch is an
    # unmasked in-batch collision, so we assert the mask, not a clean loss value.
    cfg, loss_fn, ls = _tiny_loss()
    base = _orthogonal_bank(8, cfg.d_llm)
    embs = torch.cat([base, base[:1]])                      # row 8 duplicates row 0's embedding
    caps = [f"cap {i}" for i in range(8)] + ["cap 0"]       # ...and its caption string
    floor = predict_loss_floor(embs, caps, loss_fn, ls, batch_size=8, n_batches=4, use_bank=True, seed=1)
    # Without dup-group union masking, every 8-row batch would mask exactly its own 8 rows
    # (mean == 8.0). With it, whenever the excluded 9th row is a "cap 0" twin, the twin gets
    # masked too (9 rows) -> the mean must sit strictly above 8 and never exceed 9.
    assert 8.0 < floor["mean_masked_bank_rows_per_anchor"] <= 9.0


def test_semantic_neardup_raises_floor_and_fn_mask_removes_it():
    # Five twin pairs: every anchor has exactly one different-string near-dup (cos ~1).
    # With batch 4 of 10 the twin usually sits in the bank -> raw floor pays ~log 2 on the
    # rungs where it survives; FN-masking the bank at 0.98 removes exactly that cost.
    cfg, loss_fn, ls = _tiny_loss()
    m, d = 10, cfg.d_llm
    embs = torch.zeros(m, d)
    for p in range(5):
        embs[2 * p, p] = 3.0
        embs[2 * p + 1, p] = 3.0
        embs[2 * p + 1, p + 5] = 0.03                       # twin: same direction + tiny extra
    caps = [f"cap {i}" for i in range(m)]
    raw = predict_loss_floor(embs, caps, loss_fn, ls, batch_size=4, n_batches=8, use_bank=True,
                             fn_mask_threshold=None, fn_mask_dim=d)
    masked = predict_loss_floor(embs, caps, loss_fn, ls, batch_size=4, n_batches=8, use_bank=True,
                                fn_mask_threshold=0.98, fn_mask_dim=d)
    # Expected gap ~= 0.5 * log2 * P(twin in bank, not in batch) ~= 0.5 * 0.693 * 2/3 ~= 0.23
    # (the 0.5 is the symmetric a2t/t2a average: only the a2t direction sees the bank).
    assert raw["floor_mean"] > masked["floor_mean"] + 0.15  # the twins are real, removable cost
    assert masked["mean_masked_bank_rows_per_anchor"] > raw["mean_masked_bank_rows_per_anchor"]
    # masking more aggressively can only lower the floor further
    masked_low = predict_loss_floor(embs, caps, loss_fn, ls, batch_size=4, n_batches=8, use_bank=True,
                                    fn_mask_threshold=0.9, fn_mask_dim=d)
    assert masked_low["floor_mean"] <= masked["floor_mean"] + 1e-4


def test_floor_batch_larger_than_corpus_raises():
    cfg, loss_fn, ls = _tiny_loss()
    embs = torch.randn(4, cfg.d_llm)
    with pytest.raises(ValueError):
        predict_loss_floor(embs, [f"c{i}" for i in range(4)], loss_fn, ls, batch_size=8)


def test_bank_neardup_stats_counts_semantic_not_exact():
    cfg, _, _ = _tiny_loss()
    d = cfg.d_llm
    embs = _orthogonal_bank(5, d)
    embs[1] = embs[0] + 0.01 * embs[4]                      # semantic dup of row 0 (diff string)
    embs[3] = embs[2].clone()                               # exact dup of row 2 (same string)
    caps = ["dog barks", "a dog barking", "rain", "rain", "wind"]
    stats = bank_neardup_stats(embs, caps, dim=d, thresholds=(0.95,), chunk=2, top_examples=5)
    per = stats["per_threshold"]["0.95"]
    assert per["neighbor_pairs"] == 1                        # only the semantic pair counts
    assert per["frac_anchors_with_neighbor"] == pytest.approx(2 / 5)
    assert {stats["examples"][0]["a"], stats["examples"][0]["b"]} == {"dog barks", "a dog barking"}
    assert stats["examples"][0]["cos"] > 0.95
    # exact-dup rows ("rain") must report NO neighbor: their only twin is same-string-masked
    assert stats["max_offdiag_cos_p50"] < 0.5
