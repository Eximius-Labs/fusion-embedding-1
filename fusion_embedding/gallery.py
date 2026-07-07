"""Pure, GPU-free helpers for building the audio->image retrieval gallery.

Kept out of ``modal_app`` so the ranking/packaging logic is unit-testable on CPU
(the embedding itself is exercised by ``phase0_crossmodal``, already validated).
"""
from __future__ import annotations

from typing import List, Optional

import torch


def select_gallery(sim: torch.Tensor, k: int,
                   true_idx: Optional[torch.Tensor] = None,
                   n_queries: Optional[int] = None) -> List[dict]:
    """Rank images per audio query and report where the true pair landed.

    Args:
        sim: [Nq, Nimg] similarity (higher = more similar).
        k: how many top images to keep per query.
        true_idx: [Nq] ground-truth image index per query. Defaults to the
            aligned diagonal (query i <-> image i), which is how the VGGSound
            AV pairs are stored.
        n_queries: keep only the first N queries (dataset order, deterministic).

    Returns one dict per kept query:
        query, true_idx, topk (list of image indices, best first),
        true_rank (1-based rank of the true image over ALL images),
        hit_at_k (was the true image in the top-k).
    """
    if sim.dim() != 2:
        raise ValueError(f"sim must be 2-D [Nq, Nimg], got {tuple(sim.shape)}")
    nq, nimg = sim.shape
    if k < 1 or k > nimg:
        raise ValueError(f"k={k} out of range for {nimg} images")
    if true_idx is None:
        true_idx = torch.arange(nq)
    if true_idx.shape[0] != nq:
        raise ValueError("true_idx length must match number of queries")
    limit = nq if n_queries is None else min(n_queries, nq)

    topk = sim.topk(k, dim=1).indices  # [Nq, k]
    rows: List[dict] = []
    for i in range(limit):
        t = int(true_idx[i])
        # 1-based rank: how many images score strictly higher than the true one, +1.
        rank = int((sim[i] > sim[i, t]).sum().item()) + 1
        idxs = [int(x) for x in topk[i].tolist()]
        rows.append({
            "query": i,
            "true_idx": t,
            "topk": idxs,
            "true_rank": rank,
            "hit_at_k": t in idxs,
        })
    return rows
