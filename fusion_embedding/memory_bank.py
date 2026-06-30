"""Frozen-text memory bank — the lever that makes small-VRAM contrastive training work.

InfoNCE quality scales with the number of negatives *in the same forward batch*, and
grad-accumulation does NOT add negatives. On an 8GB card the micro-batch is 1-2, which
would normally starve the contrastive signal. But the text tower here is **frozen**, so
its embeddings never drift — a bank of past/precomputed text embeddings is a set of
exact, zero-staleness negatives. Feeding that bank to the A→T denominator gives a
micro-batch of 1 thousands of real negatives.

Unlike MoCo (whose key encoder drifts, forcing a momentum encoder), this bank needs no
momentum and no staleness correction: Fusion's text path is byte-frozen.

Two usage modes:
  * ``TextMemoryBank`` — a live FIFO queue fed from past batches (MoCo-style), for when
    you stream data and want recent texts as negatives;
  * ``precompute_text_bank`` — embed a whole shard once up front (the text tower is
    frozen, so this is a one-time cost) and reuse it as a fixed negative bank.
"""

from __future__ import annotations

from typing import Optional

import torch

from .model import FusionEmbeddingModel, mrl_truncate_normalize


class TextMemoryBank:
    """FIFO queue of frozen full-dim text embeddings used as shared InfoNCE negatives.

    Discipline (avoids false negatives): on each step compute the loss against the bank
    BEFORE enqueuing the current batch, so a positive text is never its own anchor's
    negative within a step. Entries are stored detached and un-normalized (the loss
    truncates + renormalizes per MRL rung).
    """

    def __init__(self, dim: int, capacity: int, device: str | torch.device = "cpu"):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.dim = dim
        self.capacity = capacity
        self.device = torch.device(device)
        self._buf = torch.zeros(capacity, dim, device=self.device)
        self._ptr = 0
        self._count = 0   # number of valid entries (<= capacity)

    def __len__(self) -> int:
        return self._count

    @property
    def is_empty(self) -> bool:
        return self._count == 0

    @torch.no_grad()
    def enqueue(self, emb: torch.Tensor) -> None:
        """Add a [b, dim] batch of text embeddings, overwriting oldest on wraparound."""
        if emb.dim() != 2 or emb.size(1) != self.dim:
            raise ValueError(f"expected [b, {self.dim}], got {tuple(emb.shape)}")
        emb = emb.detach().to(self.device)
        b = emb.size(0)
        if b >= self.capacity:                         # batch bigger than bank: keep the last `capacity`
            self._buf.copy_(emb[-self.capacity:])
            self._ptr = 0
            self._count = self.capacity
            return
        idx = (torch.arange(b, device=self.device) + self._ptr) % self.capacity
        self._buf[idx] = emb
        self._ptr = int((self._ptr + b) % self.capacity)
        self._count = min(self.capacity, self._count + b)

    def get(self) -> Optional[torch.Tensor]:
        """Return the [count, dim] valid negatives, or None if empty."""
        if self._count == 0:
            return None
        return self._buf[: self._count]


@torch.no_grad()
def precompute_text_bank(
    model: FusionEmbeddingModel,
    manifest,
    collator,
    *,
    batch_size: int = 16,
    device: str = "cpu",
    normalize_dim: Optional[int] = None,
) -> torch.Tensor:
    """Embed every item's text once (frozen tower → reusable) → [M, d_llm] negative bank.

    Returns full-dim pooled text (un-normalized) by default so the loss can truncate per
    rung; pass ``normalize_dim`` to get an L2-normalized [M, dim] bank instead.
    """
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(manifest, batch_size=batch_size, collate_fn=collator, shuffle=False)
    chunks = []
    for batch in loader:
        ids = batch["text_input_ids"].to(device)
        mask = batch["text_attention_mask"].to(device)
        pooled = model.encode_text(ids, mask)
        if normalize_dim is not None:
            pooled = mrl_truncate_normalize(pooled, normalize_dim)
        chunks.append(pooled.cpu())
    return torch.cat(chunks)
