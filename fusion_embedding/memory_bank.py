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
        # cast to the buffer's dtype/device — the base may emit bf16 while the bank is fp32
        emb = emb.detach().to(device=self.device, dtype=self._buf.dtype)
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


class CorpusTextBank:
    """Full-corpus frozen-text negative bank built from the Step-2 text cache (zero staleness).

    Holds the WHITENED full-dim text embedding of every training caption (whitening is a fixed
    diagonal transform after fit, so whitening the corpus once is exact), plus a caption→rows
    lookup used to build the per-batch **exclusion mask**: a full-corpus bank necessarily
    contains each batch item's own caption (and any exact-duplicate captions elsewhere in the
    corpus) — those rows must not enter that anchor's InfoNCE denominator.
    """

    def __init__(self, embs_whitened: torch.Tensor, captions, device: str | torch.device = "cpu"):
        if embs_whitened.size(0) != len(captions):
            raise ValueError(f"bank rows {embs_whitened.size(0)} != captions {len(captions)}")
        self.embs = embs_whitened.to(device)                    # [M, d_llm], whitened, un-normalized
        self._rows_by_caption: dict = {}
        for i, c in enumerate(captions):
            self._rows_by_caption.setdefault(c, []).append(i)

    def __len__(self) -> int:
        return self.embs.size(0)

    @property
    def n_duplicate_captions(self) -> int:
        """Captions appearing more than once in the corpus (each occurrence gets masked)."""
        return sum(len(r) for r in self._rows_by_caption.values() if len(r) > 1)

    def exclude_mask(self, batch_captions, device=None) -> torch.Tensor:
        """[B, M] bool — True where the bank row matches ANY batch caption (exclude from denom).

        Union semantics (same rows masked for every anchor): the batch's captions are already
        represented as in-batch negatives/positives, so leaving them in the bank would (a) put
        each anchor's own positive in its denominator — the poison — and (b) double-count the
        other items' texts. The bank's job is strictly OUT-of-batch negatives.
        """
        rows: list = []
        for c in set(batch_captions):
            rows.extend(self._rows_by_caption.get(c, ()))
        mask = torch.zeros(len(batch_captions), self.embs.size(0), dtype=torch.bool,
                           device=device if device is not None else self.embs.device)
        if rows:
            mask[:, rows] = True
        return mask


@torch.no_grad()
def build_corpus_bank_from_cache(shard_paths, captions, whitening, *, exclude=None,
                                 device: str | torch.device = "cpu") -> CorpusTextBank:
    """Assemble a ``CorpusTextBank`` from sibling ``.txtemb.pt`` caches (Step 2).

    ``shard_paths``/``captions`` are the trainer's concatenated frame-shard order (bank row i
    ↔ global clip index i before exclusion). ``exclude`` drops the held-out eval clips so eval
    captions never appear as training negatives. ``whitening`` (fitted ``TextWhitening``) is
    applied once, in fp32, then stored fp16 — the loss casts per-forward.
    """
    from .data import text_emb_shard_path

    chunks = [torch.load(text_emb_shard_path(p), map_location="cpu", weights_only=False)["text_emb"]
              for p in shard_paths]
    raw = torch.cat(chunks)                                     # [N_total, d_llm] fp16 RAW
    if raw.size(0) != len(captions):
        raise ValueError(f"text cache rows {raw.size(0)} != captions {len(captions)}")
    keep = [i for i in range(raw.size(0)) if i not in (exclude or ())]
    kept_caps = [captions[i] for i in keep]
    dev = torch.device(device)
    whitened = whitening(raw[keep].to(dev).float()).half()      # fixed diagonal → exact, once
    return CorpusTextBank(whitened, kept_caps, device=dev)


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
