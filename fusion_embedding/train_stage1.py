"""Stage 1 (P1) — connector-only training loop + eval (HLD §5.2, §7, §10).

Trains *only* the FusionResampler + learnable temperature on audio↔text pairs:
symmetric InfoNCE over the MRL ladder + light CORAL. The base stays byte-frozen,
enforced every run by ``RegressionGuard`` (the param-level form of the HLD's
"MMEB-V2 must stay unchanged" guard; the real MMEB eval is a seam in ``evaluate``).

Runs unchanged on tiny CPU stand-ins (``build_tiny_training_setup``) or on the real
Qwen towers wired in ``load_components``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .config import FusionConfig
from .data import FusionAudioTextManifest, FusionCollator, make_synthetic_dataset
from .losses import FusionContrastiveLoss
from .model import FusionEmbeddingModel, mrl_truncate_normalize


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------- #
# Optimizer + schedule (HLD §5.3)
# ---------------------------------------------------------------------------- #
def build_optimizer(model: FusionEmbeddingModel, cfg: FusionConfig) -> torch.optim.Optimizer:
    params = list(model.trainable_parameters())
    return torch.optim.AdamW(
        params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay
    )


def cosine_warmup(step: int, warmup_steps: int, max_steps: int) -> float:
    """lr multiplier: linear warmup then cosine decay to 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if max_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(optimizer, cfg: FusionConfig, max_steps: int):
    warmup = int(round(cfg.warmup_ratio * max_steps))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: cosine_warmup(s, warmup, max_steps)
    )


# ---------------------------------------------------------------------------- #
# Preemption-resilient checkpoint / resume (Modal preempts long GPU jobs; the loop
# only saved at the very end, so a preemption at step 850 wiped ~4.5h — 2026-07-02).
# We snapshot ONLY the trainable state (resampler + logit_scale) plus optimizer &
# scheduler; the frozen base, whitening, and text-bank are re-derived deterministically
# from the cache on restart, so they don't need saving.
# ---------------------------------------------------------------------------- #
def save_resume_ckpt(path: str, model: FusionEmbeddingModel, opt, sched, step: int, total_steps: int,
                     config_key: str = ""):
    """Atomically write a resume checkpoint (write-tmp-then-rename, so a preemption mid-write
    can't corrupt the file). ``step`` is the last COMPLETED step. ``config_key`` fingerprints the
    run config (arch + batch + lr + data) so a resume can never silently cross A/B arms."""
    import os
    tmp = path + ".tmp"
    state = {
        "step": int(step), "total_steps": int(total_steps), "config_key": str(config_key),
        "resampler": model.resampler.state_dict(),
        "logit_scale": model.logit_scale.detach().cpu(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
    }
    if model.audio_adapters is not None:
        state["adapters"] = model.audio_adapters.state_dict()
    torch.save(state, tmp)
    os.replace(tmp, path)                                   # atomic on POSIX


def load_resume_ckpt(path: str, model: FusionEmbeddingModel, opt, sched, *, total_steps: int,
                     config_key: str = "") -> int:
    """If a valid resume checkpoint for THIS run (same ``total_steps`` AND same ``config_key``)
    exists, restore the trainable state + optimizer/scheduler into the given objects and return
    the step to resume FROM (last completed + 1). Returns 0 (start fresh) if the file is absent
    or from a different run config — resuming across configs (e.g. a different d_resampler arm
    or lr) would be silent corruption at best, a shape error at worst.
    Optimizer state tensors are moved onto the model's device (they save on CPU)."""
    import os
    if not os.path.exists(path):
        return 0
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if int(ck.get("total_steps", -1)) != int(total_steps):
        return 0                                            # config changed — don't resume
    if str(ck.get("config_key", "")) != str(config_key):
        print(f"RESUME REFUSED: config_key mismatch (ckpt {ck.get('config_key', '')!r} != "
              f"run {config_key!r}) — starting fresh")
        return 0
    if ("adapters" in ck) != (model.audio_adapters is not None):
        print("RESUME REFUSED: adapter presence mismatch between checkpoint and model "
              "— starting fresh")
        return 0
    device = next(model.resampler.parameters()).device
    model.resampler.load_state_dict(ck["resampler"])
    if model.audio_adapters is not None:
        model.audio_adapters.load_state_dict(ck["adapters"])
    with torch.no_grad():
        model.logit_scale.copy_(ck["logit_scale"].to(model.logit_scale.device))
    opt.load_state_dict(ck["optimizer"])
    for st in opt.state.values():                           # AdamW moment buffers -> param device
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)
    sched.load_state_dict(ck["scheduler"])
    return int(ck["step"]) + 1


def qb_norm(sim_qg: torch.Tensor, bank_sim_bg: torch.Tensor, beta: float = 20.0,
            mode: str = "dis") -> torch.Tensor:
    """Querybank Normalisation (Bogolin et al., CVPR 2022) — test-time hubness correction.

    ``sim_qg`` [Q, G]: test query→gallery similarities. ``bank_sim_bg`` [B, G]: querybank
    (TRAINING-set queries, never test) → gallery similarities. Inverted softmax divides each
    gallery item's score by how strongly the bank as a whole is attracted to it, deflating
    hub items. ``mode='dis'`` (Dynamic Inverted Softmax, the paper's robust variant) applies
    the correction ONLY to test queries whose raw top-1 lands on a bank-activated item
    (an item that is top-1 for ≥1 bank query); other queries keep raw scores. Returns
    adjusted similarities — ranks derived from them, values not comparable to raw."""
    if sim_qg.dim() != 2 or bank_sim_bg.dim() != 2 or sim_qg.shape[1] != bank_sim_bg.shape[1]:
        raise ValueError(f"shape mismatch: sim {tuple(sim_qg.shape)} bank {tuple(bank_sim_bg.shape)}")
    denom = torch.exp(beta * bank_sim_bg).sum(dim=0)            # [G] bank attraction per item
    adjusted = torch.exp(beta * sim_qg) / denom.unsqueeze(0)    # inverted softmax
    if mode == "is":
        return adjusted
    if mode != "dis":
        raise ValueError(f"unknown mode {mode!r} (use 'is' or 'dis')")
    activated = torch.zeros(sim_qg.shape[1], dtype=torch.bool)
    activated[bank_sim_bg.argmax(dim=1)] = True                 # bank-activated gallery items
    hub_hit = activated[sim_qg.argmax(dim=1)]                   # [Q] raw top-1 is a hub?
    out = sim_qg.clone()
    out[hub_hit] = adjusted[hub_hit]
    return out


def init_trainables_from_ckpt(model: FusionEmbeddingModel, ck: dict) -> dict:
    """Warm-start the TRAINABLE state from a FINISHED training checkpoint (the dict written by
    the trainer: ``resampler`` + ``text_whitening`` + ``logit_scale`` + ``config``) — the
    second-stage fine-tune entry point.

    Unlike ``load_resume_ckpt`` this deliberately does NOT restore optimizer/scheduler state:
    a fine-tune is a NEW optimization (fresh schedule, usually a lower lr), not a continuation
    of the same run. The pretrain's whitening buffers ARE restored — the audio side was trained
    against that exact transform, so refitting whitening on the FT corpus would shift the
    target space out from under the connector. Raises on architecture mismatch."""
    cfg_ck = ck.get("config", {}) or {}
    for key in ("d_resampler", "n_query"):
        want = getattr(model.cfg, key, None)
        have = cfg_ck.get(key)
        if have is not None and want is not None and int(have) != int(want):
            raise ValueError(f"init_from_ckpt architecture mismatch: ckpt {key}={have} "
                             f"but model has {key}={want}")
    if "adapters" in ck and model.audio_adapters is None:
        # Loading an adapter checkpoint into an adapter-less model would silently score
        # /fine-tune the UNADAPTED model — hard-fail instead (adapter plan §4).
        raise ValueError("checkpoint carries audio adapters but the model was built with "
                         "adapter_rank=0 — rebuild with the checkpoint's adapter_rank")
    model.resampler.load_state_dict(ck["resampler"])
    loaded = ["resampler"]
    if model.audio_adapters is not None:
        if "adapters" in ck:
            model.audio_adapters.load_state_dict(ck["adapters"])
            loaded.append("adapters")
        else:
            # Warm-starting a pretrained resampler under FRESH (zero-init = identity)
            # adapters is the intended Stage-3 arm — keep them as built.
            loaded.append("adapters(fresh-identity)")
    if "text_whitening" in ck:
        model.text_whitening.load_state_dict(ck["text_whitening"])
        loaded.append("text_whitening")
    if ck.get("logit_scale") is not None:
        with torch.no_grad():
            model.logit_scale.copy_(torch.as_tensor(ck["logit_scale"]).float()
                                    .reshape(()).to(model.logit_scale.device))
        loaded.append("logit_scale")
    return {"loaded": loaded, "ckpt_d_resampler": cfg_ck.get("d_resampler"),
            "ckpt_base_4bit": ck.get("base_4bit")}


# ---------------------------------------------------------------------------- #
# Regression guard — the base must not move (HLD §7 invariant)
# ---------------------------------------------------------------------------- #
class RegressionGuard:
    """Snapshot every frozen base parameter; assert it never changes during training.

    This is the cheap, exact precondition behind the MMEB-V2 guard: if a base param
    moved, a frozen weight leaked into the optimizer and MMEB would regress.
    """

    def __init__(self, model: FusionEmbeddingModel):
        self._snapshot = {pid: p.detach().clone() for pid, p in self._frozen_by_id(model)}

    @staticmethod
    def _frozen_by_id(model: FusionEmbeddingModel):
        seen = set()
        for comp in model.frozen_modules():
            if not hasattr(comp, "named_parameters"):
                continue
            for _, p in comp.named_parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield id(p), p

    def max_drift(self, model: FusionEmbeddingModel) -> float:
        drift = 0.0
        for pid, p in self._frozen_by_id(model):
            # snapshot may live on a different device than the (possibly moved) model
            snap = self._snapshot[pid].to(p.device)
            drift = max(drift, (p.detach() - snap).abs().max().item())
        return drift

    def check(self, model: FusionEmbeddingModel) -> None:
        d = self.max_drift(model)
        if d != 0.0:
            raise RuntimeError(f"REGRESSION GUARD FAILED: frozen base drifted by {d:g} (base leaked into training).")


# ---------------------------------------------------------------------------- #
# Eval — retrieval R@k (HLD §9) + the regression guard
# ---------------------------------------------------------------------------- #
@torch.no_grad()
def encode_dataset(
    model: FusionEmbeddingModel,
    manifest: FusionAudioTextManifest,
    collator: FusionCollator,
    *,
    dim: Optional[int] = None,
    batch_size: int = 16,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed every pair -> (audio_emb [M,dim], text_emb [M,dim]) at an MRL rung, L2-normed."""
    model.eval()
    device = device if device is not None else next(model.parameters()).device
    dim = dim or model.cfg.mrl_default
    loader = DataLoader(manifest, batch_size=batch_size, collate_fn=collator, shuffle=False)
    a_chunks, t_chunks = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)
        a_chunks.append(mrl_truncate_normalize(out["audio"], dim).cpu())
        t_chunks.append(mrl_truncate_normalize(out["text"], dim).cpu())
    return torch.cat(a_chunks), torch.cat(t_chunks)


@torch.no_grad()
def fit_text_whitening(
    model: FusionEmbeddingModel,
    manifest,
    collator,
    *,
    device=None,
    max_samples: int = 4096,
    batch_size: int = 32,
) -> dict:
    """Estimate the per-dim text-whitening stats on a sample of captions, then install them.

    Runs the frozen base over up to ``max_samples`` captions (audio/frames untouched — ``encode_text``
    ignores them) and fits ``model.text_whitening``. MUST run BEFORE training so the connector learns
    to align to *whitened* text targets; also saved in the checkpoint so eval reproduces the geometry.
    """
    device = device if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    loader = DataLoader(manifest, batch_size=batch_size, collate_fn=collator, shuffle=False)
    chunks, n = [], 0
    for batch in loader:
        batch = _to_device(batch, device)
        raw = model.encode_text(batch["text_input_ids"], batch["text_attention_mask"])  # un-whitened
        chunks.append(raw.float().cpu())
        n += raw.size(0)
        if n >= max_samples:
            break
    embs = torch.cat(chunks)[:max_samples]
    model.text_whitening.fit(embs.to(device))
    if was_training:
        model.train()
    # anisotropy readout: mean pairwise cosine of RAW (pre-whitening) normalised text embeddings
    raw_norm = torch.nn.functional.normalize(embs[: min(512, embs.size(0))], dim=-1)
    off = raw_norm @ raw_norm.t()
    m = off.size(0)
    mean_cos = (off.sum() - m) / (m * (m - 1))
    return {"n_samples": int(embs.size(0)), "raw_mean_pairwise_cos": float(mean_cos),
            "std_dim_mean": float(embs.std(0).mean())}


@torch.no_grad()
def fit_text_whitening_from_cache(model, loader, *, device=None, max_samples: int = 4096) -> dict:
    """Fit text whitening from the PRECOMPUTED RAW text cache (Step 2) — no frozen-base forward at all.

    Reads ``batch["text_emb_cached"]`` (raw pooled text) from ``loader`` until ``max_samples``, then
    installs the per-dim stats. Same result as ``fit_text_whitening`` but skips re-encoding captions.
    """
    device = device if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    chunks, n = [], 0
    for batch in loader:
        emb = batch.get("text_emb_cached")
        if emb is None:
            raise RuntimeError("fit_text_whitening_from_cache: batch has no 'text_emb_cached'")
        chunks.append(emb.float().cpu())
        n += emb.size(0)
        if n >= max_samples:
            break
    embs = torch.cat(chunks)[:max_samples]
    model.text_whitening.fit(embs.to(device))
    if was_training:
        model.train()
    raw_norm = torch.nn.functional.normalize(embs[: min(512, embs.size(0))], dim=-1)
    off = raw_norm @ raw_norm.t()
    m = off.size(0)
    mean_cos = (off.sum() - m) / (m * (m - 1))
    return {"n_samples": int(embs.size(0)), "raw_mean_pairwise_cos": float(mean_cos),
            "std_dim_mean": float(embs.std(0).mean()), "from_cache": True}


def recall_at_k(sims: torch.Tensor, ks=(1, 10)) -> dict:
    """Recall@k for diagonal ground-truth (row i's positive is column i)."""
    M = sims.size(0)
    targets = torch.arange(M)
    ranks = (sims.argsort(dim=1, descending=True) == targets.view(-1, 1)).float().argmax(dim=1)
    return {f"R@{k}": (ranks < k).float().mean().item() for k in ks}


def _hit_recall(sims: torch.Tensor, relevance: torch.Tensor, ks) -> dict:
    """R@k = fraction of queries with ≥1 *relevant* item in the top-k (multi-relevant aware).

    Reduces to the diagonal ``recall_at_k`` when ``relevance`` is the identity.
    """
    N = sims.size(1)
    maxk = min(max(ks), N)
    order = sims.argsort(dim=1, descending=True)[:, :maxk]        # [Q, maxk] item indices
    rel_sorted = torch.gather(relevance.to(sims.device), 1, order).bool()
    return {f"R@{k}": rel_sorted[:, :k].any(dim=1).float().mean().item() for k in ks}


def average_precision_at_k(sims: torch.Tensor, relevance: torch.Tensor, k: int = 10) -> float:
    """mAP@k over queries. For single-relevant (diagonal) this is MRR@k = mean(1/rank | rank≤k).

    ``relevance[q, i]`` marks the ground-truth-relevant items for query q — so semantically
    equivalent captions can *all* count as correct (the ECCV-Caption / DCASE multi-relevant fix).
    """
    Q, N = sims.shape
    kk = min(k, N)
    order = sims.argsort(dim=1, descending=True)[:, :kk]          # top-kk indices
    rel_sorted = torch.gather(relevance.to(sims.device), 1, order).float()   # [Q, kk]
    cum_hits = torch.cumsum(rel_sorted, dim=1)                    # relevant seen up to position i
    ranks = torch.arange(1, kk + 1, device=sims.device).float()
    precision_at_i = cum_hits / ranks                            # precision@i along the ranking
    ap_num = (precision_at_i * rel_sorted).sum(dim=1)            # sum precision at each hit
    total_rel = relevance.to(sims.device).float().sum(dim=1).clamp(min=1.0)
    denom = torch.clamp(total_rel, max=float(kk))               # normalise by min(R, k)
    return (ap_num / denom).mean().item()


def semantic_relevance(text_emb: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    """Relevance mask treating near-duplicate captions as mutually relevant.

    Two clips whose (L2-normed) caption embeddings have cosine ≥ ``threshold`` are counted
    as valid matches for each other — the "unlabelled-but-correct caption" correction that
    de-inflates R@1 when the eval set has semantically overlapping captions. Diagonal always on.
    """
    rel = (text_emb @ text_emb.transpose(0, 1)) >= threshold
    idx = torch.arange(rel.size(0), device=rel.device)
    rel[idx, idx] = True
    return rel


def lexical_relevance(captions, threshold: float = 0.5) -> torch.Tensor:
    """Relevance mask from caption *word-overlap* (Jaccard ≥ threshold) — a text-only, anisotropy-
    free way to credit near-duplicate captions. Use this instead of ``semantic_relevance`` when the
    frozen text space is too anisotropic for cosine thresholds to be meaningful. O(N²), N≈few-hundred.
    """
    toks = [set(str(c).lower().split()) for c in captions]
    N = len(toks)
    rel = torch.eye(N, dtype=torch.bool)
    for i in range(N):
        for j in range(i + 1, N):
            a, b = toks[i], toks[j]
            if a and b and len(a & b) / len(a | b) >= threshold:
                rel[i, j] = rel[j, i] = True
    return rel


def multicaption_relevance(caption_group_ids: Sequence[int], n_audio: Optional[int] = None) -> torch.Tensor:
    """Relevance matrix for the standard AudioCaps/Clotho protocol: one audio clip has SEVERAL
    reference captions (5 for the test split). ``caption_group_ids[j]`` = the audio index caption j
    belongs to. Returns ``[N_audio, N_caps]`` bool where ``rel[i, j]`` marks j as a valid ref for i.

    Feeding this to ``retrieval_report`` gives the published A→T "min-rank over the 5 refs" scoring
    (a clip's audio is correct if ANY of its 5 captions lands in top-k) plus the standard T→A
    (each caption retrieves its one audio) — the protocol competitor numbers are reported on.
    """
    groups = torch.as_tensor(list(caption_group_ids), dtype=torch.long)
    n = int(n_audio if n_audio is not None else (int(groups.max().item()) + 1 if len(groups) else 0))
    rel = torch.zeros(n, len(groups), dtype=torch.bool)
    if len(groups):
        rel[groups, torch.arange(len(groups))] = True
    return rel


def flatten_caption_groups(captions_multi: Sequence[Sequence[str]]) -> tuple[list, list]:
    """Flatten a per-clip caption-list into (flat_captions, group_ids) for the multi-caption protocol.

    ``captions_multi[i]`` is clip i's reference captions; the returned ``group_ids[j]`` is the clip
    index that flat caption j belongs to — exactly the input ``multicaption_relevance`` expects.
    """
    flat_caps, group_ids = [], []
    for ci, caps in enumerate(captions_multi):
        for c in caps:
            flat_caps.append(c)
            group_ids.append(ci)
    return flat_caps, group_ids


def filter_clips_by_allowlist(
    captions_multi: Sequence[Sequence[str]],
    clip_ids: Sequence[str],
    allowlist,
) -> tuple[list, list]:
    """Restrict an eval set to the exact canonical split. Returns ``(kept_indices, kept_captions_multi)``
    for the clips whose id is in ``allowlist`` — used to score the standard AudioCaps/Clotho test ids
    only (published-claim comparability), keeping ``kept_indices`` aligned to the on-disk frame order.
    """
    allow = set(allowlist)
    kept = [i for i, cid in enumerate(clip_ids) if cid in allow]
    return kept, [captions_multi[i] for i in kept]


# ---------------------------------------------------------------------------- #
# Floor audit (Step-3 saturation diagnosis, docs/next_steps.md revised step (a)):
# what would the training loss be if the connector were PERFECT?
# ---------------------------------------------------------------------------- #
@torch.no_grad()
def predict_loss_floor(
    bank_embs: torch.Tensor,      # [M, d_llm] WHITENED, un-normalized (CorpusTextBank.embs)
    captions: Sequence[str],      # len M, aligned to bank rows
    loss_fn: FusionContrastiveLoss,
    logit_scale: torch.Tensor,    # scalar LOG-temperature (clamped, as training passed it)
    *,
    batch_size: int = 128,
    n_batches: int = 32,
    use_bank: bool = True,
    fn_mask_threshold: Optional[float] = None,
    fn_mask_dim: int = 1024,
    seed: int = 0,
) -> dict:
    """Perfect-alignment InfoNCE floor — the training loss with ``audio == whitened text``.

    Reuses ``FusionContrastiveLoss`` itself (audio := text ⇒ every positive has cosine 1 and
    CORAL is exactly 0), so the returned number is directly comparable to a run's printed
    loss. The floor is the irreducible cost of near-duplicate captions in the denominator;
    the gap between a run's saturated loss and this floor is optimization/capacity headroom.

    ``use_bank=False`` gives the in-batch-only floor (quantifies how much the full-corpus
    bank amplifies the near-dup noise). ``fn_mask_threshold`` additionally excludes bank
    entries whose whitened cosine to the anchor (at rung ``fn_mask_dim``) is ≥ the threshold
    — a zero-training preview of what relevance-aware FN-masking (Step 6) would buy.
    Exact-string duplicates are always union-masked, mirroring ``CorpusTextBank``.
    """
    m = bank_embs.size(0)
    if batch_size > m:
        raise ValueError(f"batch_size {batch_size} > corpus {m}")
    dev = bank_embs.device
    log_scale = logit_scale.detach().float().reshape(()).to(dev)
    fn_mask_dim = min(fn_mask_dim, bank_embs.size(1))
    rows_by_caption: dict = {}
    for i, c in enumerate(captions):
        rows_by_caption.setdefault(c, []).append(i)
    bank_fn_n = None
    if use_bank and fn_mask_threshold is not None:
        bank_fn_n = torch.nn.functional.normalize(bank_embs[:, :fn_mask_dim].float(), dim=-1)
    gen = torch.Generator().manual_seed(seed)
    losses: list = []
    masked_rows: list = []
    for _ in range(n_batches):
        idx = torch.randperm(m, generator=gen)[:batch_size].to(dev)
        text = bank_embs[idx].float()
        bank_text = bank_mask = None
        if use_bank:
            cols: list = []
            for i in idx.tolist():                     # union semantics == CorpusTextBank
                cols.extend(rows_by_caption[captions[i]])
            bank_mask = torch.zeros(batch_size, m, dtype=torch.bool, device=dev)
            bank_mask[:, cols] = True
            if bank_fn_n is not None:
                a_n = torch.nn.functional.normalize(text[:, :fn_mask_dim], dim=-1)
                bank_mask |= (a_n @ bank_fn_n.transpose(0, 1)) >= fn_mask_threshold
            bank_text = bank_embs
            masked_rows.append(float(bank_mask.float().sum(dim=1).mean()))
        _, metrics = loss_fn(text, text, log_scale, bank_text=bank_text, bank_exclude_mask=bank_mask)
        losses.append(float(metrics["infonce"]))
    t = torch.tensor(losses)
    return {
        "floor_mean": float(t.mean()),
        "floor_std": float(t.std()) if len(losses) > 1 else 0.0,
        "n_batches": n_batches, "batch_size": batch_size, "use_bank": use_bank,
        "fn_mask_threshold": fn_mask_threshold,
        "mean_masked_bank_rows_per_anchor": (sum(masked_rows) / len(masked_rows)) if masked_rows else 0.0,
    }


@torch.no_grad()
def bank_neardup_stats(
    bank_embs: torch.Tensor,      # [M, d_llm] whitened (CorpusTextBank.embs)
    captions: Sequence[str],
    *,
    dim: int = 1024,
    thresholds: Sequence[float] = (0.85, 0.9, 0.95, 0.99),
    chunk: int = 2048,
    top_examples: int = 20,
) -> dict:
    """Corpus-wide SEMANTIC near-duplicate census over the whitened text space.

    Self-pairs and exact-string duplicate groups are excluded (training already union-masks
    those), so every neighbor counted here is noise the Step-3 loss actually suffered:
    a different-string caption whose whitened embedding sits within ``threshold`` cosine.
    Returns per-threshold pair counts + per-anchor max-cosine quantiles + the top offending
    caption pairs (the qualitative confirmation that these are semantic dups, not artifacts).
    """
    dev = bank_embs.device
    m = bank_embs.size(0)
    dim = min(dim, bank_embs.size(1))
    bank_n = torch.nn.functional.normalize(bank_embs[:, :dim].float(), dim=-1).half()
    gid_map: dict = {}
    gid = torch.empty(m, dtype=torch.long)
    for i, c in enumerate(captions):
        gid[i] = gid_map.setdefault(c, len(gid_map))
    gid = gid.to(dev)
    ths = sorted(float(t) for t in thresholds)
    counts = [0.0] * len(ths)
    anchors_with = [0.0] * len(ths)
    max_cos = torch.empty(m)
    top: list = []
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        cos = (bank_n[s:e] @ bank_n.transpose(0, 1)).float()          # [c, M]
        cos.masked_fill_(gid[s:e].unsqueeze(1) == gid.unsqueeze(0), -1.0)
        mc, mj = cos.max(dim=1)
        max_cos[s:e] = mc.cpu()
        for k, t in enumerate(ths):
            over = cos >= t
            counts[k] += float(over.sum())
            anchors_with[k] += float(over.any(dim=1).sum())
        vals, order = mc.topk(min(top_examples, e - s))
        top.extend((float(v), s + int(o), int(mj[o])) for v, o in zip(vals, order))
    top.sort(reverse=True)
    seen: set = set()
    examples: list = []
    for v, i, j in top:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        examples.append({"cos": round(v, 4), "a": captions[i], "b": captions[j]})
        if len(examples) >= top_examples:
            break
    q = torch.quantile(max_cos, torch.tensor([0.5, 0.9, 0.99]))
    return {
        "dim": dim, "n": m,
        "max_offdiag_cos_mean": float(max_cos.mean()),
        "max_offdiag_cos_p50": float(q[0]),
        "max_offdiag_cos_p90": float(q[1]),
        "max_offdiag_cos_p99": float(q[2]),
        "per_threshold": {
            f"{t:g}": {
                "neighbor_pairs": int(counts[k] // 2),
                "mean_neighbors_per_anchor": counts[k] / m,
                "frac_anchors_with_neighbor": anchors_with[k] / m,
            } for k, t in enumerate(ths)
        },
        "examples": examples,
    }


def retrieval_report(
    audio_emb: torch.Tensor,
    text_emb: torch.Tensor,
    ks=(1, 5, 10),
    *,
    relevance: Optional[torch.Tensor] = None,
    map_k: int = 10,
) -> dict:
    """R@k + mAP@k for A→T and T→A. Diagonal ground-truth by default; pass ``relevance``
    (e.g. from ``semantic_relevance``) to credit multi-relevant / near-duplicate captions."""
    sims = audio_emb @ text_emb.transpose(0, 1)
    N = sims.size(0)
    if relevance is None:
        relevance = torch.eye(N, dtype=torch.bool, device=sims.device)
    rel_t = relevance.transpose(0, 1)
    a2t = _hit_recall(sims, relevance, ks)                              # audio retrieves text
    a2t[f"mAP@{map_k}"] = average_precision_at_k(sims, relevance, map_k)
    t2a = _hit_recall(sims.transpose(0, 1), rel_t, ks)                  # text retrieves audio
    t2a[f"mAP@{map_k}"] = average_precision_at_k(sims.transpose(0, 1), rel_t, map_k)
    return {**{f"a2t_{k}": v for k, v in a2t.items()}, **{f"t2a_{k}": v for k, v in t2a.items()}}


def evaluate(
    model: FusionEmbeddingModel,
    eval_manifest: FusionAudioTextManifest,
    collator: FusionCollator,
    guard: Optional[RegressionGuard] = None,
    *,
    device=None,
) -> dict:
    """AudioCaps/Clotho-style R@1/R@10 (A→T, T→A) + the MMEB-V2 regression guard.

    Device follows the model unless overridden. SEAM (HLD §10.5): swap ``eval_manifest``
    for real AudioCaps/Clotho and add the real MMEB-V2 subset pass alongside ``guard``.
    """
    audio_emb, text_emb = encode_dataset(model, eval_manifest, collator, device=device)
    report = retrieval_report(audio_emb, text_emb)
    if guard is not None:
        report["base_drift"] = guard.max_drift(model)
        report["regression_ok"] = report["base_drift"] == 0.0
    return report


# ---------------------------------------------------------------------------- #
# Training loop
# ---------------------------------------------------------------------------- #
def _to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@dataclass
class TrainState:
    history: list                       # per-optimizer-step metric dicts
    final_eval: Optional[dict] = None


def train_stage1(
    model: FusionEmbeddingModel,
    train_loader: Iterable[dict],
    loss_fn: FusionContrastiveLoss,
    cfg: FusionConfig,
    *,
    steps: int,
    eval_fn: Optional[Callable[[FusionEmbeddingModel], dict]] = None,
    device: str = "cpu",
    log_every: int = 10,
    guard_every: int = 50,
    memory_bank=None,
) -> TrainState:
    """Connector-only P1 loop with grad-accum, clipping, warmup+cosine, and the base guard.

    If ``memory_bank`` (a TextMemoryBank) is given, its frozen-text entries are used as
    extra A→T negatives — the small-VRAM lever — and each micro-batch's text embeddings
    are enqueued *after* the loss (so a positive is never its own anchor's negative).
    """
    model.to(device)
    model.train()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps)
    guard = RegressionGuard(model)

    use_autocast = cfg.use_bf16 and torch.device(device).type == "cuda"
    data_iter = itertools.cycle(train_loader)
    history: list = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        accum_metrics = None
        for _ in range(cfg.grad_accum_steps):
            batch = _to_device(next(data_iter), device)
            bank_text = memory_bank.get() if memory_bank is not None else None
            if bank_text is not None:
                bank_text = bank_text.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                out = model(batch)
                loss, metrics = loss_fn(
                    out["audio"], out["text"], out["logit_scale"], out.get("hard_neg_text"),
                    bank_text=bank_text,
                )
            (loss / cfg.grad_accum_steps).backward()
            if memory_bank is not None:                  # enqueue AFTER the loss (false-neg safe)
                memory_bank.enqueue(out["text"].detach())
            accum_metrics = metrics if accum_metrics is None else accum_metrics

        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if guard_every and (step + 1) % guard_every == 0:
            guard.check(model)

        if log_every and (step % log_every == 0 or step == steps - 1):
            history.append(
                {
                    "step": step,
                    "loss": float(accum_metrics["loss"]),
                    "infonce": float(accum_metrics["infonce"]),
                    "coral": float(accum_metrics["coral"]),
                    "acc_a2t": float(accum_metrics["acc_a2t"]),
                    "lr": scheduler.get_last_lr()[0],
                    "logit_scale": float(accum_metrics["logit_scale"]),
                }
            )

    guard.check(model)   # final, hard assertion
    state = TrainState(history=history)
    if eval_fn is not None:
        model.eval()
        state.final_eval = eval_fn(model)
    return state


# ---------------------------------------------------------------------------- #
# Tiny end-to-end training setup (tests + demo)
# ---------------------------------------------------------------------------- #
@dataclass
class TinySetup:
    """Everything needed to run + evaluate the P1 loop on synthetic CPU data."""

    model: FusionEmbeddingModel
    train_loader: DataLoader
    loss_fn: FusionContrastiveLoss
    eval_fn: Callable[[FusionEmbeddingModel], dict]
    manifest: FusionAudioTextManifest
    collator: FusionCollator
    guard: "RegressionGuard"


def build_tiny_training_setup(
    cfg: Optional[FusionConfig] = None,
    *,
    n_train: int = 8,
    batch_size: int = 8,
    vocab: int = 64,
    seed: int = 0,
) -> TinySetup:
    """Assemble the P1 loop in miniature on synthetic data.

    Random synthetic mel/text carry no semantics that *generalize*, so the honest
    integration gate is memorization: ``eval_fn`` measures retrieval over the very
    pairs being aligned. A working pipeline drives those to near-perfect R@1, which
    is the end-to-end proof that inject -> pool -> loss -> optimizer is wired right.
    """
    from ._tiny import build_tiny_model

    cfg = cfg or FusionConfig.tiny()
    set_seed(seed)
    model = build_tiny_model(cfg, vocab=vocab, seed=seed)

    manifest, collator = make_synthetic_dataset(cfg, n=n_train, vocab=vocab)
    train_loader = DataLoader(manifest, batch_size=batch_size, collate_fn=collator, shuffle=True)
    loss_fn = FusionContrastiveLoss(cfg)
    guard = RegressionGuard(model)

    def eval_fn(m: FusionEmbeddingModel) -> dict:
        return evaluate(m, manifest, collator, guard)

    return TinySetup(model, train_loader, loss_fn, eval_fn, manifest, collator, guard)


# ---------------------------------------------------------------------------- #
# SEAM (HLD §10.1-10.3): load the real frozen Qwen components
# ---------------------------------------------------------------------------- #
def load_components(
    cfg: FusionConfig,
    base_model: str = "Qwen/Qwen3-VL-Embedding-2B",
    audio_model: str = "Qwen/Qwen2.5-Omni-7B",
    device: str = "cuda",
    **kwargs,
):  # pragma: no cover - real pipeline only; not exercised by CPU tests
    """Load the frozen Qwen base + Omni audio tower (HLD §10). Thin delegator to
    ``hf_components.load_components`` (kept separate so the core package never imports
    ``transformers``). Returns
    ``(cfg, embed_tokens, base_lm, audio_encoder, tokenizer, feature_extractor)`` —
    feed the three callables to ``FusionEmbeddingModel``.
    """
    from .hf_components import load_components as _load

    return _load(cfg, base_model, audio_model, device=device, **kwargs)
