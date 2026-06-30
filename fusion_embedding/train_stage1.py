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
from typing import Callable, Iterable, Optional

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
            drift = max(drift, (p.detach() - self._snapshot[pid]).abs().max().item())
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
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed every pair -> (audio_emb [M,dim], text_emb [M,dim]) at an MRL rung, L2-normed."""
    model.eval()
    dim = dim or model.cfg.mrl_default
    loader = DataLoader(manifest, batch_size=batch_size, collate_fn=collator, shuffle=False)
    a_chunks, t_chunks = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)
        a_chunks.append(mrl_truncate_normalize(out["audio"], dim).cpu())
        t_chunks.append(mrl_truncate_normalize(out["text"], dim).cpu())
    return torch.cat(a_chunks), torch.cat(t_chunks)


def recall_at_k(sims: torch.Tensor, ks=(1, 10)) -> dict:
    """Recall@k for diagonal ground-truth (row i's positive is column i)."""
    M = sims.size(0)
    targets = torch.arange(M)
    ranks = (sims.argsort(dim=1, descending=True) == targets.view(-1, 1)).float().argmax(dim=1)
    return {f"R@{k}": (ranks < k).float().mean().item() for k in ks}


def retrieval_report(audio_emb: torch.Tensor, text_emb: torch.Tensor, ks=(1, 10)) -> dict:
    sims = audio_emb @ text_emb.transpose(0, 1)
    a2t = recall_at_k(sims, ks)                              # audio retrieves text
    t2a = recall_at_k(sims.transpose(0, 1), ks)             # text retrieves audio
    return {**{f"a2t_{k}": v for k, v in a2t.items()}, **{f"t2a_{k}": v for k, v in t2a.items()}}


def evaluate(
    model: FusionEmbeddingModel,
    eval_manifest: FusionAudioTextManifest,
    collator: FusionCollator,
    guard: Optional[RegressionGuard] = None,
    *,
    device: str = "cpu",
) -> dict:
    """AudioCaps/Clotho-style R@1/R@10 (A→T, T→A) + the MMEB-V2 regression guard.

    SEAM (HLD §10.5): swap ``eval_manifest`` for real AudioCaps/Clotho and add the
    real MMEB-V2 subset pass alongside ``guard`` for the full release gate.
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
):  # pragma: no cover - real pipeline only; not exercised by CPU tests
    """Load Qwen3-VL-Embedding (base) + Qwen2.5-Omni audio tower; return the injection seams.

    Returns (cfg_with_tokens, embed_tokens, base_lm, audio_encoder) ready for
    ``FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder)``.

    TODO(fusion) — the five seams from HLD §10:
      1. Load ``base_model``; grab the LM that accepts ``inputs_embeds`` and the
         module exposing ``embed_tokens``. Add the ``<|audio_pad|>`` special token
         and resolve cfg.audio_pad_id / cfg.eos_id from the tokenizer.
      2. Confirm the base accepts ``inputs_embeds`` and that placeholder splicing
         matches its image-token path (HLD §4.1).
      3. Extract the Omni audio tower (mel + mask in 2 s blocks) -> frames [B,T,1280].
      4. Wire the Omni feature extractor into ``RealAudioProcessor`` (HLD §10.4).
      5. EOS pooling + MMEB-V2 regression guard live in ``evaluate`` (HLD §10.5).
    """
    try:
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
    except ImportError as e:
        raise ImportError("load_components needs the 'hf' extra (transformers).") from e

    raise NotImplementedError(
        "load_components is the HF wiring seam (HLD §10). The frozen-base interface it must "
        "return is fully specified and exercised by the tiny stand-ins in fusion_embedding._tiny; "
        "wire the real Qwen modules to that same (embed_tokens, base_lm, audio_encoder) contract."
    )
