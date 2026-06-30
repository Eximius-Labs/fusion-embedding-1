"""``FusionContrastiveLoss`` — the Stage-1 objective (HLD §5.1).

    L = Σ_{D ∈ MRL} w_D · InfoNCE_D(audio, text)  +  λ_coral · CORAL(audio, text)

- Symmetric InfoNCE with a learnable modality temperature (logit scale), in-batch
  negatives, computed at every MRL rung (truncate + renormalize) and weighted-summed
  so the embedding stays truncatable consistently with the base.
- CORAL: ‖Cov(audio) − Cov(text)‖²_F / d² — keeps audio from forming its own cluster.
- Debiased contrastive (P2 knob, γ⁺): corrects audio captioning's one-to-many false
  negatives. γ⁺=0 reduces *exactly* to plain InfoNCE (asserted in tests).
- Hard negatives (P2 knob): extra confusable texts added to the A→T denominator.

The loss takes ``logit_scale`` (the model's clamped log-temperature) and applies
``.exp()`` internally, so the temperature gradient flows back to the model param.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FusionConfig


def coral_penalty(audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """‖Cov(audio) − Cov(text)‖²_F / d²  (covariance / second-moment alignment)."""
    B, d = audio.shape
    if B < 2:
        return audio.new_zeros(())
    a = audio - audio.mean(dim=0, keepdim=True)
    t = text - text.mean(dim=0, keepdim=True)
    cov_a = (a.transpose(0, 1) @ a) / (B - 1)
    cov_t = (t.transpose(0, 1) @ t) / (B - 1)
    return (cov_a - cov_t).pow(2).sum() / (d * d)


def _infonce_directional(
    anchor: torch.Tensor,        # [B,d] L2-normalized
    target: torch.Tensor,        # [B,d] L2-normalized
    scale: torch.Tensor,         # scalar exp(logit_scale)
    tau_plus: float,             # debias class-prior (0 => plain InfoNCE)
    extra_neg: Optional[torch.Tensor] = None,   # [B,K,d] per-anchor hard negatives
    shared_neg: Optional[torch.Tensor] = None,  # [M,d] memory-bank negatives shared by all anchors
) -> torch.Tensor:
    """Numerically-stable (debiased) InfoNCE for one direction; returns per-anchor loss [B].

    Negatives come from three sources, all added to the denominator: in-batch (B-1),
    per-anchor ``extra_neg`` (mined hard negatives), and a shared ``shared_neg`` bank
    (frozen-text memory bank — the lever that gives small micro-batches many negatives).
    Only the in-batch term is debiased; bank/hard negatives are treated as true negatives.
    """
    B = anchor.size(0)
    logits = scale * (anchor @ target.transpose(0, 1))           # [B,B]
    eye = torch.eye(B, dtype=torch.bool, device=logits.device)

    hn_logits = None
    if extra_neg is not None and extra_neg.numel() > 0:
        hn_logits = scale * torch.einsum("bd,bkd->bk", anchor, extra_neg)   # [B,K]
    sh_logits = None
    if shared_neg is not None and shared_neg.numel() > 0:
        sh_logits = scale * (anchor @ shared_neg.transpose(0, 1))           # [B,M]

    # per-row max shift for stability — across ALL negative groups (detached)
    row_max = logits.detach().max(dim=1).values                  # [B]
    for g in (hn_logits, sh_logits):
        if g is not None:
            row_max = torch.maximum(row_max, g.detach().max(dim=1).values)
    rm = row_max.unsqueeze(1)                                     # [B,1]

    e_pos = (logits.diagonal() - row_max).exp()                  # [B]
    e_neg_sum = (logits - rm).exp().masked_fill(eye, 0.0).sum(dim=1)   # [B] in-batch (shifted)
    N = B - 1

    extra_sum = logits.new_zeros(B)
    for g in (hn_logits, sh_logits):
        if g is not None:
            extra_sum = extra_sum + (g - rm).exp().sum(dim=1)

    if tau_plus > 0:
        # Chuang et al. 2020 debiased estimator of the in-batch true-negative mass.
        ng = (e_neg_sum - tau_plus * N * e_pos) / (1.0 - tau_plus)
        lower = N * (-scale - row_max).exp()                     # >= N·e^{-scale} (sim >= -1)
        ng = torch.maximum(ng, lower)
    else:
        ng = e_neg_sum

    denom = e_pos + ng + extra_sum
    return -(e_pos.log() - denom.log())                          # [B]


class FusionContrastiveLoss(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.mrl_dims = tuple(cfg.mrl_dims)
        self.mrl_weights = cfg.normalized_mrl_weights
        self.lambda_coral = cfg.lambda_coral
        self.tau_plus = cfg.debias_gamma

    def _infonce_at(self, audio_n, text_n, scale, hard_neg_n, bank_n) -> torch.Tensor:
        # audio->text: text-side negatives = in-batch + mined hard negs + the text bank.
        a2t = _infonce_directional(
            audio_n, text_n, scale, self.tau_plus, extra_neg=hard_neg_n, shared_neg=bank_n
        )
        # text->audio: hard negatives and the text bank are texts, so they aren't
        # negatives for a text anchor; only in-batch audio negatives apply.
        t2a = _infonce_directional(text_n, audio_n, scale, self.tau_plus)
        return 0.5 * (a2t.mean() + t2a.mean())

    def forward(
        self,
        audio: torch.Tensor,          # [B,d_llm] full-dim pooled (un-normalized)
        text: torch.Tensor,           # [B,d_llm] full-dim pooled (un-normalized)
        logit_scale: torch.Tensor,    # scalar log-temperature (already clamped)
        hard_neg_text: Optional[torch.Tensor] = None,  # [B,K,d_llm] full-dim pooled
        bank_text: Optional[torch.Tensor] = None,       # [M,d_llm] frozen-text memory bank
    ) -> tuple[torch.Tensor, dict]:
        scale = logit_scale.exp()

        infonce = audio.new_zeros(())
        for dim, w in zip(self.mrl_dims, self.mrl_weights):
            a_n = F.normalize(audio[:, :dim], dim=-1)
            t_n = F.normalize(text[:, :dim], dim=-1)
            hn_n = F.normalize(hard_neg_text[..., :dim], dim=-1) if hard_neg_text is not None else None
            bank_n = F.normalize(bank_text[:, :dim], dim=-1) if bank_text is not None else None
            infonce = infonce + w * self._infonce_at(a_n, t_n, scale, hn_n, bank_n)

        coral = coral_penalty(audio, text) if self.lambda_coral > 0 else audio.new_zeros(())
        total = infonce + self.lambda_coral * coral

        with torch.no_grad():
            metrics = {
                "loss": total.detach(),
                "infonce": infonce.detach(),
                "coral": coral.detach(),
                "logit_scale": logit_scale.detach(),
                "acc_a2t": self._inbatch_accuracy(audio, text),
            }
        return total, metrics

    @torch.no_grad()
    def _inbatch_accuracy(self, audio, text) -> torch.Tensor:
        """A→T top-1 in-batch retrieval accuracy at the default MRL rung (a sanity signal)."""
        d = self.cfg.mrl_default
        a_n = F.normalize(audio[:, :d], dim=-1)
        t_n = F.normalize(text[:, :d], dim=-1)
        sims = a_n @ t_n.transpose(0, 1)
        pred = sims.argmax(dim=1)
        target = torch.arange(audio.size(0), device=audio.device)
        return (pred == target).float().mean()
