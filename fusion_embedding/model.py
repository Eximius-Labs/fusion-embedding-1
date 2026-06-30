"""``FusionResampler`` (trained connector) + ``FusionEmbeddingModel`` (audio injection,
EOS pooling, MRL truncation, freeze logic).

The frozen base is injected as three duck-typed callables so the *exact same* model
code runs with real Qwen components (wired in ``train_stage1.load_components``) or
with tiny CPU stand-ins (``_tiny.build_tiny_components``):

    embed_tokens : nn.Module   ids [B,S]                       -> embeds [B,S,d_llm]
    base_lm      : callable     (inputs_embeds, attention_mask)  -> hidden [B,S,d_llm]
    audio_encoder: callable     (mel, mel_mask)                  -> (frames [B,T,d_audio], frame_mask [B,T])

HLD §4 is the spec; the injection mechanic is §4.1, the resampler is §4.2.
"""

from __future__ import annotations

import math
from typing import Callable, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FusionConfig


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #
def sinusoidal_positions(length: int, dim: int, device, dtype) -> torch.Tensor:
    """Standard sinusoidal positional encoding [length, dim] over the time axis."""
    if dim % 2 != 0:
        # odd dims: compute on dim+1 and slice
        pe = sinusoidal_positions(length, dim + 1, device, dtype)
        return pe[:, :dim]
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """EOS / last-token pooling (HLD §3): hidden state at the last valid position.

    hidden [B,S,d], attention_mask [B,S] (1=valid) -> [B,d].
    """
    lengths = attention_mask.long().sum(dim=1) - 1            # index of last valid token
    lengths = lengths.clamp(min=0)
    idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, idx).squeeze(1)


def mrl_truncate_normalize(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Prefix-truncate to `dim` then L2-normalize (the Matryoshka read-out)."""
    return F.normalize(x[..., :dim], p=2, dim=-1)


# ---------------------------------------------------------------------------- #
# FusionResampler (HLD §4.2)
# ---------------------------------------------------------------------------- #
class _ResamplerBlock(nn.Module):
    """Pre-norm: latent self-attention -> cross-attention (queries attend frames) -> FFN."""

    def __init__(self, dim: int, heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # latent self-attention
        h = self.norm_sa(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        # cross-attention: queries -> audio frames (K/V), with frame padding mask
        h = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        q = q + self.cross_attn(h, kv_n, kv_n, key_padding_mask=key_padding_mask, need_weights=False)[0]
        # FFN
        q = q + self.ffn(self.norm_ff(q))
        return q


class FusionResampler(nn.Module):
    """Perceiver-resampler: variable-length audio frames -> N fixed latent tokens.

    The only trained component in Stage 1 (besides the learnable temperature).
    Param count is single-digit millions at production dims.
    """

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        dr = cfg.d_resampler
        # Bottleneck: process at d_resampler, project audio in and tokens out.
        self.in_proj = nn.Linear(cfg.d_audio, dr)
        self.queries = nn.Parameter(torch.empty(cfg.n_query, dr))
        nn.init.normal_(self.queries, std=0.02)
        self.blocks = nn.ModuleList(
            _ResamplerBlock(dr, cfg.resampler_heads, cfg.resampler_ffn_mult, cfg.resampler_dropout)
            for _ in range(cfg.resampler_depth)
        )
        self.out_proj = nn.Linear(dr, cfg.d_llm)
        self.out_norm = nn.LayerNorm(cfg.d_llm)

    def forward(self, frames: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """frames [B,T,d_audio], frame_mask [B,T] bool (True=valid) -> [B,N,d_llm]."""
        B, T, _ = frames.shape
        if frame_mask is None:
            frame_mask = torch.ones(B, T, dtype=torch.bool, device=frames.device)

        kv = self.in_proj(frames)                                  # [B,T,d_resampler]
        kv = kv + sinusoidal_positions(T, kv.size(-1), kv.device, kv.dtype).unsqueeze(0)

        # nn.MultiheadAttention: key_padding_mask True == ignore.
        key_padding = ~frame_mask
        # Guard: a fully-masked row makes attention return NaN. Unmask its first slot.
        fully_masked = key_padding.all(dim=1)
        if fully_masked.any():
            key_padding = key_padding.clone()
            key_padding[fully_masked, 0] = False

        q = self.queries.unsqueeze(0).expand(B, -1, -1)            # [B,N,d_resampler]
        for block in self.blocks:
            q = block(q, kv, key_padding)
        return self.out_norm(self.out_proj(q))                     # [B,N,d_llm]


# ---------------------------------------------------------------------------- #
# FusionEmbeddingModel (HLD §4.1)
# ---------------------------------------------------------------------------- #
class FusionEmbeddingModel(nn.Module):
    """Frozen base + frozen audio encoder + trained FusionResampler + learnable temp."""

    def __init__(
        self,
        cfg: FusionConfig,
        embed_tokens: nn.Module,
        base_lm: Callable[..., torch.Tensor],
        audio_encoder: Callable[..., tuple],
    ):
        super().__init__()
        if cfg.audio_pad_id < 0 or cfg.eos_id < 0:
            raise ValueError("cfg token ids must be resolved (use cfg.with_tokens(...) / FusionConfig.tiny()).")
        self.cfg = cfg
        self.embed_tokens = embed_tokens
        self.base_lm = base_lm
        self.audio_encoder = audio_encoder
        self.resampler = FusionResampler(cfg)

        scale = torch.tensor(float(cfg.logit_scale_init))
        if cfg.learnable_temperature:
            self.logit_scale = nn.Parameter(scale)
        else:
            self.register_buffer("logit_scale", scale)

        self._freeze_base()

    # ----------------------------- freeze logic ---------------------------- #
    def _freeze_base(self) -> None:
        """Freeze base LLM, embed_tokens, and audio encoder; only resampler + temp train."""
        for component in (self.embed_tokens, self.base_lm, self.audio_encoder):
            if isinstance(component, nn.Module):
                component.eval()
                for p in component.parameters():
                    p.requires_grad_(False)

    def train(self, mode: bool = True):  # noqa: D401
        """Override: frozen components stay in eval() even when the model is training."""
        super().train(mode)
        for component in (self.embed_tokens, self.base_lm, self.audio_encoder):
            if isinstance(component, nn.Module):
                component.eval()
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.resampler.parameters()
        if isinstance(self.logit_scale, nn.Parameter):
            yield self.logit_scale

    def frozen_modules(self):
        return (self.embed_tokens, self.base_lm, self.audio_encoder)

    # ------------------------------ audio path ----------------------------- #
    def audio_tokens(self, mel: torch.Tensor, mel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single 30 s window: mel [B,n_mels,F] -> N latent tokens [B,N,d_llm]."""
        frames, frame_mask = self.audio_encoder(mel, mel_mask)     # [B,T,d_audio], [B,T]
        return self.resampler(frames, frame_mask)

    def audio_tokens_windows(
        self,
        mel: torch.Tensor,            # [B,W,n_mels,F]
        window_mask: torch.Tensor,    # [B,W] bool (True=real window)
        mel_mask: Optional[torch.Tensor] = None,  # [B,W,F]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Long audio (>30 s): resample each window to N tokens, concat up to max_windows.

        Returns (tokens [B, W*N, d_llm], token_mask [B, W*N]) where padded-window
        token slots are masked out. The caller lays out W*N ``<|audio_pad|>`` slots.
        """
        B, W, n_mels, Fdim = mel.shape
        W = min(W, self.cfg.max_windows)
        mel = mel[:, :W]
        window_mask = window_mask[:, :W]
        flat_mel = mel.reshape(B * W, n_mels, Fdim)
        flat_mm = None if mel_mask is None else mel_mask[:, :W].reshape(B * W, Fdim)
        toks = self.audio_tokens(flat_mel, flat_mm)               # [B*W, N, d]
        toks = toks.reshape(B, W * self.cfg.n_query, self.cfg.d_llm)
        token_mask = window_mask.unsqueeze(-1).expand(B, W, self.cfg.n_query).reshape(B, W * self.cfg.n_query)
        return toks, token_mask

    def inject_audio(
        self,
        input_ids: torch.Tensor,        # [B,S], with exactly M <|audio_pad|> per row
        attention_mask: torch.Tensor,   # [B,S]
        audio_tokens: torch.Tensor,     # [B,M,d_llm]
    ) -> torch.Tensor:
        """Overwrite ``<|audio_pad|>`` embeddings with the resampler's audio tokens (HLD §4.1)."""
        embeds = self.embed_tokens(input_ids)                     # [B,S,d_llm]
        pad_positions = input_ids == self.cfg.audio_pad_id        # [B,S] bool
        per_row = pad_positions.sum(dim=1)
        M = audio_tokens.size(1)
        if not torch.all(per_row == M):
            raise ValueError(
                f"each row must have exactly M={M} '{self.cfg.audio_pad_token}' slots; got counts {per_row.tolist()}"
            )
        # Row-major flatten aligns slot k of row b with audio_tokens[b,k].
        embeds = embeds.clone()
        embeds[pad_positions] = audio_tokens.reshape(-1, audio_tokens.size(-1)).to(embeds.dtype)
        return embeds

    def encode_audio(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Inject audio -> frozen LLM -> EOS pooling. Returns full-dim pooled [B,d_llm]."""
        embeds = self.inject_audio(input_ids, attention_mask, audio_tokens)
        hidden = self.base_lm(inputs_embeds=embeds, attention_mask=attention_mask)
        return last_token_pool(hidden, attention_mask)

    # ------------------------------ text path ------------------------------ #
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Frozen LLM over the instruction+text side -> EOS pooling. Full-dim [B,d_llm]."""
        embeds = self.embed_tokens(input_ids)
        hidden = self.base_lm(inputs_embeds=embeds, attention_mask=attention_mask)
        return last_token_pool(hidden, attention_mask)

    # ------------------------------ read-out ------------------------------- #
    def embed(self, pooled: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        """MRL-truncate to a ladder rung (default 1024) + L2-normalize -> shared-space embedding."""
        dim = dim or self.cfg.mrl_default
        if dim not in self.cfg.mrl_dims:
            raise ValueError(f"dim={dim} not on the MRL ladder {self.cfg.mrl_dims}")
        return mrl_truncate_normalize(pooled, dim)

    def clamped_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(max=self.cfg.logit_scale_max)

    # ------------------------------ forward -------------------------------- #
    def forward(self, batch: dict) -> dict:
        """Training forward: returns full-dim pooled audio/text + the raw log temperature.

        The loss tiles over MRL rungs (truncate+renorm internally), so we hand it the
        un-truncated, un-normalized pooled vectors plus ``logit_scale``.
        """
        if "mel_windows" in batch:
            audio_tok, _ = self.audio_tokens_windows(
                batch["mel_windows"], batch["window_mask"], batch.get("mel_mask")
            )
        else:
            audio_tok = self.audio_tokens(batch["mel"], batch.get("mel_mask"))

        pooled_audio = self.encode_audio(
            batch["audio_input_ids"], batch["audio_attention_mask"], audio_tok
        )
        pooled_text = self.encode_text(batch["text_input_ids"], batch["text_attention_mask"])
        out = {"audio": pooled_audio, "text": pooled_text, "logit_scale": self.clamped_logit_scale()}
        if "hard_neg_text_input_ids" in batch:
            out["hard_neg_text"] = self.encode_text(
                batch["hard_neg_text_input_ids"], batch["hard_neg_text_attention_mask"]
            )
        return out
