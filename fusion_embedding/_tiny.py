"""Tiny CPU stand-ins for the frozen Qwen components.

These have the *interfaces* the real components expose (HLD §10 seams) but tiny
dims and random weights, so the full Fusion pipeline — injection, EOS pooling,
MRL, loss, optimizer step, eval — runs end-to-end on CPU in milliseconds. They are
NOT models of Qwen; they exist purely to exercise the plumbing the production code
will run unchanged once ``load_components`` wires the real towers in.

Tiny vocab layout (matches ``FusionConfig.tiny()``):
    0 = pad,  1 = <|audio_pad|>,  2 = <eos>,  3.. = ordinary tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import FusionConfig
from .model import FusionEmbeddingModel

TINY_VOCAB = 64


class TinyAudioEncoder(nn.Module):
    """Mel -> audio frames, with a stride-2 time downsample mimicking Omni's ~25 fps pooling."""

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.n_mels, cfg.d_audio)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_audio, nhead=2, dim_feedforward=cfg.d_audio * 2,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, mel: torch.Tensor, mel_mask: torch.Tensor | None = None):
        """mel [B,n_mels,F], mel_mask [B,F] (1=valid) -> (frames [B,T,d_audio], frame_mask [B,T])."""
        B, n_mels, Fdim = mel.shape
        x = mel.transpose(1, 2)                                   # [B,F,n_mels]
        x = self.proj(x)                                          # [B,F,d_audio]
        if mel_mask is None:
            mel_mask = torch.ones(B, Fdim, dtype=torch.bool, device=mel.device)
        key_padding = mel_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding)
        x = x.transpose(1, 2)                                     # [B,d_audio,F]
        x = self.pool(x).transpose(1, 2)                          # [B,T,d_audio]
        # Downsample the mask the same way (a frame is valid if any source frame was).
        m = mel_mask.float().unsqueeze(1)                         # [B,1,F]
        m = torch.nn.functional.max_pool1d(m, kernel_size=2, stride=2, ceil_mode=True)
        frame_mask = m.squeeze(1) > 0                             # [B,T]
        return x, frame_mask


class TinyLM(nn.Module):
    """Frozen base stand-in: owns ``embed_tokens`` and consumes ``inputs_embeds``.

    Mirrors how a real Qwen embedding model is driven: call ``embed_tokens`` to get
    input embeddings, then forward the body with ``inputs_embeds``. Uses a
    bidirectional encoder (the injection/pooling plumbing under test is identical
    regardless of attention direction).
    """

    def __init__(self, cfg: FusionConfig, vocab: int = TINY_VOCAB):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, cfg.d_llm, padding_idx=cfg.pad_id)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_llm, nhead=cfg.resampler_heads, dim_feedforward=cfg.d_llm * 2,
            batch_first=True, dropout=0.0,
        )
        self.body = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        key_padding = attention_mask == 0
        # Pin to the standard (non-fused) attention path: nn.TransformerEncoder silently
        # switches to a fused fast path under eval+no_grad UNLESS layer hooks exist, which
        # would make hook-bearing and hook-free stand-ins numerically differ for kernel
        # reasons unrelated to the architecture under test (adapter bitwise-invariance
        # tests need path parity; the real HF base has no hook-conditional kernels).
        import torch.backends.mha as _mha
        prev = _mha.get_fastpath_enabled()
        _mha.set_fastpath_enabled(False)
        try:
            return self.body(inputs_embeds, src_key_padding_mask=key_padding)
        finally:
            _mha.set_fastpath_enabled(prev)


def build_tiny_components(cfg: FusionConfig, vocab: int = TINY_VOCAB):
    """Return (embed_tokens, base_lm, audio_encoder) sharing one frozen embedding table."""
    lm = TinyLM(cfg, vocab=vocab)
    audio = TinyAudioEncoder(cfg)
    # embed_tokens is the SAME module the base_lm uses (weight tying), as with real Qwen.
    return lm.embed_tokens, lm, audio


def build_tiny_model(cfg: FusionConfig | None = None, vocab: int = TINY_VOCAB, seed: int = 0) -> FusionEmbeddingModel:
    cfg = cfg or FusionConfig.tiny()
    torch.manual_seed(seed)
    embed_tokens, base_lm, audio_encoder = build_tiny_components(cfg, vocab=vocab)
    return FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder)
