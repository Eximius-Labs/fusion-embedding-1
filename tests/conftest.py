"""Shared tiny fixtures + a hand-built batch builder (used before data.py exists)."""

import torch
import pytest

from fusion_embedding.config import FusionConfig
from fusion_embedding._tiny import build_tiny_model, TINY_VOCAB


@pytest.fixture
def tiny_cfg():
    return FusionConfig.tiny()


@pytest.fixture
def tiny_model(tiny_cfg):
    return build_tiny_model(tiny_cfg, seed=0)


def make_batch(cfg: FusionConfig, batch_size: int = 4, n_mel_frames: int = 20, seed: int = 0):
    """Hand-build a minimal training batch matching the model's expected keys.

    Audio side : [N x <|audio_pad|>] + <eos>           (neutral; instruction is on text side)
    Text  side : ordinary tokens + <eos>
    """
    g = torch.Generator().manual_seed(seed)
    N = cfg.n_query

    mel = torch.randn(batch_size, cfg.n_mels, n_mel_frames, generator=g)
    mel_mask = torch.ones(batch_size, n_mel_frames, dtype=torch.bool)

    # audio token sequence: N pads then eos
    audio_ids = torch.full((batch_size, N + 1), cfg.audio_pad_id, dtype=torch.long)
    audio_ids[:, -1] = cfg.eos_id
    audio_mask = torch.ones_like(audio_ids)

    # text: a few ordinary tokens (>=3) then eos
    text_len = 5
    text_ids = torch.randint(3, TINY_VOCAB, (batch_size, text_len), generator=g)
    text_ids[:, -1] = cfg.eos_id
    text_mask = torch.ones_like(text_ids)

    return {
        "mel": mel,
        "mel_mask": mel_mask,
        "audio_input_ids": audio_ids,
        "audio_attention_mask": audio_mask,
        "text_input_ids": text_ids,
        "text_attention_mask": text_mask,
    }


@pytest.fixture
def tiny_batch(tiny_cfg):
    return make_batch(tiny_cfg, batch_size=4, seed=1)
