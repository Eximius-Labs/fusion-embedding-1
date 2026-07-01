"""Option 2 gate: precomputed-frames path — dataset, collator, model (with and without
a live audio encoder), and gradient isolation to the connector."""

import os
import tempfile

import torch
from torch.utils.data import DataLoader

from fusion_embedding.config import FusionConfig
from fusion_embedding.model import FusionEmbeddingModel
from fusion_embedding.data import CachedFrameDataset, FrameCollator, HashingTokenizer
from fusion_embedding._tiny import build_tiny_model, build_tiny_components, TINY_VOCAB


def _frame_records(cfg, n, dir_, seed=0):
    g = torch.Generator().manual_seed(seed)
    paths = []
    for i in range(n):
        T = 5 + i % 4
        p = os.path.join(dir_, f"f{i}.pt")
        torch.save({"frames": torch.randn(T, cfg.d_audio, generator=g),
                    "text": f"sound number {i}", "task": "sound"}, p)
        paths.append(p)
    return paths


def _collator(cfg, vocab=TINY_VOCAB):
    tok = HashingTokenizer(vocab=vocab, pad_id=cfg.pad_id, audio_pad_id=cfg.audio_pad_id, eos_id=cfg.eos_id)
    return FrameCollator(cfg, tok)


def test_frame_dataset_and_collator_shapes():
    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    ds = CachedFrameDataset(_frame_records(cfg, 5, d))
    assert len(ds) == 5
    batch = _collator(cfg)([ds[i] for i in range(4)])
    assert batch["frames"].shape[0] == 4 and batch["frames"].shape[-1] == cfg.d_audio
    assert batch["frame_mask"].shape == batch["frames"].shape[:2]
    assert (batch["audio_input_ids"] == cfg.audio_pad_id).sum(1).unique().tolist() == [cfg.n_query]
    # variable T -> some padding present
    assert batch["frame_mask"].any() and not batch["frame_mask"].all()


def test_model_frames_path_matches_manual_resample():
    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg).eval()
    frames = torch.randn(3, 7, cfg.d_audio)
    fm = torch.ones(3, 7, dtype=torch.bool)
    a = model.audio_tokens_from_frames(frames, fm)
    b = model.resampler(frames, fm)
    assert torch.allclose(a, b)


def test_forward_with_frames_batch_skips_encoder():
    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    ds = CachedFrameDataset(_frame_records(cfg, 6, d))
    loader = DataLoader(ds, batch_size=3, collate_fn=_collator(cfg))
    model = build_tiny_model(cfg)
    batch = next(iter(loader))
    out = model(batch)
    assert out["audio"].shape == (3, cfg.d_llm)
    assert out["text"].shape == (3, cfg.d_llm)
    assert torch.isfinite(out["audio"]).all()


def test_model_without_audio_encoder_trains_on_frames():
    """Build the model with audio_encoder=None (train-from-frames setup): forward works,
    grads reach the connector, and the base stays frozen."""
    cfg = FusionConfig.tiny()
    torch.manual_seed(0)
    embed_tokens, base_lm, _audio = build_tiny_components(cfg)
    model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder=None)

    d = tempfile.mkdtemp()
    ds = CachedFrameDataset(_frame_records(cfg, 6, d))
    batch = _collator(cfg)([ds[i] for i in range(4)])

    out = model(batch)
    loss = out["audio"].pow(2).mean() + out["text"].pow(2).mean()
    loss.backward()
    # connector got grads
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.resampler.parameters())
    # frozen base got none
    for comp in (model.embed_tokens, model.base_lm):
        for p in comp.parameters():
            assert p.grad is None


def test_mel_path_errors_without_encoder():
    cfg = FusionConfig.tiny()
    embed_tokens, base_lm, _ = build_tiny_components(cfg)
    model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder=None)
    import pytest
    with pytest.raises(RuntimeError):
        model.audio_tokens(torch.randn(2, cfg.n_mels, 10))
