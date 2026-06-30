"""Stage C gate: audio injection -> EOS pooling -> MRL read-out, and freeze invariants."""

import copy

import pytest
import torch

from fusion_embedding.config import FusionConfig
from fusion_embedding.model import FusionEmbeddingModel, last_token_pool, mrl_truncate_normalize
from fusion_embedding._tiny import build_tiny_model
from .conftest import make_batch


# --------------------------- pure helpers --------------------------- #
def test_last_token_pool_picks_last_valid():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    pooled = last_token_pool(hidden, mask)
    assert torch.equal(pooled[0], hidden[0, 2])   # last valid = index 2
    assert torch.equal(pooled[1], hidden[1, 1])   # last valid = index 1


def test_mrl_truncate_normalize():
    x = torch.randn(5, 32)
    for d in (8, 16, 32):
        e = mrl_truncate_normalize(x, d)
        assert e.shape == (5, d)
        assert torch.allclose(e.norm(dim=-1), torch.ones(5), atol=1e-5)


# --------------------------- forward path --------------------------- #
def test_forward_shapes(tiny_model, tiny_batch):
    out = tiny_model(tiny_batch)
    B = tiny_batch["mel"].shape[0]
    assert out["audio"].shape == (B, tiny_model.cfg.d_llm)
    assert out["text"].shape == (B, tiny_model.cfg.d_llm)
    assert torch.isfinite(out["audio"]).all() and torch.isfinite(out["text"]).all()


def test_embed_truncates_and_normalizes(tiny_model, tiny_batch):
    out = tiny_model(tiny_batch)
    for d in tiny_model.cfg.mrl_dims:
        e = tiny_model.embed(out["audio"], dim=d)
        assert e.shape[-1] == d
        assert torch.allclose(e.norm(dim=-1), torch.ones(e.shape[0]), atol=1e-5)
    with pytest.raises(ValueError):
        tiny_model.embed(out["audio"], dim=7)   # not on ladder


def test_injection_overwrites_pad_positions(tiny_cfg, tiny_model, tiny_batch):
    """The injected embeddings at <|audio_pad|> must equal the resampler tokens, not embed_tokens."""
    audio_tok = tiny_model.audio_tokens(tiny_batch["mel"], tiny_batch["mel_mask"])
    embeds = tiny_model.inject_audio(
        tiny_batch["audio_input_ids"], tiny_batch["audio_attention_mask"], audio_tok
    )
    pad_pos = tiny_batch["audio_input_ids"] == tiny_cfg.audio_pad_id
    injected = embeds[pad_pos].reshape(-1, tiny_cfg.d_llm)
    assert torch.allclose(injected, audio_tok.reshape(-1, tiny_cfg.d_llm), atol=1e-5)


def test_injection_rejects_wrong_pad_count(tiny_model, tiny_batch):
    bad = tiny_batch["audio_input_ids"].clone()
    bad[0, 0] = 3                                  # drop one pad slot in row 0
    audio_tok = tiny_model.audio_tokens(tiny_batch["mel"], tiny_batch["mel_mask"])
    with pytest.raises(ValueError):
        tiny_model.inject_audio(bad, tiny_batch["audio_attention_mask"], audio_tok)


def test_audio_embedding_depends_on_audio(tiny_model, tiny_batch):
    """Changing the mel must change the audio embedding (audio actually flows through)."""
    e1 = tiny_model.embed(tiny_model(tiny_batch)["audio"])
    b2 = dict(tiny_batch)
    b2["mel"] = tiny_batch["mel"] + 5.0
    e2 = tiny_model.embed(tiny_model(b2)["audio"])
    assert not torch.allclose(e1, e2, atol=1e-4)


# --------------------------- long audio ----------------------------- #
def test_windows_path_shapes_and_cap(tiny_cfg):
    model = build_tiny_model(tiny_cfg)
    B, W, Fdim = 2, 5, 12                          # W=5 > max_windows=3 -> capped
    mel = torch.randn(B, W, tiny_cfg.n_mels, Fdim)
    window_mask = torch.ones(B, W, dtype=torch.bool)
    window_mask[1, 3:] = False
    toks, tok_mask = model.audio_tokens_windows(mel, window_mask, None)
    kept = min(W, tiny_cfg.max_windows)
    assert toks.shape == (B, kept * tiny_cfg.n_query, tiny_cfg.d_llm)
    assert tok_mask.shape == (B, kept * tiny_cfg.n_query)


# --------------------------- freeze invariants ---------------------- #
def test_only_connector_and_temp_trainable(tiny_model):
    trainable = {id(p) for p in tiny_model.trainable_parameters()}
    expected = {id(p) for p in tiny_model.resampler.parameters()} | {id(tiny_model.logit_scale)}
    assert trainable == expected

    for comp in tiny_model.frozen_modules():
        assert all(not p.requires_grad for p in comp.parameters())
    assert tiny_model.logit_scale.requires_grad
    assert all(p.requires_grad for p in tiny_model.resampler.parameters())


def test_base_does_not_change_after_backward(tiny_model, tiny_batch):
    """Regression-guard primitive: a backward+manual step must leave frozen base bytes identical."""
    before = {n: p.detach().clone() for comp in tiny_model.frozen_modules()
              for n, p in comp.named_parameters()}
    out = tiny_model(tiny_batch)
    loss = out["audio"].pow(2).mean() + out["text"].pow(2).mean()
    loss.backward()
    # nudge every grad-bearing param the way an optimizer would
    with torch.no_grad():
        for p in tiny_model.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad
    after = {n: p.detach() for comp in tiny_model.frozen_modules() for n, p in comp.named_parameters()}
    for n, v in before.items():
        assert torch.equal(v, after[n]), f"frozen base param changed: {n}"
    # frozen params received no gradient at all
    for comp in tiny_model.frozen_modules():
        for p in comp.parameters():
            assert p.grad is None


def test_train_mode_keeps_base_in_eval(tiny_model):
    tiny_model.train()
    assert tiny_model.resampler.training is True
    for comp in tiny_model.frozen_modules():
        if hasattr(comp, "training"):
            assert comp.training is False
