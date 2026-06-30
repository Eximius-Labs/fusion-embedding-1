"""Stage A gate: config invariants (HLD §3 anchors are encoded and self-consistent)."""

import math

import pytest

from fusion_embedding.config import FusionConfig, TASK_INSTRUCTIONS, TASK_KEYS


def test_default_anchors_match_hld():
    c = FusionConfig()
    assert c.d_audio == 1280
    assert c.d_llm == 2048
    assert c.mrl_dims == (2048, 1536, 1024, 512, 256, 128, 64)
    assert c.mrl_default == 1024
    assert c.frames_per_window == 750  # ~25 fps * 30 s


def test_logit_scale_init_is_clip_temp():
    c = FusionConfig()
    assert math.isclose(c.logit_scale_init, math.log(1 / 0.07))


def test_mrl_weights_equal_and_normalized():
    c = FusionConfig()
    w = c.normalized_mrl_weights
    assert len(w) == len(c.mrl_dims)
    assert math.isclose(sum(w), 1.0)
    assert all(math.isclose(x, w[0]) for x in w)


def test_custom_mrl_weights_normalize():
    c = FusionConfig(mrl_weights=(1, 1, 2, 0, 0, 0, 0))
    w = c.normalized_mrl_weights
    assert math.isclose(sum(w), 1.0)
    assert math.isclose(w[2], 0.5)


def test_taxonomy_has_five_tasks():
    assert len(TASK_KEYS) == 5
    assert set(TASK_KEYS) == set(TASK_INSTRUCTIONS)
    assert "speech_content" in TASK_KEYS


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(d_llm=2048, resampler_heads=7),          # not divisible
        dict(mrl_dims=(4096,)),                         # rung > d_llm
        dict(mrl_default=999),                          # default not in ladder
        dict(mrl_dims=(2048, 2048)),                    # duplicate rung
        dict(mrl_weights=(1.0, 1.0)),                   # wrong length
        dict(n_query=0),                                # non-positive
    ],
)
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError):
        FusionConfig(**kwargs)


def test_tiny_is_valid_and_structurally_same():
    c = FusionConfig.tiny()
    assert c.mrl_default in c.mrl_dims
    assert c.d_llm % c.resampler_heads == 0
    assert all(d <= c.d_llm for d in c.mrl_dims)
    assert c.audio_pad_id == 1 and c.eos_id == 2


def test_with_tokens_returns_copy():
    c = FusionConfig()
    c2 = c.with_tokens(audio_pad_id=151000, eos_id=151643)
    assert c.audio_pad_id == -1          # original untouched
    assert c2.audio_pad_id == 151000 and c2.eos_id == 151643
