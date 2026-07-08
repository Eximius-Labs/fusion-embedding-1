"""init_trainables_from_ckpt — the second-stage fine-tune warm-start (Stage 3)."""
import pytest
import torch

from fusion_embedding.config import FusionConfig
from fusion_embedding.train_stage1 import build_tiny_training_setup, init_trainables_from_ckpt


def _ckpt_from(model) -> dict:
    return {
        "resampler": model.resampler.state_dict(),
        "text_whitening": model.text_whitening.state_dict(),
        "logit_scale": model.logit_scale.detach().cpu(),
        "config": {"d_resampler": model.cfg.d_resampler, "n_query": model.cfg.n_query},
        "base_4bit": False,
    }


def _first_param(model):
    return next(iter(model.resampler.parameters()))


def test_init_loads_trainables():
    cfg = FusionConfig.tiny(d_resampler=32)
    src = build_tiny_training_setup(cfg, n_train=4, batch_size=2, seed=0).model
    dst = build_tiny_training_setup(cfg, n_train=4, batch_size=2, seed=1).model
    assert not torch.allclose(_first_param(src), _first_param(dst)), "seeds must differ"

    with torch.no_grad():
        src.logit_scale.copy_(torch.tensor(3.21))
    info = init_trainables_from_ckpt(dst, _ckpt_from(src))

    assert torch.allclose(_first_param(src), _first_param(dst))
    assert float(dst.logit_scale) == pytest.approx(3.21, abs=1e-5)
    assert set(info["loaded"]) == {"resampler", "text_whitening", "logit_scale"}


def test_init_rejects_architecture_mismatch():
    cfg = FusionConfig.tiny(d_resampler=32)
    src = build_tiny_training_setup(cfg, n_train=4, batch_size=2, seed=0).model
    dst = build_tiny_training_setup(cfg, n_train=4, batch_size=2, seed=1).model
    ck = _ckpt_from(src)
    ck["config"]["d_resampler"] = 999
    before = _first_param(dst).clone()
    with pytest.raises(ValueError, match="mismatch"):
        init_trainables_from_ckpt(dst, ck)
    assert torch.allclose(_first_param(dst), before), "failed init must not mutate the model"


def test_init_tolerates_missing_optional_keys():
    cfg = FusionConfig.tiny(d_resampler=32)
    src = build_tiny_training_setup(cfg, n_train=4, batch_size=2, seed=0).model
    dst = build_tiny_training_setup(cfg, n_train=4, batch_size=2, seed=1).model
    ck = _ckpt_from(src)
    del ck["text_whitening"]
    ck["logit_scale"] = None
    info = init_trainables_from_ckpt(dst, ck)
    assert info["loaded"] == ["resampler"]
    assert torch.allclose(_first_param(src), _first_param(dst))
