"""Stage E gate: taxonomy + manifest + collator emit model-ready, instruction-templated batches."""

import pytest
import torch
from torch.utils.data import DataLoader

from fusion_embedding.config import FusionConfig, TASK_KEYS
from fusion_embedding.data import (
    HashingTokenizer,
    FusionCollator,
    instruction_for,
    make_synthetic_dataset,
    make_synthetic_records,
)
from fusion_embedding._tiny import build_tiny_model


def test_instruction_lookup():
    assert instruction_for("speech_content") == "Retrieve audio by spoken content."
    with pytest.raises(KeyError):
        instruction_for("nope")


def test_hashing_tokenizer_deterministic_and_in_range():
    tok = HashingTokenizer(vocab=64)
    a = tok.encode("hello world foo")
    b = tok.encode("hello world foo")
    assert a == b
    assert all(tok.first_real_id <= t < tok.vocab for t in a)
    assert tok.encode("") == []                      # empty -> no ordinary tokens


def test_manifest_item_shape_and_fields():
    cfg = FusionConfig.tiny()
    manifest, _ = make_synthetic_dataset(cfg, n=6)
    item = manifest[0]
    assert item["mel"].shape[0] == cfg.n_mels
    assert item["task"] in TASK_KEYS
    assert item["instruction"] == instruction_for(item["task"])


def test_collator_batch_keys_and_shapes():
    cfg = FusionConfig.tiny()
    manifest, collator = make_synthetic_dataset(cfg, n=8)
    batch = collator([manifest[i] for i in range(4)])

    assert batch["mel"].shape[0] == 4 and batch["mel"].shape[1] == cfg.n_mels
    assert batch["mel_mask"].shape == batch["mel"][:, 0, :].shape
    # audio side: exactly N <|audio_pad|> + eos
    assert batch["audio_input_ids"].shape == (4, cfg.n_query + 1)
    assert (batch["audio_input_ids"] == cfg.audio_pad_id).sum(1).unique().tolist() == [cfg.n_query]
    assert torch.equal(batch["audio_input_ids"][:, -1], torch.full((4,), cfg.eos_id))
    # text side ends on eos within the valid region
    assert batch["text_input_ids"].shape[0] == 4
    for b in range(4):
        last = batch["text_attention_mask"][b].sum() - 1
        assert batch["text_input_ids"][b, last] == cfg.eos_id


def test_mel_padding_mask_marks_padding():
    cfg = FusionConfig.tiny()
    manifest, collator = make_synthetic_dataset(cfg, n=8)
    batch = collator([manifest[i] for i in range(5)])
    # padded frames must be masked out; at least one item shorter than max triggers padding
    assert batch["mel_mask"].dtype == torch.bool
    assert batch["mel_mask"].any() and not batch["mel_mask"].all()


def test_batch_drives_model_end_to_end():
    """The collator's output must be directly consumable by the model (interface contract)."""
    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg)
    manifest, collator = make_synthetic_dataset(cfg, n=8)
    loader = DataLoader(manifest, batch_size=4, collate_fn=collator)
    batch = next(iter(loader))
    out = model(batch)
    assert out["audio"].shape == (4, cfg.d_llm)
    assert out["text"].shape == (4, cfg.d_llm)
    assert torch.isfinite(out["audio"]).all()


def test_records_round_robin_tasks():
    recs = make_synthetic_records(10)
    assert {r["task"] for r in recs} == set(TASK_KEYS)
    assert len({r["text"] for r in recs}) == 10        # all captions distinct
