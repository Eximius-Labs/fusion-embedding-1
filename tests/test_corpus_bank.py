"""Step 3 gate: full-corpus frozen-text negative bank — exclusion mask correctness,
loss integration (bank must change the loss but never poison the positive), E2E train step."""

import os
import tempfile

import torch

from fusion_embedding.config import FusionConfig
from fusion_embedding.losses import FusionContrastiveLoss
from fusion_embedding.memory_bank import CorpusTextBank, build_corpus_bank_from_cache
from fusion_embedding.data import (
    FrameCollator, HashingTokenizer, shard_starts_from, write_frame_shard, write_text_emb_shard,
)
from fusion_embedding._tiny import build_tiny_model, TINY_VOCAB


def _bank(cfg, captions, seed=0):
    g = torch.Generator().manual_seed(seed)
    return CorpusTextBank(torch.randn(len(captions), cfg.d_llm, generator=g), captions)


def test_exclude_mask_unions_batch_captions_and_duplicates():
    cfg = FusionConfig.tiny()
    caps = ["dog barks", "rain falls", "dog barks", "bell rings"]     # dup at rows 0 and 2
    bank = _bank(cfg, caps)
    mask = bank.exclude_mask(["dog barks", "unseen caption"])
    assert mask.shape == (2, 4)
    # union semantics: batch captions' rows (incl. ALL duplicates) masked for EVERY anchor —
    # the bank supplies strictly out-of-batch negatives.
    assert mask[0].tolist() == [True, False, True, False]
    assert mask[1].tolist() == [True, False, True, False]
    assert bank.exclude_mask(["unseen caption"]).sum() == 0           # unseen masks nothing
    assert bank.n_duplicate_captions == 2


def test_bank_changes_loss_but_masked_positive_does_not():
    cfg = FusionConfig.tiny()
    torch.manual_seed(0)
    B = 4
    audio = torch.randn(B, cfg.d_llm)
    text = torch.randn(B, cfg.d_llm)
    scale = torch.tensor(2.0)
    loss_fn = FusionContrastiveLoss(cfg)
    batch_caps = [f"cap {i}" for i in range(B)]

    base_loss, _ = loss_fn(audio, text, scale)

    # bank of unrelated negatives -> loss must INCREASE (denominator grows)
    bank = CorpusTextBank(torch.randn(50, cfg.d_llm), [f"bank {j}" for j in range(50)])
    with_bank, _ = loss_fn(audio, text, scale, bank_text=bank.embs,
                           bank_exclude_mask=bank.exclude_mask(batch_caps))
    assert with_bank > base_loss

    # bank that CONTAINS the batch positives, properly masked -> positives contribute nothing:
    # loss equals the same bank WITHOUT those rows entirely.
    extra = torch.randn(10, cfg.d_llm)
    bank_with_pos = CorpusTextBank(torch.cat([text, extra]),
                                   batch_caps + [f"bank {j}" for j in range(10)])
    masked, _ = loss_fn(audio, text, scale, bank_text=bank_with_pos.embs,
                        bank_exclude_mask=bank_with_pos.exclude_mask(batch_caps))
    bank_without_pos = CorpusTextBank(extra, [f"bank {j}" for j in range(10)])
    reference, _ = loss_fn(audio, text, scale, bank_text=bank_without_pos.embs,
                           bank_exclude_mask=bank_without_pos.exclude_mask(batch_caps))
    assert torch.allclose(masked, reference, atol=1e-5)

    # UNMASKED positives in the bank poison the loss (the failure the mask prevents)
    poisoned, _ = loss_fn(audio, text, scale, bank_text=bank_with_pos.embs)
    assert poisoned > masked


def test_build_corpus_bank_from_cache_roundtrip_and_eval_exclusion():
    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg)
    d = tempfile.mkdtemp()
    n, shard_size = 10, 4
    g = torch.Generator().manual_seed(3)
    recs = [{"frames": torch.randn(3, cfg.d_audio, generator=g), "text": f"cap {i}", "task": "sound"}
            for i in range(n)]
    paths = []
    raws = []
    for s, start in enumerate(range(0, n, shard_size)):
        p = os.path.join(d, f"shard-{s:03d}.pt")
        chunk = recs[start:start + shard_size]
        write_frame_shard(p, chunk, half=True)
        raw = torch.randn(len(chunk), cfg.d_llm, generator=g)
        write_text_emb_shard(p, raw)
        raws.append(raw)
        paths.append(p)
    all_raw = torch.cat(raws)
    model.text_whitening.fit(torch.randn(64, cfg.d_llm, generator=g))   # non-trivial whitening

    bank = build_corpus_bank_from_cache(paths, [r["text"] for r in recs],
                                        model.text_whitening, exclude={0, 5})
    assert len(bank) == n - 2                                          # eval rows dropped
    # row order = global order minus excluded; whitened values match (fp16 tolerance)
    keep = [i for i in range(n) if i not in (0, 5)]
    expect = model.text_whitening(all_raw[keep].float())
    assert torch.allclose(bank.embs.float(), expect, atol=1e-2)
    # excluded captions are not maskable rows (they're simply absent)
    assert bank.exclude_mask(["cap 0"]).sum() == 0
    assert bank.exclude_mask(["cap 1"]).sum() == 1


def test_train_step_with_bank_and_accumulation_updates_only_connector():
    cfg = FusionConfig.tiny()
    torch.manual_seed(1)
    model = build_tiny_model(cfg)
    tok = HashingTokenizer(vocab=TINY_VOCAB, pad_id=cfg.pad_id, audio_pad_id=cfg.audio_pad_id,
                           eos_id=cfg.eos_id)
    collator = FrameCollator(cfg, tok)
    loss_fn = FusionContrastiveLoss(cfg)
    from fusion_embedding.train_stage1 import build_optimizer
    opt = build_optimizer(model, cfg)

    bank = _bank(cfg, [f"bank cap {j}" for j in range(30)], seed=2)
    before = [p.detach().clone() for p in model.resampler.parameters()]

    accum = 2
    opt.zero_grad(set_to_none=True)
    for micro in range(accum):                                        # the trainer's accumulation shape
        recs = [{"frames": torch.randn(4, cfg.d_audio), "text": f"clip {micro}-{i}", "task": "sound",
                 "instruction": "describe the sound", "text_emb": torch.randn(cfg.d_llm)}
                for i in range(3)]
        batch = collator(recs)
        assert batch["texts"] == [r["text"] for r in recs]            # collator passes captions through
        out = model(batch)
        loss, _ = loss_fn(out["audio"], out["text"], out["logit_scale"], bank_text=bank.embs,
                          bank_exclude_mask=bank.exclude_mask(batch["texts"]))
        (loss / accum).backward()
    opt.step()

    assert any(not torch.equal(b, a) for b, a in zip(before, model.resampler.parameters()))
    for comp in model.frozen_modules():
        for p in comp.parameters():
            assert p.grad is None                                     # base untouched, drift stays 0
