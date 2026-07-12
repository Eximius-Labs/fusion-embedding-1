"""Option 2 gate: precomputed-frames path — dataset, collator, model (with and without
a live audio encoder), and gradient isolation to the connector."""

import os
import tempfile

import torch
from torch.utils.data import DataLoader

from fusion_embedding.config import FusionConfig
from fusion_embedding.model import FusionEmbeddingModel
from fusion_embedding.data import (
    CachedFrameDataset, FrameCollator, HashingTokenizer,
    ShardedFrameDataset, cumulative_starts, load_frame_clips, load_recaption,
    shard_starts_from, write_frame_shard, text_emb_shard_path, write_text_emb_shard,
)
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


# ---------------------------- sharded frames (streaming) ---------------------------- #
def _write_shards(cfg, n, shard_size, dir_, seed=0):
    """Write n clips (variable T) across shard files; return (records, sorted shard paths)."""
    g = torch.Generator().manual_seed(seed)
    recs = [{"frames": torch.randn(3 + i % 5, cfg.d_audio, generator=g),
             "text": f"sound number {i}", "task": "sound"} for i in range(n)]
    paths = []
    for s, start in enumerate(range(0, n, shard_size)):
        p = os.path.join(dir_, f"shard-{s:03d}.pt")
        write_frame_shard(p, recs[start:start + shard_size], half=True)
        paths.append(p)
    return recs, paths


def test_shard_write_and_load_by_global_index():
    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    recs, paths = _write_shards(cfg, n=25, shard_size=10, dir_=d)
    starts = shard_starts_from(len(paths), shard_size=10, n_total=25)     # [0, 10, 20]
    got = load_frame_clips(paths, starts, global_indices=[0, 12, 24])
    assert [g["text"] for g in got] == ["sound number 0", "sound number 12", "sound number 24"]
    assert got[0]["frames"].dtype == torch.float32                       # fp16-on-disk -> float at use
    assert got[1]["frames"].shape == recs[12]["frames"].shape            # correct clip pulled


def test_sharded_dataset_yields_all_nonexcluded_once():
    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    _write_shards(cfg, n=25, shard_size=10, dir_=d)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("shard-"))
    starts = shard_starts_from(len(paths), shard_size=10, n_total=25)
    exclude = {0, 12, 24}                                                # held-out eval indices
    ds = ShardedFrameDataset(paths, starts, exclude=exclude, shuffle_buffer=4, seed=1)
    seen = [it["text"] for it in ds]
    expected = {f"sound number {i}" for i in range(25)} - {f"sound number {i}" for i in exclude}
    assert set(seen) == expected and len(seen) == len(expected)          # each non-excluded exactly once
    assert all(set(it) == {"frames", "text", "task", "instruction"} for it in ds)  # collator-ready


def test_sharded_dataset_captions_override_and_skip_exclusion():
    """Leg B recaption path: overridden global indices yield the NEW caption, SKIP indices
    (routed into ``exclude``) never appear, everything else keeps its original text."""
    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    _write_shards(cfg, n=25, shard_size=10, dir_=d)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("shard-"))
    starts = shard_starts_from(len(paths), shard_size=10, n_total=25)
    override = {3: "a dog barks", 11: "rain falls on a roof", 24: "a bell rings"}
    skip = {0, 12}
    ds = ShardedFrameDataset(paths, starts, exclude=skip, shuffle_buffer=4, seed=1,
                             captions_override=override)
    seen = {it["text"] for it in ds}
    assert {"a dog barks", "rain falls on a roof", "a bell rings"} <= seen   # overrides applied
    assert not {f"sound number {i}" for i in override} & seen               # originals replaced
    assert not {f"sound number {i}" for i in skip} & seen                   # SKIPs dropped
    assert len(seen) == 23                                                  # 25 - 2 skipped


def test_load_frame_clips_applies_captions_override():
    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    _write_shards(cfg, n=25, shard_size=10, dir_=d)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("shard-"))
    starts = shard_starts_from(len(paths), shard_size=10, n_total=25)
    clips = load_frame_clips(paths, starts, [2, 13, 20],
                             captions_override={13: "waves crash on a shore"})
    assert [c["text"] for c in clips] == ["sound number 2", "waves crash on a shore",
                                          "sound number 20"]


def test_captions_override_aligns_on_varying_shard_sizes():
    """Regression for the 2026-07-10 miswrite: webdataset shards VARY in size (one per source
    tar, 515–527 observed), so global offsets MUST come from cumulative actual lengths — the
    uniform p*shard_size formula drifts and pairs overrides with the wrong clips."""
    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    sizes = [7, 5, 9, 4]                                       # deliberately non-uniform
    g = torch.Generator().manual_seed(0)
    recs = [{"frames": torch.randn(3 + i % 5, cfg.d_audio, generator=g),
             "text": f"sound number {i}", "task": "sound"} for i in range(sum(sizes))]
    paths, at = [], 0
    for s, n in enumerate(sizes):
        p = os.path.join(d, f"shard-{s:03d}.pt")
        write_frame_shard(p, recs[at:at + n], half=True)
        paths.append(p); at += n
    starts = cumulative_starts(sizes)
    assert starts == [0, 7, 12, 21]
    # Override the FIRST clip of each shard by its true global index; with uniform-512-style
    # starts these land inside the wrong shard.
    override = {0: "override A", 7: "override B", 12: "override C", 21: "override D"}
    ds = ShardedFrameDataset(paths, starts, shuffle_buffer=2, seed=3,
                             captions_override=override)
    seen = {it["text"] for it in ds}
    assert {"override A", "override B", "override C", "override D"} <= seen
    assert not {"sound number 0", "sound number 7", "sound number 12", "sound number 21"} & seen
    clips = load_frame_clips(paths, starts, [7, 21], captions_override=override)
    assert [c["text"] for c in clips] == ["override B", "override D"]


def test_load_recaption_splits_skip_and_offsets(tmp_path):
    import json
    rp = tmp_path / "src.json"
    rp.write_text(json.dumps({"0": "a horn honks", "1": "SKIP", "2": " skip ",
                              "3": "wind blows through trees"}), encoding="utf-8")
    override, skip = load_recaption(rp, src_base=100)
    assert override == {100: "a horn honks", 103: "wind blows through trees"}
    assert skip == {101, 102}                                # SKIP matching is trim+case-insensitive


def test_sharded_dataset_reiterates_full_epochs_and_collates():
    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    _write_shards(cfg, n=20, shard_size=8, dir_=d)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("shard-"))
    starts = shard_starts_from(len(paths), shard_size=8, n_total=20)
    ds = ShardedFrameDataset(paths, starts, shuffle_buffer=4, seed=0)
    a = [it["text"] for it in ds]
    b = [it["text"] for it in ds]                                        # re-iterate = fresh epoch
    assert set(a) == set(b) and len(a) == len(b) == 20
    loader = DataLoader(ds, batch_size=6, collate_fn=_collator(cfg), drop_last=True)
    batch = next(iter(loader))
    assert batch["frames"].shape[0] == 6 and batch["frames"].shape[-1] == cfg.d_audio
    assert batch["frame_mask"].shape == batch["frames"].shape[:2]


# ------------------------- Step 2: precomputed text cache ------------------------- #
def _write_text_caches(cfg, paths, seed=7):
    """Write a sibling text-emb shard for each frame shard; return {global_pos: emb} for checking."""
    g = torch.Generator().manual_seed(seed)
    embs_by_pos, base = {}, 0
    for p in paths:
        shard = torch.load(p, map_location="cpu", weights_only=False)
        n = len(shard["text"])
        embs = torch.randn(n, cfg.d_llm, generator=g)
        write_text_emb_shard(p, embs)
        for off in range(n):
            embs_by_pos[base + off] = embs[off]
        base += n
    return embs_by_pos


def test_text_cache_sibling_path_and_roundtrip():
    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    _, paths = _write_shards(cfg, n=20, shard_size=8, dir_=d)
    assert text_emb_shard_path(paths[0]).endswith(".txtemb.pt")     # sibling naming
    embs_by_pos = _write_text_caches(cfg, paths)
    starts = shard_starts_from(len(paths), shard_size=8, n_total=20)
    got = load_frame_clips(paths, starts, [0, 9, 19], with_text_emb=True)
    assert all("text_emb" in g for g in got)
    for rec, gi in zip(got, [0, 9, 19]):
        assert rec["text_emb"].shape == (cfg.d_llm,)
        assert torch.allclose(rec["text_emb"], embs_by_pos[gi], atol=1e-3)   # fp16 store → loose tol


def test_sharded_dataset_streams_text_emb_and_collates():
    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    _, paths = _write_shards(cfg, n=20, shard_size=8, dir_=d)
    _write_text_caches(cfg, paths)
    starts = shard_starts_from(len(paths), shard_size=8, n_total=20)
    ds = ShardedFrameDataset(paths, starts, shuffle_buffer=4, seed=0, use_text_emb=True)
    items = list(ds)
    assert len(items) == 20 and all(it["text_emb"].shape == (cfg.d_llm,) for it in items)
    batch = FrameCollator(cfg, HashingTokenizer(vocab=TINY_VOCAB, pad_id=cfg.pad_id,
                          audio_pad_id=cfg.audio_pad_id, eos_id=cfg.eos_id))([items[i] for i in range(6)])
    assert batch["text_emb_cached"].shape == (6, cfg.d_llm)          # collator stacks the cache


def test_missing_text_cache_raises():
    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    _, paths = _write_shards(cfg, n=10, shard_size=8, dir_=d)         # NO text caches written
    starts = shard_starts_from(len(paths), shard_size=8, n_total=10)
    import pytest
    with pytest.raises(FileNotFoundError):
        load_frame_clips(paths, starts, [0], with_text_emb=True)


def test_sharded_multiworker_loader_covers_every_clip_once():
    """Regression for the 2026-07-02 probe crash: the prod loader runs num_workers>0 with the
    'file_system' sharing strategy (modal_app sets it to dodge Modal's tiny /dev/shm). This drives
    that exact multi-worker path — shards partition disjointly across workers, so every clip must
    appear EXACTLY once (no worker overlap, no drops) and the cached text-emb must survive the
    worker->main-process transfer intact."""
    import torch.multiprocessing as _mp
    _mp.set_sharing_strategy("file_system")                          # mirror prod (modal_app.py)

    cfg = FusionConfig.tiny()
    d = tempfile.mkdtemp()
    _, paths = _write_shards(cfg, n=40, shard_size=8, dir_=d)        # 5 shards -> splits across 2 workers
    _write_text_caches(cfg, paths)
    starts = shard_starts_from(len(paths), shard_size=8, n_total=40)
    ds = ShardedFrameDataset(paths, starts, shuffle_buffer=4, seed=0, use_text_emb=True)
    loader = DataLoader(ds, batch_size=4, collate_fn=_collator(cfg), num_workers=2,
                        persistent_workers=True, prefetch_factor=2, drop_last=False)

    seen = []
    for batch in loader:
        assert batch["text_emb_cached"].shape == (batch["frames"].shape[0], cfg.d_llm)
        seen += batch["texts"]
    assert sorted(seen) == sorted(f"sound number {i}" for i in range(40))    # each clip exactly once


def test_cached_text_matches_live_text():
    """THE correctness gate: forward with the RAW text cache == forward re-encoding text live
    (both apply the SAME whitening to the SAME raw pooled vectors)."""
    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg).eval()
    tok = HashingTokenizer(vocab=TINY_VOCAB, pad_id=cfg.pad_id, audio_pad_id=cfg.audio_pad_id, eos_id=cfg.eos_id)
    collator = FrameCollator(cfg, tok)
    recs = [{"frames": torch.randn(4 + i, cfg.d_audio), "text": f"sound number {i}", "task": "sound",
             "instruction": "describe the sound"} for i in range(5)]

    live = collator(recs)                                            # no text_emb -> encodes live
    raw = model.encode_text(live["text_input_ids"], live["text_attention_mask"])  # what the cache stores
    model.text_whitening.fit(raw)                                    # non-trivial whitening, both paths share it

    cached = collator([{**r, "text_emb": raw[i].detach()} for i, r in enumerate(recs)])
    assert cached["text_emb_cached"].shape == (5, cfg.d_llm)
    out_live = model(live)
    out_cache = model(cached)
    assert torch.allclose(out_live["text"], out_cache["text"], atol=1e-5)   # cache == live
    # and the cached path did NOT need text tokens to be correct
    assert torch.allclose(out_cache["text"], model.text_whitening(raw), atol=1e-5)


def test_fit_whitening_from_cache_matches_direct_fit():
    from fusion_embedding.train_stage1 import fit_text_whitening_from_cache
    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg)
    embs = torch.randn(200, cfg.d_llm)
    # a loader-like iterable of batches carrying the cache
    batches = [{"text_emb_cached": embs[i:i + 40]} for i in range(0, 200, 40)]
    fit_text_whitening_from_cache(model, batches, device="cpu", max_samples=200)
    m1, s1 = model.text_whitening.mean.clone(), model.text_whitening.std.clone()
    model2 = build_tiny_model(cfg)
    model2.text_whitening.fit(embs)                                  # direct fit on the same sample
    assert torch.allclose(m1, model2.text_whitening.mean, atol=1e-5)
    assert torch.allclose(s1, model2.text_whitening.std, atol=1e-5)


def test_train_step_with_text_cache_updates_only_connector():
    from fusion_embedding.losses import FusionContrastiveLoss
    from fusion_embedding.train_stage1 import build_optimizer
    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg)
    tok = HashingTokenizer(vocab=TINY_VOCAB, pad_id=cfg.pad_id, audio_pad_id=cfg.audio_pad_id, eos_id=cfg.eos_id)
    collator = FrameCollator(cfg, tok)
    recs = [{"frames": torch.randn(5 + i, cfg.d_audio), "text": f"clip {i}", "task": "sound",
             "instruction": "describe the sound", "text_emb": torch.randn(cfg.d_llm)} for i in range(6)]
    batch = collator(recs)
    assert "text_emb_cached" in batch
    opt = build_optimizer(model, cfg)
    before = [p.detach().clone() for p in model.resampler.parameters()]
    out = model(batch)
    loss, _ = FusionContrastiveLoss(cfg)(out["audio"], out["text"], out["logit_scale"])
    loss.backward(); opt.step()
    assert any(not torch.equal(b, a) for b, a in zip(before, model.resampler.parameters()))
    for comp in model.frozen_modules():                              # base untouched
        for p in comp.parameters():
            assert p.grad is None


def test_multi_source_concat_with_partial_shards():
    """Two sources with PARTIAL last shards concatenated: global-index exclusion stays correct."""
    cfg = FusionConfig.tiny()
    dA, dB = tempfile.mkdtemp(), tempfile.mkdtemp()
    _write_shards(cfg, n=7, shard_size=5, dir_=dA, seed=1)               # sizes [5, 2] -> partial
    _write_shards(cfg, n=6, shard_size=5, dir_=dB, seed=2)               # sizes [5, 1] -> partial
    pathsA = sorted(os.path.join(dA, f) for f in os.listdir(dA) if f.startswith("shard-"))
    pathsB = sorted(os.path.join(dB, f) for f in os.listdir(dB) if f.startswith("shard-"))
    # concat with a running global offset (what _train_frames_impl does)
    paths = pathsA + pathsB
    starts, running, capsA, capsB = [], 0, [], []
    for src_paths, n, caps in ((pathsA, 7, capsA), (pathsB, 6, capsB)):
        for p, sp in enumerate(src_paths):
            starts.append(running)
            cnt = 5 if p < len(src_paths) - 1 else (n - 5 * (len(src_paths) - 1))
            running += cnt
    # A's captions are "sound number 0..6", B's are the SAME strings (seed differs, text same) — so
    # verify by global-index identity instead: exclude two clips, one from each source's partial shard.
    excl = {6, 12}                                                       # A's last clip (idx6), B's last (idx 7+5=12)
    ds = ShardedFrameDataset(paths, starts, exclude=excl, shuffle_buffer=2, seed=3)
    n_yield = sum(1 for _ in ds)
    assert n_yield == (7 + 6) - 2                                        # every non-excluded clip once
    # load_frame_clips round-trips the excluded global indices to real clips
    got = load_frame_clips(paths, starts, [6, 12])
    assert len(got) == 2 and all("frames" in g for g in got)
