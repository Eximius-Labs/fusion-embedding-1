"""ShardedFrameDataset max_frames random crop (long-clip corpora I/O + CLAP-standard windows)."""

import tempfile

import torch

from fusion_embedding.data import ShardedFrameDataset, write_frame_shard


def _mk_shard(path, n=8, t_long=600, d=16):
    recs = [{"frames": torch.randn(t_long if i % 2 == 0 else 100, d),
             "text": f"cap {i}", "task": "sound"} for i in range(n)]
    write_frame_shard(path, recs, half=True)
    return recs


def test_max_frames_crops_long_clips_only():
    with tempfile.TemporaryDirectory() as td:
        p = f"{td}/shard-0000.pt"
        _mk_shard(p)
        ds = ShardedFrameDataset([p], [0], shuffle_buffer=1, max_frames=250, seed=0)
        recs = list(ds)
        assert len(recs) == 8
        for r in recs:
            assert r["frames"].shape[0] <= 250                  # long clips cropped
        assert any(r["frames"].shape[0] == 100 for r in recs)   # short clips untouched
        assert any(r["frames"].shape[0] == 250 for r in recs)   # crops hit the cap exactly


def test_max_frames_zero_disables_crop():
    with tempfile.TemporaryDirectory() as td:
        p = f"{td}/shard-0000.pt"
        _mk_shard(p)
        ds = ShardedFrameDataset([p], [0], shuffle_buffer=1, max_frames=0, seed=0)
        assert max(r["frames"].shape[0] for r in ds) == 600     # untouched


def test_crop_start_varies_across_epochs():
    with tempfile.TemporaryDirectory() as td:
        p = f"{td}/shard-0000.pt"
        base = _mk_shard(p, n=2, t_long=600)
        ds = ShardedFrameDataset([p], [0], shuffle_buffer=1, max_frames=250, seed=0)
        first = [r["frames"] for r in ds if r["frames"].shape[0] == 250]
        second = [r["frames"] for r in ds if r["frames"].shape[0] == 250]
        # same clip, different epoch -> (with overwhelming probability) different crop window
        assert not torch.equal(first[0], second[0])
