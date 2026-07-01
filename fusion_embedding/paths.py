"""Portable storage layout — one env var decouples the whole pipeline from any provider.

Set ``FUSION_DATA_ROOT`` to point features/frames/checkpoints/HF-cache anywhere:
  * Modal:  a Volume mount, e.g. ``/vol``
  * RunPod/Lambda/bare-metal: a local dir or mounted disk, e.g. ``/workspace/fusion_data``
  * local dev: defaults to ``./fusion_data``

Nothing else in the codebase hardcodes a path, so moving off Modal is a config change,
not a rewrite. (For S3/GCS, point FUSION_DATA_ROOT at a mounted bucket, or swap these
``os``/``open`` calls for ``fsspec`` — the call sites are centralized here.)
"""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("FUSION_DATA_ROOT", str(Path.cwd() / "fusion_data")))


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def hf_cache_dir() -> Path:
    return _ensure(data_root() / "hf-cache")


def features_dir(shard: str) -> Path:
    """Precomputed mel features (audio decode paid once)."""
    return _ensure(data_root() / "features" / shard)


def frames_dir(shard: str) -> Path:
    """Precomputed frozen-audio-encoder frames (encoder run paid once — Option 2 speedup)."""
    return _ensure(data_root() / "frames" / shard)


def checkpoints_dir() -> Path:
    return _ensure(data_root() / "checkpoints")
