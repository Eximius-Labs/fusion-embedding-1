"""Fusion Embedding 1 — graft an audio pathway onto a frozen Qwen3-VL-Embedding base.

Public API is intentionally small; see ``master_hld.md`` for the design.
Top-level names are resolved lazily so importing one submodule never forces the
others to import (keeps ``config`` usable before ``model``/``losses`` exist, and
avoids importing torch when only the config is needed).
"""

from __future__ import annotations

__all__ = [
    "FusionConfig",
    "FusionContrastiveLoss",
    "FusionEmbeddingModel",
    "FusionResampler",
    "TextMemoryBank",
    "precompute_text_bank",
]

__version__ = "0.1.0"

_LAZY = {
    "FusionConfig": ("fusion_embedding.config", "FusionConfig"),
    "FusionContrastiveLoss": ("fusion_embedding.losses", "FusionContrastiveLoss"),
    "FusionEmbeddingModel": ("fusion_embedding.model", "FusionEmbeddingModel"),
    "FusionResampler": ("fusion_embedding.model", "FusionResampler"),
    "TextMemoryBank": ("fusion_embedding.memory_bank", "TextMemoryBank"),
    "precompute_text_bank": ("fusion_embedding.memory_bank", "precompute_text_bank"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
