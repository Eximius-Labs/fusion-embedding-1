"""Modality-gated deep adapters — in-layer trainable capacity for the AUDIO path only.

The 2B gap analysis (docs/adapter_experiment_plan.md) points at one bottleneck: the
frozen base LM never learned to process audio features, and all audio understanding
must squeeze through the 16.4M input-side resampler. OEA (arXiv:2604.18360) shows the
capacity belongs IN the layers — but their LoRA touches every token, sacrificing
backbone byte-identity. This module adds per-decoder-layer bottleneck adapters that
are **hard-gated to audio forwards**:

* the gate is a depth counter held open by the caller (``FusionEmbeddingModel``
  scopes it around audio encodes; the trainer holds it across forward+backward so
  gradient-checkpoint recomputation sees the same gate state — see the warning on
  :class:`AdapterGate`);
* when the gate is closed the hook returns ``None`` (i.e. "keep the original
  output") **before any arithmetic**, so non-audio forwards are bitwise identical
  to the frozen base — not "low drift", identical;
* adapters are owned by :class:`FusionEmbeddingModel` (NOT registered under the
  base module), so ``RegressionGuard``'s frozen-parameter snapshot is untouched and
  ``base_drift == 0`` keeps meaning what it always meant;
* ``up`` is zero-initialised, so a freshly-built adapter stack is the exact
  identity: step-0 behaviour equals the current architecture, and warm-starting a
  pretrained resampler under fresh adapters is safe.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACTS = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}


class AdapterGate:
    """Depth-counted on/off switch shared by every hook.

    WARNING (gradient checkpointing): checkpointed layers re-run their forward DURING
    ``backward()``. If the gate has been closed by then, the recomputed graph silently
    drops the adapters and gradients are wrong. Therefore the gate must be held open
    across forward AND backward of any audio step trained with checkpointing — use
    ``FusionEmbeddingModel.adapter_scope()`` around the whole step. The depth counter
    makes nested scopes (trainer scope + ``encode_audio``'s own scope) safe.
    """

    __slots__ = ("_depth",)

    def __init__(self) -> None:
        self._depth = 0

    @property
    def active(self) -> bool:
        return self._depth > 0

    def __enter__(self) -> "AdapterGate":
        self._depth += 1
        return self

    def __exit__(self, *exc) -> None:
        self._depth -= 1
        if self._depth < 0:
            raise RuntimeError("AdapterGate depth underflow — unbalanced enter/exit")


class GatedAdapter(nn.Module):
    """Parallel bottleneck adapter: ``h + up(act(down(LN(h))))``, computed in fp32.

    ``up`` is zero-initialised => the module is the identity at init.
    """

    def __init__(self, d_model: int, rank: int, act: str = "silu"):
        super().__init__()
        if act not in _ACTS:
            raise ValueError(f"unknown adapter_act {act!r} (choose from {sorted(_ACTS)})")
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, rank, bias=False)
        self.act = _ACTS[act]()
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.up.weight)                     # identity at init

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # fp32 compute for stability (base emits bf16 at scale); cast back to blend
        # into the residual stream at the base's dtype.
        return self.up(self.act(self.down(self.norm(h.float())))).to(h.dtype)


def find_decoder_layers(base_lm: nn.Module) -> nn.ModuleList:
    """Locate the decoder-layer ModuleList inside an arbitrary base wrapper.

    Works for both production and test stand-ins: ``BaseLMAdapter`` wraps the HF Qwen3
    text model (…``.layers``) and ``TinyLM`` holds a ``TransformerEncoder`` whose
    stack is also named ``layers``. Picks the LONGEST ModuleList named ``layers``
    (the decoder stack) if several match.
    """
    if not isinstance(base_lm, nn.Module):
        raise TypeError(f"adapters need an nn.Module base, got {type(base_lm).__name__}")
    best: nn.ModuleList | None = None
    for name, mod in base_lm.named_modules():
        if isinstance(mod, nn.ModuleList) and name.rsplit(".", 1)[-1] == "layers":
            if best is None or len(mod) > len(best):
                best = mod
    if best is None or len(best) == 0:
        raise ValueError("no decoder ModuleList named 'layers' found in base_lm")
    return best


def _make_hook(adapter: GatedAdapter, gate: AdapterGate):
    def hook(_module, _inputs, output):
        if not gate.active:
            return None                                    # keep original output — bitwise no-op
        if isinstance(output, tuple):                      # HF decoder layers -> (hidden, ...)
            h = output[0]
            return (h + adapter(h),) + tuple(output[1:])
        return output + adapter(output)                    # plain-tensor layers (TinyLM)
    return hook


def attach_gated_adapters(base_lm: nn.Module, d_model: int, rank: int,
                          act: str = "silu") -> tuple[nn.ModuleList, AdapterGate, list]:
    """Build one adapter per decoder layer and register the gated forward hooks.

    Returns ``(adapters, gate, handles)``. The caller MUST register ``adapters`` on a
    module OUTSIDE the frozen base (``FusionEmbeddingModel.audio_adapters``) so the
    RegressionGuard snapshot and the base's ``state_dict`` stay adapter-free.
    """
    layers = find_decoder_layers(base_lm)
    gate = AdapterGate()
    adapters = nn.ModuleList([GatedAdapter(d_model, rank, act) for _ in layers])
    handles = [layer.register_forward_hook(_make_hook(ad, gate))
               for layer, ad in zip(layers, adapters)]
    return adapters, gate, handles
