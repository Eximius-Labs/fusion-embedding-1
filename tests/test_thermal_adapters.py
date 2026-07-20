"""Multi-gate adapter registry (P0b thermal pack, docs/sensor_extension_plan.md §3).

The thermal probe adds a SECOND independently-gated adapter pack alongside the audio
pack. The load-bearing claims under test mirror the audio Stage-0 gate, per pack and
across packs:

1. with all gates CLOSED, a non-target (text / image-like) forward is BITWISE identical
   to the frozen base — a second pack does not perturb the preserved modalities;
2. a thermal forward CHANGES once the thermal pack's weights are nonzero;
3. gradients reach only the pack being trained — never the base, never the other pack;
4. the gate must be HELD OPEN through backward under gradient checkpointing (the
   recompute hazard) or the pack silently drops out of the gradient graph;
5. LEGACY ALIAS: the audio pack keeps the ``audio_adapters`` submodule name, so an
   audio-only state_dict still loads after a thermal pack is added.

Run on the tiny CPU stand-in (no GPU / transformers). ``AdapterPacks.add_pack`` reuses
the exact ``attach_gated_adapters`` primitive the audio path uses, so a green suite here
is evidence the released audio path is untouched.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import pytest

from fusion_embedding.adapters import AdapterPacks, find_decoder_layers
from fusion_embedding._tiny import build_tiny_components
from fusion_embedding.config import FusionConfig

RANK = 8


def _tiny_base():
    """A tiny frozen base_lm plus a token embedder, matching the audio-test harness."""
    cfg = FusionConfig.tiny()
    embed_tokens, base_lm, _ = build_tiny_components(cfg)
    for p in base_lm.parameters():
        p.requires_grad_(False)
    return cfg, embed_tokens, base_lm


def _embeds(cfg, embed_tokens, B=3, S=7, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 60, (B, S), generator=g)
    ids[:, -1] = cfg.eos_id
    return embed_tokens(ids), torch.ones(B, S, dtype=torch.long)


def _randomize_up(pack: nn.Module, seed=3):
    g = torch.Generator().manual_seed(seed)
    for m in pack.modules():
        if isinstance(m, nn.Linear) and m.weight.shape[0] >= m.weight.shape[1]:
            # the 'up' projections are the zero-inited ones (rank -> d_model)
            if torch.count_nonzero(m.weight) == 0:
                m.weight.data = torch.randn(m.weight.shape, generator=g) * 0.02


def test_all_gates_closed_is_bitwise_base():
    """Two packs (audio + thermal) attached; every gate closed -> base output exact."""
    cfg, embed_tokens, base_lm = _tiny_base()
    x, mask = _embeds(cfg, embed_tokens)
    ref = base_lm(inputs_embeds=x, attention_mask=mask)

    packs = AdapterPacks()
    packs.add_pack("audio", base_lm, cfg.d_llm, RANK)
    packs.add_pack("thermal", base_lm, cfg.d_llm, RANK)
    _randomize_up(packs, seed=7)                       # even with nonzero weights...
    out = base_lm(inputs_embeds=x, attention_mask=mask)   # ...all gates closed
    assert torch.equal(out, ref), "closed-gate forward must be bitwise-identical to base"


def test_thermal_forward_changes_when_open_and_nonzero():
    cfg, embed_tokens, base_lm = _tiny_base()
    x, mask = _embeds(cfg, embed_tokens)
    packs = AdapterPacks()
    packs.add_pack("thermal", base_lm, cfg.d_llm, RANK)
    ref = base_lm(inputs_embeds=x, attention_mask=mask)   # zero-init identity
    with packs.scope("thermal"):
        assert torch.equal(base_lm(inputs_embeds=x, attention_mask=mask), ref), \
            "zero-init thermal pack is the identity even with the gate open"
    _randomize_up(packs, seed=5)
    with packs.scope("thermal"):
        changed = base_lm(inputs_embeds=x, attention_mask=mask)
    assert not torch.equal(changed, ref), "nonzero thermal pack must change the output"


def test_gradient_isolation_thermal_only():
    """Training the thermal pack must not touch the base or the audio pack."""
    cfg, embed_tokens, base_lm = _tiny_base()
    x, mask = _embeds(cfg, embed_tokens)
    packs = AdapterPacks()
    packs.add_pack("audio", base_lm, cfg.d_llm, RANK)
    packs.add_pack("thermal", base_lm, cfg.d_llm, RANK)
    _randomize_up(packs, seed=2)
    with packs.scope("thermal"):
        out = base_lm(inputs_embeds=x, attention_mask=mask)
    out.pow(2).mean().backward()
    base_grad = any(p.grad is not None for p in base_lm.parameters())
    audio_grad = any(p.grad is not None for p in packs.parameters_of("audio"))
    thermal_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in packs.parameters_of("thermal"))
    assert not base_grad, "base params must have no gradient"
    assert not audio_grad, "the non-active audio pack must have no gradient"
    assert thermal_grad, "the active thermal pack must receive gradient"


def test_gate_must_span_backward_under_checkpointing():
    """If the gate closes before backward, checkpoint recompute drops the pack and the
    autograd graph is inconsistent -> the run must fail loudly, not silently mis-train."""
    cfg, embed_tokens, base_lm = _tiny_base()
    x, mask = _embeds(cfg, embed_tokens)
    packs = AdapterPacks()
    packs.add_pack("thermal", base_lm, cfg.d_llm, RANK)
    _randomize_up(packs, seed=1)
    layer = find_decoder_layers(base_lm)[0]

    gate = packs.gate("thermal")
    gate.__enter__()                                   # open for forward only
    h = x.requires_grad_(True)
    y = cp.checkpoint(lambda z: layer(z), h, use_reentrant=False)
    gate.__exit__()                                    # WRONG: closed before backward
    with pytest.raises(Exception):
        y.pow(2).mean().backward()                     # recompute sees a closed gate


def test_legacy_audio_alias_loads():
    """An audio-only state_dict (key 'audio_adapters.*') still loads after a thermal
    pack is added -> released FE2 audio checkpoints are unaffected."""
    cfg, _, base_lm = _tiny_base()
    audio_only = AdapterPacks()
    audio_only.add_pack("audio", base_lm, cfg.d_llm, RANK)
    _randomize_up(audio_only, seed=9)
    sd = audio_only.state_dict()
    assert all(k.startswith("audio_adapters.") for k in sd), \
        f"audio pack must live under the legacy 'audio_adapters' key; got {list(sd)[:2]}"

    both = AdapterPacks()
    both.add_pack("audio", base_lm, cfg.d_llm, RANK)
    both.add_pack("thermal", base_lm, cfg.d_llm, RANK)
    missing, unexpected = both.load_state_dict(sd, strict=False)
    assert not unexpected, f"audio checkpoint had unexpected keys: {unexpected}"
    assert all(k.startswith("thermal_adapters.") for k in missing), \
        "only the fresh thermal pack should be missing from an audio-only checkpoint"
