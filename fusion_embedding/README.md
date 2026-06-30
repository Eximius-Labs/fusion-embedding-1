# `fusion_embedding/` — Fusion Embedding 1 scaffold

Graft an audio pathway onto a **frozen** Qwen3-VL-Embedding base by training only a
small connector. Full design: [`../docs/master_hld.md`](../docs/master_hld.md).

## File map

| file               | role                                                                                         |
| ------------------ | -------------------------------------------------------------------------------------------- |
| `config.py`        | `FusionConfig` — every confirmed dim/hyperparam (HLD §3); `.tiny()` for CPU tests             |
| `model.py`         | `FusionResampler` (trained connector) + `FusionEmbeddingModel` (audio injection, EOS pooling, MRL, freeze) |
| `losses.py`        | `FusionContrastiveLoss` — symmetric InfoNCE over the MRL ladder + CORAL (+ debias / hard-neg knobs) |
| `data.py`          | instruction taxonomy (HLD §6) + `FusionAudioTextManifest` + `FusionCollator` + synthetic backend |
| `train_stage1.py`  | P1 connector-only loop, optimizer/schedule, retrieval eval, **regression guard**, `load_components` seam |
| `memory_bank.py`   | `TextMemoryBank` + `precompute_text_bank` — frozen-text negatives so small (8GB) micro-batches still get many InfoNCE negatives |
| `_tiny.py`         | tiny frozen stand-ins for the Qwen towers (CPU, dependency-free) — exercise the real code paths |
| `demo_stage1.py`   | `python -m fusion_embedding.demo_stage1` — the whole P1 loop end to end                       |

## The frozen-base contract (the HLD §10 seams)

`FusionEmbeddingModel` takes the frozen base as three injected callables. Wiring the
real Qwen models means satisfying this same contract in `train_stage1.load_components`:

```
embed_tokens : nn.Module   ids [B,S]                      -> embeds [B,S,d_llm]
base_lm      : callable     (inputs_embeds, attention_mask) -> hidden [B,S,d_llm]
audio_encoder: callable     (mel, mel_mask)                 -> (frames [B,T,d_audio], frame_mask [B,T])
```

`_tiny.build_tiny_components` implements this contract with random CPU modules, so the
entire pipeline — injection → EOS pooling → MRL → InfoNCE/CORAL → optimizer step → eval
— is testable without GPUs, `transformers`, or model downloads.

## Run it

```bash
pip install -e .            # torch + numpy
python -m fusion_embedding.demo_stage1
pytest                      # 56 staged E2E tests (see ../tests)
```

For the real towers: `pip install -e ".[hf]"` and implement the `load_components`
seam (the `# TODO(fusion)` markers).

## Small-VRAM training (8GB) — the frozen-text memory bank

InfoNCE wants many negatives *in one forward batch*; grad-accumulation doesn't add any.
On an 8GB card the micro-batch is 1–2, which would starve the contrastive signal. But
the **text tower is frozen**, so its embeddings never drift — a bank of past/precomputed
text embeddings is a set of exact, zero-staleness negatives (no MoCo momentum needed).
Feed it to the A→T denominator and a micro-batch of 1 sees thousands of real negatives:

```python
from fusion_embedding.memory_bank import TextMemoryBank
bank = TextMemoryBank(dim=cfg.d_llm, capacity=16384)
train_stage1(model, loader, loss_fn, cfg, steps=..., memory_bank=bank)  # enqueues after each step
```

`tests/test_memory_bank.py` proves it: at micro-batch 2 the bank retrieves at least as
well as (and meaningfully better than) the in-batch-only baseline.

## P1 exit gate (HLD §7)

The E2E test (`tests/test_e2e.py`) asserts the connector learns to place audio in the
shared space (retrieval R@1 climbs from the random floor toward ~1.0), the loss drops,
and **the frozen base does not move by a single bit** (`base_drift == 0`) — the
param-level form of the "MMEB-V2 unchanged" regression guard.
