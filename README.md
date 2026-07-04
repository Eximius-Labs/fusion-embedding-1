<div align="center">

# Fusion Embedding

**One model. One vector space. Text, image, video, audio — and PDF.**

*An open-weight multimodal embedding model that grafts audio onto the strongest
open vision-language embedding base — without touching a single base weight.*

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](#)
[![License](https://img.shields.io/badge/license-Apache--2.0%20(target)-green.svg)](#license)
[![Status](https://img.shields.io/badge/status-research%20preview-orange.svg)](#roadmap)

</div>

---

## What is Fusion Embedding?

Fusion Embedding 1 is a family of open-weight embedding models that map **five
modalities into a single shared vector space**, built for retrieval, RAG,
clustering, and cross-modal search — and designed to be **fully self-hostable**.

Instead of training a multimodal embedder from scratch, Fusion takes
[Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) — an
open state-of-the-art text/image/video embedding model — **freezes it
byte-for-byte**, and grafts an audio pathway onto it: a frozen
[Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) audio tower feeds a
small trained connector (the **FusionResampler**, ~7M parameters, ≈0.36% of the
base) that translates audio into the base's input space. Audio is aligned to
text contrastively; because text, image, and video already share the base's
space, **audio↔image and audio↔video alignment emerge through the text bridge**
(the ImageBind property).

The result: the base model's text, image, and video embeddings are **provably
unchanged** — every training run asserts parameter-level `base_drift == 0` — so
you inherit the base's retrieval quality exactly, and add audio on top.

## Highlights

- **Five modalities, one space** — text, image, video, audio, and PDF embed
  into the same vector space; any-to-any retrieval works out of the box.
- **Frozen-base grafting** — only a ~7M-parameter connector and a temperature
  are trained. The base is never fine-tuned, so its MMEB-V2 performance is
  inherited *by construction*, not re-benchmarked and hoped for.
- **Matryoshka embeddings** — truncate to any rung of
  `{2048, 1536, 1024, 512, 256, 128, 64}` and re-normalize; embeddings stay
  consistent at every dimension (default 1024).
- **Instruction-aware audio** — the same clip embeds differently for
  different tasks (*sound description* vs *spoken content* vs *speaker/emotion*),
  matching the base's instruction conditioning.
- **Full speech ambition** — the audio tower is Whisper-large-v3-derived;
  speech content, language, and paralinguistics are first-class targets, not an
  afterthought.
- **Self-hostable** — no API, no rate limits; runs quantized on a single
  consumer GPU for inference, and the connector-only design makes training
  radically cheap.
- **Test-first engineering** — the entire pipeline runs end-to-end on tiny
  CPU stand-ins (no GPU, no `transformers`) via dependency injection at the
  model seams; 125+ unit and E2E tests.

## Architecture

```
                          ┌───────────────── FROZEN base (Qwen3-VL-Embedding) ─────────────────┐
text / image / video ───▶ │  (the base's own paths — untouched, byte-identical to the release) │
                          │                                                                    │
audio ─▶ [Qwen2.5-Omni    │                                                                    │
         audio encoder]   │                                                                    │
         (FROZEN)         │                                                                    │
         frames [B,T,3584]│                                                                    │
   └─▶ [FusionResampler] ─┼─▶ audio tokens [B,N,2048] ─▶ spliced at <|audio_pad|> positions ──▶│
        (TRAINED, ~7M)    │      ─▶ frozen LLM ─▶ EOS-token hidden state [B, 2048]             │
                          └───────────────────────────────────────────────────┬───────────────┘
                                                                              ▼
                                       MRL-truncate (any ladder rung) ─▶ L2-normalize ─▶ embedding
```

The **FusionResampler** is a Flamingo-style perceiver resampler running at a
256-d bottleneck: `in_proj 3584→256` → N=64 learnable latent queries through
L=6 pre-norm blocks (self-attention → cross-attention over audio frames → FFN)
→ `out_proj 256→2048`. Its N output tokens overwrite `<|audio_pad|>` placeholder
positions in the frozen LLM's input-embedding stream — the exact mirror of the
base's image-token mechanism. EOS pooling and the MRL ladder are the base's own;
**audio conforms to the base, never the reverse.**

A per-dimension **text whitening** module (diagonal, MRL-safe, fitted once from
the frozen text embeddings) corrects the anisotropy of decoder-LM embedding
spaces before the contrastive loss — a refinement uniquely available to
frozen-text architectures.

## Model family

| Model | Base | Params (trained / total) | Embedding dim | MRL ladder | Status |
|---|---|---|---|---|---|
| `fusion-embedding-1-2B` | Qwen3-VL-Embedding-2B | ~7M / 2B | 2048 | 2048 → 64 | in training (P1) |
| `fusion-embedding-1-8B` | Qwen3-VL-Embedding-8B | scaled / 8B | ~4096 | ~4096 → 64 | planned |

> **Research preview.** Weights are not yet released — the audio pathway is in
> active Stage-1 (audio↔text alignment) training. Benchmarks will be published
> with the first checkpoint release, evaluated on the standard protocols
> (AudioCaps / Clotho multi-reference retrieval, MAEB, MMEB-V2 regression).

## Usage

The inference API mirrors the training code in this repository. Once weights
are released, embedding audio and text looks like this:

```python
import torch
from fusion_embedding.config import FusionConfig
from fusion_embedding.model import FusionEmbeddingModel
from fusion_embedding.hf_components import load_components

# Load the frozen Qwen base + Omni audio tower + the trained connector
cfg = FusionConfig()
cfg, embed_tokens, base_lm, audio_encoder, tokenizer, feature_extractor = load_components(
    cfg, device="cuda", load_in_4bit=True,     # 4-bit base: inference fits consumer GPUs
)
model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder)
ckpt = torch.load("fusion-embedding-1-2b-p1.pt")
model.resampler.load_state_dict(ckpt["resampler"])
model.text_whitening.load_state_dict(ckpt["text_whitening"])

# --- embed an audio clip -----------------------------------------------------
mel = feature_extractor(wav, sampling_rate=16_000, return_tensors="pt")["input_features"]
audio_tok = model.audio_tokens(mel.cuda())                 # frozen tower -> resampler
pooled = model.encode_audio(audio_ids, audio_mask, audio_tok)
audio_emb = model.embed(pooled, dim=1024)                  # MRL-truncate + L2-normalize

# --- embed a query (instruction-conditioned) ---------------------------------
query = "Retrieve audio by sound description. A dog barks while rain falls."
ids = tokenizer(query, return_tensors="pt")
pooled_t = model.text_whitening(model.encode_text(ids["input_ids"].cuda(),
                                                  ids["attention_mask"].cuda()))
text_emb = model.embed(pooled_t, dim=1024)

score = (audio_emb @ text_emb.T)                           # cosine similarity
```

**Matryoshka truncation** — pick your speed/quality point at query time, no
re-encoding:

```python
emb_full  = model.embed(pooled, dim=2048)   # maximum quality
emb_fast  = model.embed(pooled, dim=256)    # 8× smaller index
emb_edge  = model.embed(pooled, dim=64)     # 32× smaller index
```

**Instruction taxonomy** — prefix the *query* side with the task instruction
(the audio side is always neutral):

| Task | Instruction |
|---|---|
| `sound` | Retrieve audio by sound description. |
| `speech_content` | Retrieve audio by spoken content. |
| `music` | Retrieve music by description. |
| `speech_language` | Retrieve speech by language. |
| `speech_paralinguistic` | Retrieve speech by speaker and emotion. |

## Training

Training is deliberately cheap: **both towers are frozen**, so every expensive
forward pass is paid once and cached.

```
audio ──▶ Whisper mel (once) ──▶ frozen audio tower (once) ──▶ frame shards
captions ──▶ frozen text tower (once) ──▶ text-embedding cache ─┐
                                                                ▼
        connector-only training: frames + cached text ─▶ InfoNCE over the MRL
        ladder + CORAL, with a full-corpus frozen-text negative bank
```

Key properties:

- **Frozen-text negative bank** — because the text tower never moves, cached
  text embeddings are exact forever: every training step scores audio against
  the *entire corpus* of captions as negatives, with zero staleness (the classic
  memory-bank failure mode doesn't exist here).
- **Symmetric InfoNCE over the MRL ladder** + learnable temperature + light
  CORAL covariance alignment; debiased-contrastive and hard-negative knobs for
  later stages.
- **Regression guard** — every run snapshots the base's parameters and asserts
  `base_drift == 0.0` at the end. If the base moved, the run fails.
- **Crash-safe by default** — atomic resume checkpoints every 100 steps
  (config-fingerprinted so A/B arms can never cross-resume), automatic retry on
  preemption, and a divergence guard that stops non-finite losses immediately.

The reference training stack runs on [Modal](https://modal.com) (L4 for
preprocessing, A100/H100 for training) with storage decoupled behind a single
`FUSION_DATA_ROOT` env var — a Modal Volume, local disk, or S3 bucket all work.

```bash
uv sync                                                        # install (uv + torch cu124)
uv run pytest tests/ -q                                        # 125+ tests, CPU-only, no GPU needed
uv run python -m fusion_embedding.demo_stage1                  # whole P1 loop on tiny CPU stand-ins

# real pipeline (Modal)
uv run --env-file .env modal run modal_app.py::preprocess           # audio -> mel
uv run --env-file .env modal run modal_app.py::precompute_frames    # frozen tower -> frame shards
uv run --env-file .env modal run modal_app.py::precompute_text_cache
uv run --env-file .env modal run --detach modal_app.py::train_frames_a100  # connector training
```

## Evaluation protocol

We evaluate on the standard published protocols so numbers are directly
comparable to prior audio-text and multimodal embedding work:

- **AudioCaps test** (multi-reference): A→T scored as min-rank over the 5
  ground-truth captions; R@1/5/10 and mAP@10, both directions.
- **Clotho v2 evaluation** (1045 clips × 5 references, from the canonical
  Zenodo release): strictly **zero-shot** — Clotho never appears in training.
- **MAEB** for breadth across sound, music, and multilingual speech.
- **MMEB-V2 regression**: the base's text/image/video scores must be unchanged —
  guaranteed mechanically by the frozen-base design and asserted every run.

Every training run auto-scores the comparable protocol at the end; eval sets
are blacklisted from all training data by clip id.

## Repository layout

```
fusion_embedding/
  config.py          FusionConfig — every locked dimension + hyperparameter
  model.py           FusionResampler, FusionEmbeddingModel, TextWhitening
  losses.py          FusionContrastiveLoss (InfoNCE × MRL + CORAL + debias/hard-neg)
  memory_bank.py     frozen-text negative banks (FIFO + full-corpus)
  data.py            instruction taxonomy, manifests, collators, sharded streaming
  train_stage1.py    P1 loop, retrieval metrics, whitening, resume ckpts, RegressionGuard
  hf_components.py   real frozen-Qwen wiring (base + Omni audio tower adapters)
  _tiny.py           tiny CPU stand-ins implementing the same three-callable contract
  demo_stage1.py     end-to-end P1 demo on the stand-ins
modal_app.py         the full cloud pipeline: ingestion, caching, training, scoring
tests/               unit + E2E suites — the whole pipeline with no GPU/transformers
```

The frozen base is injected as **three duck-typed callables** (`embed_tokens`,
`base_lm`, `audio_encoder`), so the identical model code runs against the real
Qwen towers or the tiny CPU stand-ins — that seam is what makes the pipeline
fully testable without hardware.

## Roadmap

- [x] **P0 — Infrastructure**: frozen-base wiring, eval harness, CPU-testable pipeline
- [x] **P1 (in progress) — Audio→text alignment**: connector training at scale;
      published-protocol eval wired into every run
- [ ] **P2 — Hardening**: clean data, mined hard negatives, full alignment suite
      (modality temperature, debiased contrastive)
- [ ] **P3 — Speech parity + grounding**: heavy multilingual speech, more query
      tokens, direct audio↔video pairs
- [ ] **P4 — Release**: pre-registered five-modality benchmark, model soup,
      2B release; then the 8B tier
- [ ] **Track C corpus**: self-generated, CLAP-gated captions on permissively
      licensed audio — the commercially clean training set

## License

Code: **Apache-2.0** (target). Model weights: the release tier will ship under
a permissive license pending a license audit of the frozen audio tower and the
training-data track used (research-posture vs commercially-clean corpora are
kept strictly separate).

## Acknowledgments

Fusion Embedding stands on outstanding open work:
[Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) (the
frozen base), [Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) (the
audio tower), the frozen-tower composition precedent of jina-embeddings-v5-omni,
the alignment recipe of e5-omni, ImageBind's emergent-alignment result,
Matryoshka Representation Learning, and the audio-caption data ecosystem
(WavCaps, AudioCaps, Clotho, FSD50K, and the AudioSetCaps pipeline).

## Citation

```bibtex
@software{fusion_embedding_2026,
  title  = {Fusion Embedding: Grafting Audio onto Frozen Vision-Language
            Embedding Models},
  author = {Tonmoy, Abdul Basit},
  year   = {2026},
  url    = {https://github.com/eximius-labs/fusion-embeddings}
}
```
