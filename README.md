<div align="center">

# Fusion Embedding

**One model. One vector space. Text, image, video, audio — and PDF.**

*An open-weight multimodal embedding model that extends a state-of-the-art
vision-language embedding base with audio — without modifying a single base weight.*

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
byte-for-byte**, and adds an audio pathway to it: a frozen
[Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) audio tower feeds a
small trained connector (the **FusionResampler**, ~16M parameters, <1% of the
base) that translates audio into the base's input space. Audio is aligned to
text contrastively; because text, image, and video already share the base's
space, **audio↔image and audio↔video alignment emerge through the text bridge**
(the ImageBind property).

The result: the base model's text, image, and video embeddings are **provably
unchanged** — every training run asserts parameter-level `base_drift == 0` — so
you inherit the base's retrieval quality exactly, and add audio on top.

## Highlights

- **Five modalities, one space** — text, image, video, audio, and PDF in a
  single vector space, designed for retrieval in any direction between modalities.
- **Frozen-base architecture** — only a ~16M-parameter connector and a temperature
  are trained. The base is never fine-tuned; its MMEB-V2 performance is
  inherited unchanged, by construction.
- **Matryoshka embeddings** — truncate to any rung of
  `{2048, 1536, 1024, 512, 256, 128, 64}` and re-normalize; embeddings stay
  consistent at every dimension (default 1024).
- **Instruction-aware audio** — the same clip embeds differently for
  different tasks (*sound description* vs *spoken content* vs *speaker/emotion*),
  matching the base's instruction conditioning.
- **Speech as a first-class target** — the audio tower is Whisper-large-v3-derived;
  speech content, language, and paralinguistics are on the roadmap alongside
  sound events and music.
- **Self-hostable** — no API, no rate limits; runs quantized on a single
  consumer GPU for inference, and the connector-only design keeps training
  costs low.
- **Test-first engineering** — the entire pipeline runs end-to-end on tiny
  CPU stand-ins (no GPU, no `transformers`) via dependency injection at the
  model seams; 125+ unit and E2E tests.

## What audio→image retrieval looks like

The connector is trained **only** on audio↔text pairs — it never sees a single
audio–image example. But because text, image, and video already share the frozen
base's space, audio lands there too, and **audio→image retrieval emerges for free**.
On VGGSound-696 (held out, in the training blacklist → zero leakage), the released
preview retrieves the matching image from sound alone at **R@10 0.368 — 26× the 0.014
random-chance rate**. Real examples:

**Direct hits** — the clip's own frame comes back in the top 5, surrounded by the same kind of scene:

| The sound | What it retrieved (top-5) | Exact frame |
|---|---|---|
| A siren | its own ambulance, then more emergency vehicles | rank 2 |
| *"Switch on the good piece"* (speech) | the appliance being switched on | rank 1 |
| A whirring kitchen motor | its own blender, among other mixers | rank 3 |
| An emergency-vehicle siren | fire engines and ambulances | rank 4 |
| Metallic clanking and banging | the kitchen it came from | rank 5 |

**Right neighbourhood** — the exact frame ranks lower (often it's a poor still), but every
top result is the correct *sound* category — the shared space placing audio among the right images:

| The sound | What it retrieved (top-5) | Exact frame |
|---|---|---|
| A distinct click | people typing at keyboards | rank 109 |
| A power motor | an angle grinder, a belt sander, a table saw | rank 68 |
| A single cowbell note | cattle herds — one wearing a bell | rank 6 |
| High squeaks and chirps | chickens, crows, a quail, a parrot | rank 104 |

Scored with the released `fusion-embedding-1-2b-preview` on `mteb/VGGSound_AV_RETRIEVAL`,
per-modality-centered geometry (see the model card for the full cross-modal table, including
an ImageBind comparison).

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
        (TRAINED, ~16M)   │      ─▶ frozen LLM ─▶ EOS-token hidden state [B, 2048]             │
                          └───────────────────────────────────────────────────┬───────────────┘
                                                                              ▼
                                       MRL-truncate (any ladder rung) ─▶ L2-normalize ─▶ embedding
```

The **FusionResampler** is a Flamingo-style perceiver resampler running at a
384-d bottleneck: `in_proj 3584→384` → N=64 learnable latent queries through
L=6 pre-norm blocks (self-attention → cross-attention over audio frames → FFN)
→ `out_proj 384→2048`. Its N output tokens overwrite `<|audio_pad|>` placeholder
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
| [`fusion-embedding-1-2b-preview`](https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview) | Qwen3-VL-Embedding-2B | ~16M / 2B | 2048 | 2048 → 64 | **preview released** |
| `fusion-embedding-1-8B` | Qwen3-VL-Embedding-8B | scaled / 8B | ~4096 | ~4096 → 64 | planned |

> **Research preview available:**
> [EximiusLabs/fusion-embedding-1-2b-preview](https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview)
> — connector weights, model card with benchmarks (AudioCaps / Clotho zero-shot /
> VGGSound cross-modal incl. an ImageBind comparison), and a packaged inference API.
> Training continues; updated checkpoints will be released under the same family.

## Usage

For the packaged high-level API, see `inference.py` on the
[HF model repo](https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview)
(`FusionEmbedder.embed_audio / embed_text / embed_image`). The low-level path
using this repository's components:

```python
import torch
from fusion_embedding.config import FusionConfig
from fusion_embedding.model import FusionEmbeddingModel
from fusion_embedding.hf_components import load_components

# Load the frozen Qwen base + Omni audio tower + the trained connector.
# NOTE: base precision must match the checkpoint's training precision (recorded in the
# checkpoint as `base_4bit`); released checkpoints are bf16-trained.
cfg = FusionConfig()
cfg, embed_tokens, base_lm, audio_encoder, tokenizer, feature_extractor = load_components(
    cfg, device="cuda", load_in_4bit=False,
)
model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder)
ckpt = torch.load("fusion-embedding-1-2b-preview.pt")
model.resampler.load_state_dict(ckpt["resampler"])
model.text_whitening.load_state_dict(ckpt["text_whitening"])

# --- embed an audio clip -----------------------------------------------------
mel = feature_extractor(wav, sampling_rate=16_000, return_tensors="pt")["input_features"]
audio_tok = model.audio_tokens(mel.cuda())                 # frozen tower -> resampler
pooled = model.encode_audio(audio_ids, audio_mask, audio_tok)
audio_emb = model.embed(pooled, dim=1024)                  # MRL-truncate + L2-normalize

# --- embed a query (the base model's chat-template format is REQUIRED) --------
query = ("<|im_start|>system\nRetrieve audio by sound description.<|im_end|>\n"
         "<|im_start|>user\nA dog barks while rain falls.<|im_end|>\n"
         "<|im_start|>assistant\n")
ids = tokenizer(query, return_tensors="pt", add_special_tokens=False)
pooled_t = model.text_whitening(model.encode_text(ids["input_ids"].cuda(),
                                                  ids["attention_mask"].cuda()))
text_emb = model.embed(pooled_t, dim=1024)

score = (audio_emb @ text_emb.T)                           # cosine similarity
```

For a packaged, higher-level API (`FusionEmbedder.embed_audio / embed_text / embed_image`),
see `release/inference.py` — it applies the correct templates and pooling automatically.

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

## Results — preview checkpoints

Numbers for [`fusion-embedding-1-2b-preview`](https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview)
— v0.1 (131K-pair corpus) and v0.2 (484K-pair corpus incl. the full AudioCaps train split
and a 318K LAION-FreeSound subset), same d=384 connector. Full tables and protocol details
are on the model card.

**Audio–text retrieval** (published protocols):

| Benchmark | Version | A→T R@1 | A→T R@10 | T→A R@10 |
|---|---|---|---|---|
| AudioCaps test (883 clips, 5-ref min-rank) | v0.1 | 0.216 | 0.626 | 0.680 |
| AudioCaps test | **v0.2** | **0.279** | **0.717** | **0.736** |
| Clotho v2.1 eval, zero-shot (1,045 × 5 refs) | v0.1 | 0.064 | 0.252 | 0.329 |
| Clotho v2.1 eval, zero-shot | **v0.2** | **0.135** | **0.448** | **0.449** |

CLAP-family models that fine-tune both encoders end-to-end score higher on AudioCaps
(A→T R@10 0.906–0.928); this model keeps both towers frozen.

**Cross-modal retrieval** (VGGSound-AV, 696 pairs, chance R@10 = 0.014; this model trains
on audio–text only — its audio↔image alignment is emergent):

| Model | audio↔image | audio↔text | text↔image |
|---|---|---|---|
| ImageBind-Huge | **0.718 / 0.720** | 0.404 / 0.348 | 0.243 / 0.282 |
| fusion-embedding-1-2b-preview v0.1 | 0.368 / 0.388 | 0.555 / 0.592 | 0.331 / 0.319 |
| **fusion-embedding-1-2b-preview v0.2** | 0.418 / 0.440 | **0.588 / 0.631** | **0.331 / 0.319** |

Full v0.2 audio→image metrics (per-modality mean-centered readout): R@1 0.088, R@5 0.315,
R@10 0.418 (29× chance), mAP@10 0.179 — with zero audio–image training pairs. What that
looks like (v0.1 examples; query clip's frame left; green = the clip's exact frame among
the top 5):

![Audio-to-image retrieval examples](assets/audio_to_image_gallery.png)

*Example frames from the [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) dataset (CC-BY-4.0), shown for evaluation illustration.*

Text, image, and video performance is the frozen base model's published MMEB-V2 results,
unchanged by construction.

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
  title  = {Fusion Embedding 1: A Unified Embedding Space for Text,
            Image, Video, and Audio},
  author = {Tonmoy, Abdul Basit},
  year   = {2026},
  url    = {https://github.com/Eximius-Labs/fusion-embedding-1}
}
```
