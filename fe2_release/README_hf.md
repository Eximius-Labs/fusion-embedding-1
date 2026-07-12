---
license: cc-by-nc-4.0
language:
  - en
pipeline_tag: feature-extraction
tags:
  - embeddings
  - multimodal
  - audio
  - retrieval
  - matryoshka
  - qwen3-vl
  - adapters
base_model: Qwen/Qwen3-VL-Embedding-2B
---

# fusion-embedding-2-2b-preview

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://github.com/Eximius-Labs/fusion-embedding)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://github.com/Eximius-Labs/fusion-embedding)
[![Weights](https://img.shields.io/badge/weights-CC--BY--NC--4.0-green.svg)](#license)
[![Status](https://img.shields.io/badge/status-research%20preview%20v0.1-orange.svg)](#)
[![Code](https://img.shields.io/badge/code-GitHub-black.svg)](https://github.com/Eximius-Labs/fusion-embedding)

</div>

`fusion-embedding-2-2b-preview` is the second generation of Eximius Labs' unified
multimodal embedding models: **text, images, video, and audio in one vector space**.
It extends the first generation with modality-gated deep adapters — in-layer audio
capacity added to a byte-frozen base. For the first-generation architecture, see
[fusion-embedding-1-2b-preview](https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview)
(that line is final at v0.3).

[GitHub](https://github.com/Eximius-Labs/fusion-embedding) | [fusion-embedding-1](https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview) | Technical report: in preparation

## Model Overview

<p align="center">
<img src="assets/fe2_model_overview.png" alt="fusion-embedding-2 architecture: frozen Qwen3-VL-Embedding base with modality-gated adapters inside; frozen audio tower and trained FusionResampler on the audio branch; one shared embedding space" width="820px">
</p>

`fusion-embedding-2-2b-preview` embeds all four modalities with a
[Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) base that is
**byte-identical to its original release** — its text, image, and video behaviour (and
benchmark scores) carry over exactly. Audio is added by training 60.6M parameters
(~2.3% of the stack): a perceiver-resampler that translates frozen
[Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) audio-tower features into
the base's input space, and — new in this generation — **28 gated adapters** (44.2M)
that give the frozen language model in-layer capacity to process audio. The adapters
are active only while encoding audio; every other forward pass returns the frozen
layers' output untouched, so the invariance is bitwise, not approximate
(`base_drift == 0` is asserted on every training run, and this model reproduces the
base's text→image retrieval scores to four decimal places). Trained on 518K
audio–caption pairs with a full-corpus frozen-text negative bank, it leads every
unified embedding model we measured on audio↔text retrieval — ahead of ImageBind,
LanguageBind, and Gemini Embedding 2 in both directions — and improves on
fusion-embedding-1 v0.3 in 8 of 12 release-protocol cells, including every
recorded text→audio direction. Audio↔image alignment is emergent (zero
audio–image pairs in training).

| Feature | Value |
| --- | --- |
| Parameters | ~2.06B frozen base + 640M frozen audio tower; **60.6M trained** |
| Modalities | text, image, video, audio |
| Supported tasks | `retrieval` (all modality pairs), `zero-shot classification` |
| Max input | 254 text tokens · 30 s audio per window (up to 8 windows) |
| Embedding dimension | 2048 |
| Matryoshka dimensions | 64, 128, 256, 512, 1024, 1536, 2048 |
| Pooling strategy | Last-token pooling |
| Base model | Qwen/Qwen3-VL-Embedding-2B (byte-frozen) |
| Audio tower | Qwen/Qwen2.5-Omni-7B audio encoder (frozen) |
| Trained components | FusionResampler 16.4M + 28× gated adapters 44.2M |
| Distribution | ~250 MB trained components; frozen towers download from their original repos |

## Training and Evaluation

Contrastive training (InfoNCE over the Matryoshka ladder, symmetric) against the
frozen base's native chat-template text embeddings: 518,183 audio–caption pairs from
six sources (73,716 clips with content-free metadata excluded), a full-corpus
frozen-text negative bank, soft labels 0.3, false-negative masking 0.98, bf16, 3,900
steps at effective batch 1,024, then a 400-step in-domain fine-tune on the AudioCaps
train split. All evaluation-set audio (Clotho, ESC-50, UrbanSound8K, VGGSound,
AudioCaps test/val) is excluded from training by ID blacklists at ingestion. A
technical report is in preparation.

All numbers below use the release protocol (bf16 base precision, native chat-template
text). Bold marks the better value per row/column.

<p align="center">
<img src="assets/fe_positioning.png" alt="Positioning: VGGSound-696 cross-modal retrieval versus model parameters; the fusion-embedding family leads unified models on audio-text and leads the emergent audio-image cluster (ImageBind's supervised pair annotated)" width="860px">
</p>

<details open>
  <summary><b>Versus fusion-embedding-1 v0.3</b></summary>

| Board / direction | fusion-embedding-1 v0.3 | fusion-embedding-2 (this repo) |
|---|---|---|
| AudioCaps A→T R@1 | **0.332** | 0.302 |
| AudioCaps A→T R@10 | 0.741 | **0.743** |
| AudioCaps T→A R@1 | — | **0.292** |
| AudioCaps T→A R@10 | 0.746 | **0.775** |
| Clotho (zero-shot) A→T R@1 | **0.135** | 0.127 |
| Clotho (zero-shot) A→T R@10 | **0.433** | 0.421 |
| Clotho (zero-shot) T→A R@1 | 0.136 | **0.151** |
| Clotho (zero-shot) T→A R@10 | 0.460 | **0.482** |
| VGGSound audio→text R@1 | **0.213** | 0.211 |
| VGGSound audio→text R@10 | 0.625 | **0.665** |
| VGGSound text→audio R@1 | 0.213 | **0.266** |
| VGGSound text→audio R@10 | 0.645 | **0.681** |
| VGGSound audio→image R@10 (emergent) | **0.407** | 0.392 |

fusion-embedding-2 takes the majority of cells, with its largest gains in the
text→audio direction (searching audio with a text query) and on the cross-modal
audio↔text pair. fusion-embedding-1 v0.3 retains the AudioCaps and Clotho A→T R@1
cells and a ~1.5-point edge on emergent audio→image at this fine-tuned operating
point; the pre-fine-tune fusion-embedding-2 checkpoint scores 0.443 on that cell — the
project record — and may be released separately as the emergent-alignment operating
point.

</details>

<details>
  <summary><b>Cross-modal retrieval — versus unified embedding models</b> (VGGSound-AV, 696 pairs, chance R@10 = 0.014)</summary>

R@10 shown as audio-side → other / other → audio-side:

| Model | audio↔image | audio↔text | text↔image |
|---|---|---|---|
| ImageBind-Huge | **0.718 / 0.720** | 0.404 / 0.348 | 0.243 / 0.282 |
| LanguageBind | 0.365 / 0.415 | 0.547 / 0.331 | 0.221 / 0.283 |
| Gemini Embedding 2 (API, 2026-07-09) | 0.312 / 0.316 | 0.379 / 0.374 | 0.273 / **0.366** |
| fusion-embedding-1-2b-preview v0.3 | 0.407 / 0.428 | 0.625 / 0.645 | **0.331** / 0.319 |
| **fusion-embedding-2-2b-preview** | 0.392 / 0.430 | **0.665 / 0.681** | **0.331** / 0.319 |

ImageBind trains directly on audio–image pairs, so that pair is its supervised
direction; its audio–text alignment is emergent. LanguageBind trains audio against
language; its audio↔image is emergent. Both fusion-embedding generations train on
audio–text only; their audio–image alignment is emergent. All models evaluated with
identical clips, frames, and scoring, using the released imagebind_huge checkpoint and
revision-pinned LanguageBind checkpoints. Gemini Embedding 2 is Google's natively
multimodal embedding API, evaluated at its documented default invocation on the date
shown; API models may change after that date. fusion-embedding-2's text↔image cells
are identical to fusion-embedding-1's by construction — text and images never touch
the trained components — and this is verified: its own readout run reproduces
fusion-embedding-1 v0.3's text→image scores to four decimal places.

</details>

<details>
  <summary><b>Audio–text retrieval — versus specialist CLAP models</b></summary>

Specialist CLAP models fine-tune their text towers on audio captions — the direct
trade this architecture declines in order to keep one shared space for all four
modalities. They remain ahead on the audio-caption boards (e.g., AudioCaps T→A R@1:
M2D-CLAP 41.4 vs 29.2 here); this model family is the strongest option we measured
when one model must serve text, images, video, and audio together. See the
[fusion-embedding-1 card](https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview)
for the full CLAP comparison tables; fusion-embedding-2 improves on fusion-embedding-1
in the text→audio direction on every board.

</details>

## Usage

<details>
  <summary>Requirements</summary>

- `fusion_embedding` package: `pip install git+https://github.com/Eximius-Labs/fusion-embedding`
- `transformers>=4.46`, `torch` (CUDA), `torchvision`, `pillow`, `soundfile`, `librosa`
- ~14 GB GPU memory at bf16

</details>

<details open>
  <summary>via <code>inference.py</code> (this repository)</summary>

```python
from inference import FusionEmbedder

fe = FusionEmbedder.from_pretrained(
    "EximiusLabs/fusion-embedding-2-2b-preview",
    revision="v0.1-preview",   # pin a tag if you build on this model
)

a = fe.embed_audio("dog.wav")            # audio file or (array, sr=...)
t = fe.embed_text("a dog barks")         # uses the base's native chat template
i = fe.embed_image("dog.jpg")            # PIL image or path

print((a @ t).item(), (a @ i).item())    # cosine similarities in the shared space

# Matryoshka: pass dim= for smaller embeddings (64..2048)
t_small = fe.embed_text("a dog barks", dim=256)
```

The checkpoint contains the gated adapters and the loader refuses to run without them —
an adapter checkpoint can never be silently executed as the first-generation
architecture. All inputs use the base model's chat-template format; embedding quality
is sensitive to this formatting, so use the templates provided by `FusionEmbedder`
rather than constructing your own.

</details>

<details>
  <summary>Cross-modal ranking tip</summary>

When ranking a gallery of one modality against queries of another, per-modality
mean-centering of the gallery improves cross-modal recall by roughly two points across
modality pairs:

```python
gallery = FusionEmbedder.center(gallery_embeddings)
```

</details>

## License

Code is Apache-2.0 ([GitHub](https://github.com/Eximius-Labs/fusion-embedding));
model weights in this repository are **CC BY-NC 4.0** (research preview). The frozen
base and audio tower retain their original licenses.

## Citation

```bibtex
@software{fusion_embedding_2_2026,
  title  = {Fusion Embedding 2: Modality-Gated Deep Adapters for a
            Unified Text, Image, Video, and Audio Embedding Space},
  author = {Tonmoy, Abdul Basit},
  year   = {2026},
  url    = {https://huggingface.co/EximiusLabs/fusion-embedding-2-2b-preview}
}
```
