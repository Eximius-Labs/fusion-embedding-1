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
base_model: Qwen/Qwen3-VL-Embedding-2B
---

![Fusion Embedding 1 — 2B Preview, by Eximius Labs](assets/banner.png)

<div align="center">

**One model. One vector space. Text, image, video, audio — and PDF.**

*An open-weight multimodal embedding model that extends a state-of-the-art
vision-language embedding base with audio — without modifying a single base weight.*

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://github.com/Eximius-Labs/fusion-embedding-1)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://github.com/Eximius-Labs/fusion-embedding-1)
[![Weights](https://img.shields.io/badge/weights-CC--BY--NC--4.0-green.svg)](#training-data-and-license)
[![Status](https://img.shields.io/badge/status-research%20preview%20v0.3-orange.svg)](#)
[![Code](https://img.shields.io/badge/code-GitHub-black.svg)](https://github.com/Eximius-Labs/fusion-embedding-1)

</div>

Fusion Embedding 1 extends [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
with an audio modality. A trained connector (~16M parameters) maps frozen
[Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) audio-tower features into the
base model's embedding space; the base model itself is unmodified. The result is a single
embedding space covering **text, images, video, and audio**, with retrieval supported in
any direction between modalities.

**Highlights**

- **Leads every unified embedding model we measured on audio↔text.** On a single
  cross-modal protocol, this model exceeds ImageBind, LanguageBind, and Gemini
  Embedding 2 on audio↔text in both directions, and both language-bound baselines on
  emergent audio↔image (full tables below).
- **Unmodified base.** Only the connector is trained; the base model's parameters are
  byte-identical to the original release, so its text/image/video retrieval performance
  (MMEB-V2) carries over unchanged.
- **Emergent cross-modal alignment.** The connector is trained exclusively on audio–text
  pairs. Audio→image retrieval nonetheless reaches R@10 0.407 over 696 VGGSound candidates
  (chance: 0.014) with no audio-visual pairs in training — alignment to text places audio
  in the space the base already shares across modalities.
- **Matryoshka representation.** Embeddings truncate to {2048, 1536, 1024, 512, 256, 128,
  64} dimensions with renormalization.
- **Compact distribution.** This repository ships the connector and normalization
  statistics (~60 MB); the frozen towers are downloaded from their original repositories.
  The parameter count shown for this repository (16.4M) is the trained connector —
  `model.safetensors` and the `.pt` checkpoint contain the same weights; `inference.py`
  loads the `.pt`.

This is a **research preview**, currently at **v0.3**: the v0.2 contrastive stage (484K
pairs) followed by a connector-only in-domain fine-tune on the AudioCaps train split.
Earlier versions remain downloadable via the `v0.1-preview` and `v0.2-preview` tags;
`v0.3-preview` pins the current version. All are compared below; pin a tag if you build
on this model.

## Architecture

![fusion-embedding-1 model overview: audio through the frozen Qwen2.5-Omni tower and the trained FusionResampler into the byte-frozen Qwen3-VL-Embedding-2B layer stack, then last-token pooling and Matryoshka truncation into one shared embedding space](assets/fe1_model_overview.png)

A perceiver-resampler (width 384, 64 latent queries) translates frozen audio-tower frames
into the base model's input embedding space; its outputs occupy placeholder positions in
the input stream, mirroring the base model's image-token mechanism. Training is
contrastive (InfoNCE over the Matryoshka ladder, symmetric, with a full-corpus
frozen-text negative bank — 484K captions at v0.2) against the base model's text
embeddings in its native input format. v0.3 adds a second, connector-only fine-tuning
stage on the AudioCaps train split (400 steps at a reduced learning rate), warm-started
from the v0.2 checkpoint.

**Input formatting.** All inputs use the base model's chat-template format (instruction in
the system turn, content in the user turn, last-token pooling). Embedding quality is
sensitive to this formatting; use the templates in `inference.py`. For cross-modal
ranking, per-modality mean-centering of the gallery is recommended (`FusionEmbedder.center`).

## Evaluation

### Cross-modal retrieval — versus unified embedding models

<p align="center">
<img src="assets/fe_positioning.png" alt="Positioning: VGGSound-696 cross-modal retrieval versus trained parameters; the fusion-embedding family leads unified models on audio-text and leads the emergent audio-image cluster (ImageBind's supervised pair annotated)" width="860px">
</p>

VGGSound-AV, 696 audio/video-frame pairs (chance R@10 = 0.014). R@10 shown as
audio-side → other / other → audio-side:

| Model | audio↔image | audio↔text | text↔image |
|---|---|---|---|
| ImageBind-Huge | **0.718 / 0.720** | 0.404 / 0.348 | 0.243 / 0.282 |
| LanguageBind | 0.365 / 0.415 | 0.547 / 0.331 | 0.221 / 0.283 |
| Gemini Embedding 2 (API, 2026-07-09) | 0.312 / 0.316 | 0.379 / 0.374 | 0.273 / **0.366** |
| fusion-embedding-1-2b-preview v0.1 | 0.368 / 0.388 | 0.555 / 0.592 | 0.331 / 0.319 |
| fusion-embedding-1-2b-preview v0.2 | 0.418 / 0.440 | 0.588 / 0.631 | 0.331 / 0.319 |
| **fusion-embedding-1-2b-preview v0.3** | 0.407 / 0.428 | **0.625 / 0.645** | **0.331** / 0.319 |

*ImageBind trains directly on audio–image pairs, so that pair is its supervised direction;
its audio–text alignment is emergent. LanguageBind trains audio against language (its
audio↔text is supervised; the value shown is its best readout, using the audio branch's
own text tower); its audio↔image is emergent. This model trains on audio–text only; its
audio–image alignment is emergent. All models evaluated with identical clips, frames, and
scoring, using the released imagebind_huge checkpoint and revision-pinned LanguageBind
checkpoints (LanguageBind_Audio_FT + LanguageBind_Image). Note on LanguageBind: its
branches fine-tune separate copies of the text tower, which diverge (mean caption cosine
0.55 between the audio and image branches' text embeddings) — the cross-branch binding
weakens, which is consistent with its emergent audio↔image score. This model's shared
space cannot drift by construction (the base is frozen; every training run asserts
parameter-level identity). Gemini Embedding 2 is Google's natively multimodal embedding
API (text/image/video/audio in one space), evaluated at its documented default invocation
(model id `gemini-embedding-2`, 3072-d native output, inline audio+image+text,
google-genai 2.10.0) on the evaluation date shown; API models may change after that date.
One shared caveat: the evaluation captions are model-generated, which could favor models
whose text tower shares that caption style — all models received identical inputs.*

Full audio→image metrics (per-modality mean-centered gallery — the readout implemented by
`FusionEmbedder.center`; chance R@10 = 0.014):

| Version | R@1 | R@5 | R@10 | mAP@10 |
|---|---|---|---|---|
| v0.1 | 0.085 | 0.260 | 0.368 | 0.155 |
| v0.2 | **0.088** | **0.315** | **0.418** | **0.179** |
| v0.3 | 0.085 | 0.297 | 0.407 | 0.177 |

*The v0.3 in-domain fine-tune costs ~1 point of emergent audio→image alignment while
improving audio↔text (see the cross-modal table); v0.2 remains available if audio→image
is the primary use case.*

**What audio→image retrieval looks like.** These numbers are not only aggregates — the
retrievals are organized by sound. Real examples (v0.2 checkpoint) on VGGSound-696
(query clip's frame left, top-5 retrieved images right; green = the clip's exact frame):

![Audio-to-image retrieval examples](assets/audio_to_image_gallery.png)

*Example frames from the [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) dataset (CC-BY-4.0), shown for evaluation illustration.*

*Direct hits* — the clip's own frame is returned in the top 5, among the same kind of scene:

| Sound | Top-5 retrieval | Exact frame |
|---|---|---|
| Metallic clanking and banging | the kitchen it came from, first | rank 1 |
| A dog howling | its own dog, then more howling dogs | rank 1 |
| A cat purring | its own cat, then more purring and meowing cats | rank 1 |
| A siren with a dog howling | its own scene among howling dogs | rank 2 |
| *"Switch on the good piece"* (speech) | the blender being switched on | rank 2 |
| A female singer in a reverberant space | stage performances and singers | rank 3 |

*Right neighbourhood* — the exact frame ranks lower (often a poor still), but the top
results are the correct sound category:

| Sound | Top-5 retrieval | Exact frame |
|---|---|---|
| A man speaking Spanish amid birdsong | a man speaking with birds chirping behind | rank 13 |
| A cat's rhythmic purring | purring and meowing cats | rank 15 |
| Bird chirps and tweets | songbirds, owls, a cawing crow | rank 18 |
| A power-tool whirring | drills and small motors | rank 32 |

### MAEB (beta)

On 10 tasks of the MTEB team's Massive Audio Embedding Benchmark
(mteb 2.18.0, v0.2 checkpoint; ranks vs the live leaderboard as of 2026-07-09, 21–65
models per task): UrbanSound8K T2A retrieval #3, Ravdess zero-shot #4, FSD2019Kaggle #6
(disclosed only — 13.6% of its test clips appear in the FSD50K dev split used in
training, verified by Freesound id, so it is withheld from the official submission),
BeijingOpera #6, with mid-field placements on speech/music tasks the model was never
trained for. Official leaderboard submission in progress.

### Audio–text retrieval — versus specialist CLAP models

**AudioCaps test** — 883 clips, five reference captions per clip, recall computed as
min-rank over references:

| Model | A→T R@1 | A→T R@10 | T→A R@10 |
|---|---|---|---|
| LAION-CLAP | 0.468 | 0.907 | 0.839 |
| WavCaps HTSAT-BERT | 0.517 | 0.906 | 0.861 |
| Cacophony | 0.553 | 0.924 | 0.864 |
| M2D-CLAP | **0.593** | **0.928** | **0.886** |
| fusion-embedding-1-2b-preview v0.1 | 0.216 | 0.626 | 0.680 |
| fusion-embedding-1-2b-preview v0.2 | 0.279 | 0.717 | 0.736 |
| **fusion-embedding-1-2b-preview v0.3** | 0.332 | 0.741 | 0.746 |

*CLAP-family models fine-tune both encoders end-to-end and include AudioCaps and Clotho
training data; this model keeps both towers frozen and trains only the connector.*

**Clotho v2.1 evaluation** — 1,045 clips × 5 references, zero-shot (Clotho is excluded
from training data):

| Model | A→T R@10 | T→A R@10 |
|---|---|---|
| WavCaps CNN14-BERT (zero-shot) | **0.576** | **0.549** |
| fusion-embedding-1-2b-preview v0.1 | 0.252 | 0.329 |
| fusion-embedding-1-2b-preview v0.2 | 0.448 | 0.449 |
| **fusion-embedding-1-2b-preview v0.3** | 0.433 | 0.460 |

*v0.3's in-domain AudioCaps stage trades 1.5 points of zero-shot Clotho A→T for the
AudioCaps gains above; T→A improves in both settings.*

Text, image, and video benchmarks are the base model's published MMEB-V2 results, which
are unaffected by this extension.

## Usage

```python
# pip install git+https://github.com/Eximius-Labs/fusion-embedding-1  (+ transformers, torchvision, pillow)
from inference import FusionEmbedder

fe = FusionEmbedder.from_pretrained("EximiusLabs/fusion-embedding-1-2b-preview",
                                    device="cuda")
# or pin a version: revision="v0.3-preview" (current) / "v0.2-preview" / "v0.1-preview"

a = fe.embed_audio("dog_barking.wav")                        # [2048]
t = fe.embed_text("a dog barks while rain falls")            # [2048]
i = fe.embed_image("dog_photo.jpg")                          # [2048]

print((a @ t), (a @ i), (t @ i))                             # cosine similarities

a256 = fe.embed_audio("dog_barking.wav", dim=256)            # Matryoshka truncation
```

## Training data and license

v0.2 was trained on ~484K audio–caption pairs: the full AudioCaps train split (45K),
FSD50K, WavCaps/AudioSet_SL, and a 318K-clip subset of LAION-FreeSound, using 10-second
training windows (random crop for longer clips). v0.3 continues the v0.2 checkpoint with
a 400-step fine-tune on the AudioCaps train split only. v0.1 used a 131K-pair subset of
the same sources. As this mix includes YouTube-sourced and research-licensed corpora, the preview
is released under **CC-BY-NC-4.0**. Evaluation sets (AudioCaps test, Clotho, VGGSound,
ESC-50) are excluded from training by clip id.

## Limitations

- Trained on sound-event data; speech content, speaker attributes, and music description
  are supported by the instruction taxonomy but not yet trained to comparable quality.
- English captions; 16 kHz mono input; 30 s per window (longer audio is chunked).
- Audio–text retrieval is below fully fine-tuned CLAP-family models at this checkpoint
  (see Evaluation).

## Roadmap

Further corpus scaling, speech and music coverage, a commercially licensed release tier,
and the 8B model.

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

Built on Qwen3-VL-Embedding and Qwen2.5-Omni, with training data from AudioCaps, WavCaps,
and FSD50K.
