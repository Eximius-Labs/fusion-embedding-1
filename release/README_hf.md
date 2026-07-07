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

# fusion-embedding-1-2b-preview

Fusion Embedding 1 extends [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
with an audio modality. A trained connector (~16M parameters) maps frozen
[Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) audio-tower features into the
base model's embedding space; the base model itself is unmodified. The result is a single
embedding space covering **text, images, video, and audio**, with retrieval supported in
any direction between modalities.

**Highlights**

- **Unmodified base.** Only the connector is trained; the base model's parameters are
  byte-identical to the original release, so its text/image/video retrieval performance
  (MMEB-V2) carries over unchanged.
- **Emergent cross-modal alignment.** The connector is trained exclusively on audio–text
  pairs. Audio→image retrieval nonetheless reaches R@10 0.418 over 696 VGGSound candidates
  (chance: 0.014) with no audio-visual pairs in training — alignment to text places audio
  in the space the base already shares across modalities.
- **Matryoshka representation.** Embeddings truncate to {2048, 1536, 1024, 512, 256, 128,
  64} dimensions with renormalization.
- **Compact distribution.** This repository ships the connector and normalization
  statistics (~60 MB); the frozen towers are downloaded from their original repositories.

This is a **research preview**, currently at **v0.2** (trained on a 484K-pair corpus;
v0.1, trained on 131K pairs, remains available as a pinned revision). Both versions are
compared below; pin a revision if you build on this model.

## Evaluation

**AudioCaps test** — 883 clips, five reference captions per clip, recall computed as
min-rank over references:

| Model | A→T R@1 | A→T R@10 | T→A R@10 |
|---|---|---|---|
| LAION-CLAP | 0.468 | 0.907 | 0.839 |
| WavCaps HTSAT-BERT | 0.517 | 0.906 | 0.861 |
| Cacophony | 0.553 | 0.924 | 0.864 |
| M2D-CLAP | **0.593** | **0.928** | **0.886** |
| fusion-embedding-1-2b-preview v0.1 | 0.216 | 0.626 | 0.680 |
| **fusion-embedding-1-2b-preview v0.2** | **0.279** | **0.717** | **0.736** |

*CLAP-family models fine-tune both encoders end-to-end and include AudioCaps and Clotho
training data; this model keeps both towers frozen and trains only the connector.*

**Clotho v2.1 evaluation** — 1,045 clips × 5 references, zero-shot (Clotho is excluded
from training data):

| Model | A→T R@10 | T→A R@10 |
|---|---|---|
| WavCaps CNN14-BERT (zero-shot) | **0.576** | **0.549** |
| fusion-embedding-1-2b-preview v0.1 | 0.252 | 0.329 |
| **fusion-embedding-1-2b-preview v0.2** | **0.448** | **0.449** |

**Cross-modal retrieval** — VGGSound-AV, 696 audio/video-frame pairs (chance R@10 = 0.014).
R@10 shown as audio-side → other / other → audio-side:

| Model | audio↔image | audio↔text | text↔image |
|---|---|---|---|
| ImageBind-Huge | **0.718 / 0.720** | 0.404 / 0.348 | 0.243 / 0.282 |
| fusion-embedding-1-2b-preview v0.1 | 0.368 / 0.388 | 0.555 / 0.592 | 0.331 / 0.319 |
| **fusion-embedding-1-2b-preview v0.2** | **0.418 / 0.440** | **0.588 / 0.631** | **0.331 / 0.319** |

*ImageBind trains directly on audio–image pairs, so that pair is its supervised direction;
its audio–text alignment is emergent. This model trains on audio–text only; its
audio–image alignment is emergent. Both evaluated with identical clips, frames, and
scoring; ImageBind numbers computed with the released imagebind_huge checkpoint.*

Full audio→image metrics (per-modality mean-centered gallery — the readout implemented by
`FusionEmbedder.center`; chance R@10 = 0.014):

| Version | R@1 | R@5 | R@10 | mAP@10 |
|---|---|---|---|---|
| v0.1 | 0.085 | 0.260 | 0.368 | 0.155 |
| **v0.2** | **0.088** | **0.315** | **0.418** | **0.179** |

**What audio→image retrieval looks like.** The 0.368 above is not only an aggregate — the
retrievals are organized by sound. Real examples (v0.1 checkpoint) on VGGSound-696
(query clip's frame left, top-5 retrieved images right; green = the clip's exact frame):

![Audio-to-image retrieval examples](assets/audio_to_image_gallery.png)

*Example frames from the [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) dataset (CC-BY-4.0), shown for evaluation illustration.*

*Direct hits* — the clip's own frame is returned in the top 5, among the same kind of scene:

| Sound | Top-5 retrieval | Exact frame |
|---|---|---|
| A siren | its own ambulance, then more emergency vehicles | rank 2 |
| *"Switch on the good piece"* (speech) | the appliance being switched on | rank 1 |
| A whirring kitchen motor | its own blender, among mixers | rank 3 |
| An emergency-vehicle siren | fire engines and ambulances | rank 4 |
| Metallic clanking and banging | the kitchen it came from | rank 5 |

*Right neighbourhood* — the exact frame ranks lower (often a poor still), but every top
result is the correct sound category:

| Sound | Top-5 retrieval | Exact frame |
|---|---|---|
| A distinct click | people typing at keyboards | rank 109 |
| A power motor | an angle grinder, a belt sander, a table saw | rank 68 |
| A single cowbell note | cattle herds, one wearing a bell | rank 6 |
| High squeaks and chirps | chickens, crows, a quail, a parrot | rank 104 |

Text, image, and video benchmarks are the base model's published MMEB-V2 results, which
are unaffected by this extension.

## Architecture

```
                          ┌────────── Qwen3-VL-Embedding-2B (frozen) ────────┐
text / image / video ───▶ │  native encoding paths, unmodified               │
                          │                                                  │
audio ─▶ Qwen2.5-Omni     │                                                  │
         audio tower      │                                                  │
         (frozen)         │                                                  │
   └─▶ FusionResampler ───┼─▶ audio tokens in the input stream ─▶ LLM ─▶     │
        (trained, ~16M)   │                       last-token pooled embedding│
                          └──────────────────────────────────────────────────┘
```

A perceiver-resampler (width 384, 64 latent queries) translates frozen audio-tower frames
into the base model's input embedding space; its outputs occupy placeholder positions in
the input stream, mirroring the base model's image-token mechanism. Training is
contrastive (InfoNCE over the Matryoshka ladder, symmetric, with a 131K-caption
full-corpus negative bank) against the base model's text embeddings in its native
input format.

**Input formatting.** All inputs use the base model's chat-template format (instruction in
the system turn, content in the user turn, last-token pooling). Embedding quality is
sensitive to this formatting; use the templates in `inference.py`. For cross-modal
ranking, per-modality mean-centering of the gallery is recommended (`FusionEmbedder.center`).

## Usage

```python
# pip install git+https://github.com/Eximius-Labs/fusion-embedding-1  (+ transformers, torchvision, pillow)
from inference import FusionEmbedder

fe = FusionEmbedder.from_pretrained("EximiusLabs/fusion-embedding-1-2b-preview",
                                    device="cuda")

a = fe.embed_audio("dog_barking.wav")                        # [2048]
t = fe.embed_text("a dog barks while rain falls")            # [2048]
i = fe.embed_image("dog_photo.jpg")                          # [2048]

print((a @ t), (a @ i), (t @ i))                             # cosine similarities

a256 = fe.embed_audio("dog_barking.wav", dim=256)            # Matryoshka truncation
```

## Training data and license

v0.2 was trained on ~484K audio–caption pairs: the full AudioCaps train split (45K),
FSD50K, WavCaps/AudioSet_SL, and a 318K-clip subset of LAION-FreeSound, using 10-second
training windows (random crop for longer clips). v0.1 used a 131K-pair subset of the same
sources. As this mix includes YouTube-sourced and research-licensed corpora, the preview
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
