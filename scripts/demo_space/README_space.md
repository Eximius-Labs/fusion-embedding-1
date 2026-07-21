---
title: Fusion Embedding Demo
emoji: 🔊
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: mit
models:
  - EximiusLabs/fusion-embedding-2-2b-preview
  - EximiusLabs/fusion-embedding-1-2b-preview
short_description: One embedding space for text, images, and audio
---

# Fusion Embedding demo

One embedding space for text, images, video, and audio, built on frozen
models. This Space runs the released
[fusion-embedding-2-2b-preview](https://huggingface.co/EximiusLabs/fusion-embedding-2-2b-preview)
checkpoint at a pinned revision through its public `trust_remote_code` path.

Three views:

- **Sound → Images** — upload or record a sound, retrieve matching images.
  The model was never trained on an audio–image pair; audio is aligned to
  text only, and retrieval to images emerges through the frozen base's
  existing text–image geometry.
- **Text → Sound** — describe a sound, retrieve real recordings.
- **One Space** — a single query ranked against audio and images together
  with one similarity metric.

Galleries are pre-embedded; only the query is encoded live.

## Gallery media and attribution

All gallery media are redistributable: audio clips are CC0/CC-BY recordings
from the FSD50K evaluation collection (Freesound), images are CC0/CC-BY via
Openverse. Per-item attribution (title, creator, license, source link) ships
in `audio_attribution.json` and `image_attribution.json`. Gallery audio is
drawn from FSD50K's evaluation split, which is outside the model's training
corpus.

## Links

- Code and training recipe: https://github.com/Eximius-Labs/fusion-embedding
- Both model generations appear on the public MTEB audio (MAEB) and video
  (MVEB) leaderboards via the official `mteb` harness.
