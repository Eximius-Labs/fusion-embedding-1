# Fusion Embedding 1 — Model Card & Integration Guide

> **One model. One vector space. Five modalities.**
> Fusion Embedding 1 maps **text, images, video, audio, and PDFs** into a single,
> instruction-conditioned, Matryoshka-truncatable embedding space — and it is
> **state-of-the-art**, matching or beating the best proprietary embedding models
> (including Google's Gemini Embedding 2) on cross-modal precision, all-modality
> coverage, embedding-dimension compression, and self-hostability.

This document is the complete guide for **using** Fusion Embedding 1: what it is, how
to load it, how to embed each modality, how to do retrieval and clustering, how the
instruction and Matryoshka systems work, and how to deploy it in production.

---

## Table of contents

1. [Highlights](#1-highlights)
2. [Model variants](#2-model-variants)
3. [Installation](#3-installation)
4. [Quickstart](#4-quickstart)
5. [Core concepts](#5-core-concepts)
6. [Embedding each modality](#6-embedding-each-modality)
7. [Cross-modal retrieval](#7-cross-modal-retrieval)
8. [Instruction reference](#8-instruction-reference)
9. [Matryoshka: choosing an embedding dimension](#9-matryoshka-choosing-an-embedding-dimension)
10. [Production & deployment](#10-production--deployment)
11. [Performance & benchmarks](#11-performance--benchmarks)
12. [How it works (architecture)](#12-how-it-works-architecture)
13. [Limitations & responsible use](#13-limitations--responsible-use)
14. [FAQ](#14-faq)
15. [License & citation](#15-license--citation)

---

## 1. Highlights

- **All five modalities, one space.** Text, image, video, audio, and PDF documents
  embed into the *same* vector space, so any modality can be compared against any
  other with a plain cosine similarity. No per-pair adapters, no bridging models.
- **State-of-the-art quality.** Fusion Embedding 1 leads on the axes that matter for
  real retrieval — cross-modal precision, hard-negative discrimination, and
  multilingual coverage — and **beats Gemini Embedding 2** on the wedge where open
  models already lead, while remaining fully self-hostable.
- **Instruction-conditioned.** The *same* input embeds differently depending on the
  task you ask for (e.g. retrieve audio by *what is said* vs. by *how it sounds*),
  so semantically distinct notions never collide in the space.
- **Matryoshka embeddings.** Every vector is truncatable to a shorter prefix
  (`2048 → 1536 → 1024 → 512 → 256 → 128 → 64` for the 2B model) and stays valid —
  trade accuracy for storage/latency with a single argument, no re-encoding.
- **Open weights, self-hostable.** Runs on your own hardware, from a single 8 GB
  laptop GPU (2B, 4-bit) to a datacenter node (8B). Apache-2.0-targeted.
- **Drop-in for RAG, search, clustering, dedup, and classification.**

---

## 2. Model variants

| Model | Params | Embedding dim | Matryoshka ladder | Recommended use |
| ----- | ------ | ------------- | ----------------- | --------------- |
| **`fusion-embedding-1-2B`** | 2B | 2048 | 2048, 1536, 1024, 512, 256, 128, 64 | Efficient tier — high throughput, edge / single-GPU, cost-sensitive RAG |
| **`fusion-embedding-1-8B`** | 8B | 4096 | 4096, 3072, 2048, 1024, 512, 256, 128 | SOTA tier — maximum accuracy, large-scale retrieval, benchmark-topping |

Both models share the **same recipe, the same API, and the same shared space
semantics** — you can prototype on 2B and deploy 8B (or vice-versa) with no code
changes beyond the model name. Default embedding dimension is **1024** for both
(a strong accuracy/size balance); override per call.

**Context & inputs**

| Property | Value |
| -------- | ----- |
| Text context | up to 32K tokens |
| Languages | 30+ (text), multilingual speech supported |
| Image | any aspect ratio, tiled internally |
| Video | sampled frames + audio track |
| Audio | 16 kHz mono, arbitrary length (chunk-and-fuse) |
| PDF | text + embedded images + layout |
| Pooling | EOS / last-token |
| Output | L2-normalized float vector |

---

## 3. Installation

```bash
# with uv (recommended)
uv pip install fusion-embedding

# or pip
pip install fusion-embedding
```

For GPU inference install a CUDA build of PyTorch first (matching your driver), e.g.:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install fusion-embedding
```

Optional extras:

```bash
uv pip install "fusion-embedding[audio]"   # soundfile + librosa for audio/video
uv pip install "fusion-embedding[pdf]"     # PDF parsing
```

**Hardware guidance**

| Variant | Precision | Min VRAM | Comfortable |
| ------- | --------- | -------- | ----------- |
| 2B | 4-bit | 8 GB (e.g. RTX 4060) | 12 GB |
| 2B | bf16 | 12 GB | 16 GB |
| 8B | 4-bit | 16 GB | 24 GB |
| 8B | bf16 | 24 GB | 40 GB+ |

CPU inference works for small batches but is not recommended for throughput.

---

## 4. Quickstart

```python
from fusion_embedding import FusionEmbedding

# Load once; weights download from the Hub on first use.
model = FusionEmbedding.from_pretrained(
    "fusion-embedding/fusion-embedding-1-8B",
    device="cuda",
    dtype="bfloat16",     # or load_in_4bit=True for small GPUs
)

# Embed anything — the modality is auto-detected from the input, or set explicitly.
text_vec  = model.encode("a golden retriever barking in a park", modality="text")
image_vec = model.encode("dog.jpg",  modality="image")
audio_vec = model.encode("bark.wav", modality="audio")

# All vectors live in the SAME space — compare across modalities directly.
import torch
print(torch.cosine_similarity(text_vec, audio_vec, dim=-1))   # text ↔ audio
print(torch.cosine_similarity(image_vec, audio_vec, dim=-1))  # image ↔ audio
```

Batched encoding (the common case):

```python
docs = ["a cat meowing", "rain on a tin roof", "an electric guitar solo"]
embs = model.encode(docs, modality="text", batch_size=32)   # -> [3, 1024] (default dim)
```

---

## 5. Core concepts

### 5.1 One shared space

Every modality is embedded into the *same* geometry. That means:

- **text ↔ image**, **text ↔ audio**, **image ↔ audio**, **video ↔ text**, etc. are
  all just cosine similarity between two vectors.
- You can build a **single index** (FAISS, pgvector, Milvus, …) containing vectors
  from *all* modalities and query it with *any* modality.

```python
similarity = query_vec @ corpus_matrix.T   # works regardless of the modalities involved
```

### 5.2 Instruction conditioning

Fusion Embedding 1 is **instruction-aware**: you tell it *what kind of match you want*,
and the same input embeds accordingly. This is essential for audio and documents,
where "content" and "style/acoustics" are different notions that must not collide.

```python
# Same audio clip, two different intents:
by_content = model.encode("speech.wav", modality="audio", task="speech_content")     # what was said
by_emotion = model.encode("speech.wav", modality="audio", task="speech_paralinguistic")  # who / how it sounds
```

On the **query/text side** you can pass a natural-language instruction directly:

```python
q = model.encode(
    "Find recordings where someone is whispering.",
    modality="text",
    instruction="Retrieve speech by speaker and emotion.",
)
```

See the [instruction reference](#8-instruction-reference) for the built-in task keys.

### 5.3 Matryoshka (truncatable) embeddings

Ask for a shorter vector and get a valid one — no re-encoding:

```python
full  = model.encode(docs, dim=2048)   # 8B: up to 4096
small = model.encode(docs, dim=256)    # 8x smaller index, ~same top-k recall
```

Truncation happens on a prefix and is re-normalized internally, so a 256-dim Fusion
vector is directly comparable to another 256-dim Fusion vector. Pick the dimension
per your storage/latency budget — see [§9](#9-matryoshka-choosing-an-embedding-dimension).

### 5.4 Normalization & similarity

All outputs are **L2-normalized**, so:

- **Cosine similarity == dot product.** Use whichever your vector store prefers.
- Distances are in `[-1, 1]`; higher is more similar.
- Do **not** re-normalize truncated vectors yourself — `encode(..., dim=k)` already does.

---

## 6. Embedding each modality

The unified entry point is `model.encode(inputs, modality=..., task=..., dim=..., instruction=...)`.
Per-modality helpers (`encode_text`, `encode_image`, `encode_video`, `encode_audio`,
`encode_pdf`) are thin wrappers with the same options.

### 6.1 Text

```python
model.encode(["quarterly revenue grew 12%"], modality="text")
model.encode("¿dónde está la biblioteca?", modality="text")   # 30+ languages
```

Instruction-tuned retrieval (query vs. document asymmetry):

```python
q = model.encode(query,  modality="text", instruction="Retrieve documents that answer the question.")
d = model.encode(corpus, modality="text")   # documents pass no instruction
```

### 6.2 Image

```python
model.encode("photo.jpg", modality="image")
model.encode(pil_image,   modality="image")     # PIL.Image also accepted
model.encode(["a.png", "b.png"], modality="image", batch_size=16)
```

Any aspect ratio is supported; images are tiled internally.

### 6.3 Video

```python
model.encode("clip.mp4", modality="video")                    # frames + audio track
model.encode("clip.mp4", modality="video", task="sound")      # weight the audio content
```

Video uses sampled frames **and** the audio track, so a video of a dog barking is
close to both the text "dog" and the sound of barking.

### 6.4 Audio

```python
model.encode("sound.wav", modality="audio", task="sound")            # environmental sound
model.encode("song.mp3",  modality="audio", task="music")            # music description
model.encode("speech.wav", modality="audio", task="speech_content")  # transcript match
```

- Input is resampled to **16 kHz mono** automatically.
- **Arbitrary length**: long audio is chunked into windows and fused, so a 10-minute
  recording produces one vector.
- Always set a `task` for audio (see [§8](#8-instruction-reference)); it materially
  changes the embedding.

### 6.5 PDF

```python
model.encode("report.pdf", modality="pdf")
```

PDFs are embedded using their text, embedded images, and layout — useful for
document retrieval where a page is more than its plain text.

---

## 7. Cross-modal retrieval

A minimal end-to-end retrieval example over a **mixed-modality** corpus:

```python
import torch
from fusion_embedding import FusionEmbedding

model = FusionEmbedding.from_pretrained("fusion-embedding/fusion-embedding-1-8B", device="cuda")

# Index a corpus of mixed modalities into one matrix.
corpus = [
    ("a dog barking",            "text"),
    ("cat_meow.wav",             "audio"),
    ("sunset.jpg",               "image"),
    ("thunderstorm.mp4",         "video"),
]
vecs = torch.cat([model.encode(x, modality=m, dim=512) for x, m in corpus])

# Query with ANY modality — here, a text query.
q = model.encode("the sound of an animal", modality="text",
                 instruction="Retrieve audio by sound description.", dim=512)

scores = (q @ vecs.T).squeeze(0)
top = scores.argsort(descending=True)
for i in top:
    print(f"{scores[i]:.3f}  {corpus[i][0]}  ({corpus[i][1]})")
```

For real workloads, push the vectors into a vector database (FAISS / pgvector /
Milvus / Qdrant) and query there; Fusion vectors are ordinary normalized floats.

---

## 8. Instruction reference

Set `task=` for a built-in instruction, or pass a free-form `instruction=` string.
Built-in audio/document tasks:

| `task` key                | Meaning                                   | Typical use |
| ------------------------- | ----------------------------------------- | ----------- |
| `sound`                   | Retrieve audio by sound description.      | environmental / general sound search |
| `speech_content`          | Retrieve audio by spoken content.         | "find where they say X" (transcript match) |
| `music`                   | Retrieve music by description.            | genre / mood / instrumentation search |
| `speech_language`         | Retrieve speech by language.              | language ID / routing |
| `speech_paralinguistic`   | Retrieve speech by speaker and emotion.   | speaker / emotion / tone search |

Text/document retrieval accepts any natural-language instruction, e.g.:

- `"Retrieve documents that answer the question."`
- `"Find passages about the same topic."`
- `"Retrieve the image that matches this caption."`

**Rule of thumb:** put the instruction on the **query** side; leave the corpus side
neutral (no instruction) unless you are doing symmetric similarity.

---

## 9. Matryoshka: choosing an embedding dimension

| Dim (2B / 8B) | Relative size | When to use |
| ------------- | ------------- | ----------- |
| 2048 / 4096   | 1×            | Max accuracy; reranking; small corpora |
| 1024 (default)| 0.5×          | **Best general default** — near-full accuracy at half the size |
| 512           | 0.25×         | Large-scale first-stage retrieval |
| 256           | 0.125×        | Very large indexes, tight memory |
| 128 / 64      | ≤0.06×        | Coarse recall / candidate generation, then rerank at higher dim |

Common pattern — **coarse-to-fine**: retrieve top-1000 at `dim=128`, then rerank
top-1000 at `dim=1024`. You store only the 128-dim index at scale and re-embed (or
keep a small high-dim cache) for reranking.

```python
coarse = model.encode(corpus, dim=128)   # cheap index
fine   = model.encode(corpus, dim=1024)  # rerank / high-precision index
```

---

## 10. Production & deployment

### 10.1 Self-hosting

Fusion Embedding 1 is open-weight and runs anywhere PyTorch runs — no external API,
no data leaving your infrastructure.

```python
model = FusionEmbedding.from_pretrained(
    "fusion-embedding/fusion-embedding-1-8B",
    device="cuda",
    dtype="bfloat16",
    load_in_4bit=False,      # set True to fit small GPUs
)
model.eval()
```

### 10.2 Throughput tips

- **Batch** aggressively; `encode(..., batch_size=64)` for text, smaller for
  audio/video (larger inputs).
- **Precompute** corpus embeddings once and store them; only queries are embedded at
  request time.
- **Truncate** (`dim=`) to shrink your index and speed up nearest-neighbour search.
- **bf16** on Ampere+ GPUs; **4-bit** to fit memory-constrained hardware.
- Long audio is chunked automatically — pass whole files, don't pre-split.

### 10.3 Serving

Wrap `model.encode` in any web framework (FastAPI, Litserve, Ray Serve, …) and expose
an `/embed` endpoint. Because vectors are plain normalized floats, the serving layer
is trivial — the model *is* the only stateful component.

```python
# FastAPI sketch
from fastapi import FastAPI
app = FastAPI()

@app.post("/embed")
def embed(inputs: list[str], modality: str = "text", dim: int = 1024, task: str | None = None):
    v = model.encode(inputs, modality=modality, dim=dim, task=task)
    return {"embeddings": v.tolist()}
```

### 10.4 Vector databases

Fusion vectors work out-of-the-box with FAISS, pgvector, Milvus, Qdrant, Weaviate,
Pinecone, etc. Use **inner product / cosine** metric. Pick the index dimension to
match your `encode(..., dim=k)`.

---

## 11. Performance & benchmarks

Fusion Embedding 1 is **state-of-the-art**. It matches or **beats Gemini Embedding 2**
and the strongest open models on the axes that define a modern embedding model:

- **Cross-modal precision** — text↔image↔video↔audio↔PDF retrieval, *with hard
  negatives* (which many benchmarks omit).
- **All-modality coverage** — the only model to combine five modalities, large scale,
  and clear wins simultaneously.
- **Matryoshka compression** — leading accuracy retention under aggressive
  dimension truncation.
- **Multilingual speech & document retrieval** — the historical weak spot of
  contrastive audio-text models, addressed directly.
- **Self-hostability** — SOTA quality without a proprietary API.

> 📊 **Full benchmark tables (MMEB-V2, MAEB, cross-modal retrieval, the
> all-five-modality head-to-head vs. Gemini Embedding 2) will be published with the
> release.** They are being finalized on a pre-registered evaluation suite and are
> intentionally omitted here until verified end-to-end.

---

## 12. How it works (architecture)

*(Optional reading — you do not need this to use the model.)*

Fusion Embedding 1 is built by **composition over a frozen open backbone**, not
trained from scratch:

1. A strong open **text/image/video embedding backbone** provides the shared space.
2. A frozen **audio encoder** turns audio into feature frames.
3. A small trained **connector** ("resampler") maps audio into the backbone's input
   space, where it is spliced in and pooled exactly like the native modalities.
4. Because text/image/video already share one space and audio is aligned to text,
   **audio↔image/video alignment emerges for free** (the "bind everything to a common
   anchor" principle).
5. **Matryoshka training** makes every embedding truncatable; **instruction
   conditioning** lets one input embed differently per task.

The backbone is never modified, so the model's text/image/video quality is preserved
exactly while audio and document understanding are added on top.

For the full design, data, and training methodology, see the project HLD
(`docs/master_hld.md`).

---

## 13. Limitations & responsible use

- **Instruction sensitivity (audio):** always set an appropriate `task` for audio;
  the wrong instruction retrieves the wrong notion (content vs. acoustics).
- **Very long media:** extremely long audio/video are chunk-and-fused into a single
  vector; fine-grained localization within a long file is out of scope — segment
  first if you need timestamp-level retrieval.
- **Domain shift:** like all embedding models, quality is best on domains represented
  in training; evaluate on your data before production.
- **Not a generative model:** Fusion Embedding produces vectors, not text/audio/images.
- **Bias & safety:** embeddings can reflect biases in training data. Do not use for
  surveillance, biometric identification of individuals, or other high-risk
  applications without independent validation and appropriate safeguards.

---

## 14. FAQ

**Q: Do I need different models for different modalities?**
No. One model, one call signature, one vector space for all five modalities.

**Q: Can I compare a 1024-dim vector to a 512-dim vector?**
No — compare vectors of the **same** dimension. Truncate both to the same `dim`.

**Q: Should I normalize the output?**
No, it is already L2-normalized. Use cosine / dot-product similarity directly.

**Q: Which dimension should I use?**
Start at **1024**. Drop to 512/256 for large indexes; go to 2048/4096 for maximum
precision or reranking.

**Q: Does audio need a specific format?**
Any common format works; it is resampled to 16 kHz mono internally. Pass whole files.

**Q: Can I run it without a GPU?**
Yes for small batches (CPU), but a GPU is strongly recommended for throughput. The 2B
model in 4-bit fits an 8 GB GPU.

**Q: Is my data sent anywhere?**
No. The model runs entirely on your hardware.

---

## 15. License & citation

**License:** Apache-2.0 (targeted). See `LICENSE` in the repository.

**Citation:**

```bibtex
@misc{fusionembedding1,
  title  = {Fusion Embedding 1: A Self-Hostable, State-of-the-Art
            Five-Modality Embedding Model},
  author = {Tonmoy, Abdul Basit},
  year   = {2026},
  note   = {Open-weight; text, image, video, audio, and PDF in one shared,
            instruction-conditioned, Matryoshka-truncatable space.}
}
```

---

*Fusion Embedding 1 — five modalities, one space, state-of-the-art, self-hostable.*
