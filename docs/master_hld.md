# Fusion Embedding 1 — Master HLD

## Author: **Abdul Basit Tonmoy**

> Single source of truth. Self-contained: read this top-to-bottom and you have
> the whole project — goal, decisions, exact numbers, architecture, loss, staged
> plan, data, eval, repo state, and references. Code scaffold lives in the
> `fusion_embedding/` package (file map in §10).

---

## 0. TL;DR

Build **Fusion Embedding 1**: an open-weight embedding model that maps **text,
image, video, audio, and PDF** into one vector space — a self-hostable competitor
to Google's Gemini Embedding 2. We do **not** train from scratch. We take
**Qwen3-VL-Embedding** (open SOTA on text/image/video, but no audio), **freeze
it**, and **graft an audio pathway** onto it by training only a small connector.
Audio is aligned to text; because text/image/video already share the base's
space, audio↔image/video alignment emerges for free. Prototype on the **2B**
base, ship the **8B** as the SOTA tier. Everything is named `fusion`.

**The wedge:** match Gemini's five-modality breadth while beating it on
cross-modal precision, embedding-dimension (Matryoshka) compression, and
self-hostability — the axes where open models already lead and where Gemini is
weakest.

---

## 1. Goal & positioning

- **Deliverable:** `fusion-embedding-1-2B` (efficient tier) and
  `fusion-embedding-1-8B` (SOTA tier), open-weight, Apache-targeted.
- **Capability:** one model, one shared space, five modalities, instruction-
  conditioned, Matryoshka-truncatable, drop-in for RAG/retrieval/clustering.
- **Why achievable:** the open ecosystem is already at parity on quality — e.g.
  Qwen3-VL-Embedding-8B leads MMEB-V2 (77.8) and beats Gemini on cross-modal
  retrieval; jina-v5-omni already covers all modalities at small scale. The gap
  is that **no open model has all five modalities AND large scale AND clear wins
  at once.** Qwen3-VL-Embedding is large + SOTA but lacks audio. We add the one
  missing modality to the open cross-modal leader.

---

## 2. Locked decisions

1. **Graft, don't build.** Freeze Qwen3-VL-Embedding; add audio via a trained
   connector. Never retrain or modify the base.
2. **Frozen base + frozen audio encoder; train only the connector** (perceiver-
   resampler) and a learnable temperature. (Precedent: jina-v5-omni trains
   ~0.35% of weights.)
3. **Text is the bridge.** Train audio↔text; audio↔image/video emerges
   (ImageBind property).
4. **2B first → 8B.** De-risk the whole pipeline cheaply on 2B; the 8B is a
   re-run, not a redesign.
5. **Full speech parity** (not sound/music-first). This is the hard commitment;
   it drives the encoder choice, the speech data, and the eval weighting.
6. **Single audio encoder = Qwen2.5-Omni audio tower** (Whisper-large-v3-derived,
   general-audio trained). **Dual-encoder fallback** (Whisper + a general-audio
   encoder) is reserved, pulled only if P3 shows the single encoder can't carry
   both speech and sound.
7. **Apache 2.0 target** (verify the Omni audio-encoder license before release;
   fall back to Whisper-MIT + a permissive general-audio encoder if restrictive).
8. **Naming:** everything starts with `fusion`
   (`FusionEmbeddingModel`, `FusionResampler`, `FusionContrastiveLoss`, …).

---

## 3. Confirmed config anchors (use these exact numbers)

**Base — Qwen3-VL-Embedding-2B** (`Qwen/Qwen3-VL-Embedding-2B`)

- Dual-tower **bi-encoder**; **EOS / last-token pooling** (not mean/CLS).
- `d_LLM = d_emb = 2048`.
- **Matryoshka ladder: {2048, 1536, 1024, 512, 256, 128, 64}**, default 1024.
- Quantization-Aware Training; 32K context; 30+ languages; instruction-aware.
- Vision path: `image → vision encoder → merger → tokens → LLM`, pooled with
  last-token. (Our audio path is the exact parallel.)

**Base 8B (final tier):** `d ≈ 4096`; identical recipe.

**Audio encoder — Qwen2.5-Omni audio tower** (from `Qwen/Qwen2.5-Omni-7B`)

- Derived from **Whisper-large-v3**; `d_audio = 1280`.
- Input: 16 kHz → **128-channel mel** (25 ms window, 10 ms hop).
- **~25 output frames/s** (~40 ms/frame, after its stride-2 pooling) →
  **~750 frames per 30 s**; processes in **2-second blocks**.

**Connector (FusionResampler):** Flamingo-style **bottleneck at `d_resampler =
256`** (in_proj 1280→256, blocks at 256, out_proj 256→2048); latent queries at
256; **N = 64** for sound/music; **N = 128–200** for speech parity (Uni-MoE
precedent: 200 query tokens / 30 s captures timbre/intonation/emotion).
**≈ 7.2M params (~0.36% of the 2B base)** — consistent with the §2 thin-adapter
thesis.

---

## 4. Architecture

```
                          ┌─────────────────────── FROZEN base (Qwen3-VL-Embedding) ───────────────────────┐
text / image / video ───▶ │  (the base's own paths — UNTOUCHED, byte-identical to the release)             │
                          │                                                                                │
audio ─▶ [Qwen2.5-Omni    │                                                                                │
         audio encoder]   │                                                                                │
         (FROZEN)         │                                                                                │
         frames [B,T,1280]│                                                                                │
   └─▶ [FusionResampler] ─┼─▶ audio tokens [B,N,2048] ─▶ spliced at <|audio_pad|> positions in            │
        (TRAINED)         │      the LLM input-embedding stream ─▶ frozen LLM ─▶ EOS-token hidden [B,2048] │
                          └────────────────────────────────────────────────────────────────────┬─────────┘
                                                                                                 ▼
                                                          MRL-truncate (any ladder rung) ─▶ L2-normalize ─▶ embedding
```

### 4.1 Audio injection (the key mechanic)

Mirror the base's image-token mechanism:

1. Build the audio item's token sequence: `[N × <|audio_pad|>] + <EOS>` (the
   query/text side carries the instruction; the audio side is neutral).
2. Get input embeddings from the frozen LLM's `embed_tokens`.
3. **Overwrite the `<|audio_pad|>` positions' embeddings with the
   FusionResampler's N audio tokens.**
4. Forward the frozen LLM with `inputs_embeds`; take the **EOS hidden state**.
5. Truncate to an MRL rung, L2-normalize → embedding in the shared space.

**Invariant:** EOS pooling and the 2048 MRL ladder are fixed — **audio conforms
to the base, never the reverse.** The connector touches only the _input_.

### 4.2 FusionResampler (perceiver-resampler, Flamingo-style bottleneck)

The connector is a _translator_, not an understander: it maps the frozen audio
encoder's already-rich 1280-d frames into the LLM's input space. To honor the
thin-adapter thesis (§2, ~0.35% trained) it runs at a **bottleneck width
`d_resampler = 256`**, not at the LLM width:

- `in_proj: Linear(1280 → 256)` on audio frames; add positional encoding over the
  frame (time) axis.
- `N` learnable latent queries ∈ ℝ^{N×256}.
- `L = 6` pre-norm blocks **at width 256**; each = latent **self-attention** →
  **cross-attention** (queries attend to audio frames as K/V, with frame padding
  mask) → FFN (4×).
- `out_proj: Linear(256 → 2048)` + LayerNorm → audio tokens ∈ ℝ^{N×2048} (back up
  to the LLM input width for splicing).
- **Long audio (>30 s):** chunk into windows, resample each to N tokens,
  concatenate up to `max_windows` (cap), optionally feature-fuse a global summary
  with local windows (LAION-CLAP trick).
- **Param count ≈ 7.2M ≈ 0.36% of the 2B base** — consistent with §2. (Full width
  at 2048 would be **~405M ≈ 20%** of the base, contradicting the thesis; rejected.
  Even a single full-width block alone is ~70M ≈ 3.5%.)

**Capacity dial.** `d_resampler` and `N` are the knobs. 256 is deliberately lean;
dense multilingual speech-content retrieval is where it may pinch. Escalation
order if P2/P3 shows the connector is the bottleneck: **widen `d_resampler` first
(256 → 384 → 512; ~tens of M, ~1%), then raise `N`. Never widen the blocks to the
full LLM width.**

### 4.3 Frozen vs trained

| Component                                                   | P1          | P2                       | P3                 |
| ----------------------------------------------------------- | ----------- | ------------------------ | ------------------ |
| Base LLM / vision / text / output head                      | frozen      | frozen                   | frozen             |
| Qwen2.5-Omni audio encoder                                  | frozen      | frozen (opt. small LoRA) | frozen (opt. LoRA) |
| **FusionResampler**                                         | **trained** | **trained**              | **trained**        |
| Learnable temperature                                       | trained     | trained                  | trained            |
| e5-omni alignment modules (per-modality τ, whitening/CORAL) | —           | trained                  | trained            |
| 2nd (speech) encoder + connector (dual-encoder fallback)    | —           | —                        | only if triggered  |
| LLM LoRA                                                    | off         | off (default)            | off (default)      |

---

## 5. Training methodology

### 5.1 Loss (implemented in `losses.py`)

```
L = Σ_{D ∈ MRL} w_D · InfoNCE_D(audio, text)   +   λ_coral · CORAL(audio, text)
```

- **Symmetric InfoNCE** with a **learnable modality temperature** (logit scale),
  in-batch negatives:
  `½[CE(scale·A·Tᵀ ; i) + CE(scale·T·Aᵀ ; i)]`, init temp ≈ base text temp.
- **Matryoshka tiling:** compute InfoNCE at each ladder rung (truncate +
  renormalize), weighted sum → embeddings are truncatable consistently with the
  base.
- **CORAL / covariance alignment:** `‖Cov(audio) − Cov(text)‖²_F / d²`, light in
  P1 (`λ≈0.05`), raised in P2 — keeps audio from forming its own cluster.
- **Debiased contrastive** (P2, `γ⁺≈0.1`): corrects audio captioning's heavy
  one-to-many false negatives.
- **Hard negatives** (P2): mine confusable negatives with the P1 model; add to
  the denominator.

### 5.2 Multi-stage schedule

- **Stage 1 (P1) — connector alignment:** train _only_ the connector on large
  noisy audio-caption data; symmetric InfoNCE + light CORAL + MRL + instructions.
  Goal: audio lands in the space.
- **Stage 2 (P2) — contrastive + alignment hardening:** clean data + mined hard
  negatives + the full e5-omni alignment suite + optional tiny audio-encoder
  LoRA. Goal: competitive on sound + music; small modality gap.
- **Stage 3 (P3) — speech parity + grounding:** heavy multilingual speech; raise
  N to 128–200 for speech; content-vs-acoustic instructions at scale; add direct
  audio↔video/image pairs. Trigger the dual-encoder fallback if needed.

### 5.3 Optimization (defaults in `config.py`)

- AdamW, **lr ≈ 1e-4** (connector-only → high is fine), warmup 5%, cosine decay.
- **Large effective batch** (1024–4096) via micro-batch × grad-accum × world
  size — InfoNCE quality scales with negatives.
- bf16 autocast; keep trained params in fp32; FlashAttention; FSDP/DeepSpeed
  ZeRO. Trained-params checkpoints only (connector + temperature).
- Audio frontend: 16 kHz mono, 128-mel, 30 s windows + chunk-and-fuse.

---

## 6. Instruction taxonomy (the content-vs-acoustic split)

The base is instruction-aware. Each pair is tagged with the instruction matching
its supervision; the **same audio embeds differently** depending on the task, so
content and acoustic notions don't fight. Query/text side carries the
instruction; audio side is neutral.

| `task` key              | Instruction                             | Trains on                                    |
| ----------------------- | --------------------------------------- | -------------------------------------------- |
| `sound`                 | Retrieve audio by sound description.    | sound ↔ caption                              |
| `speech_content`        | Retrieve audio by spoken content.       | speech ↔ transcript                          |
| `music`                 | Retrieve music by description.          | music ↔ caption (genre/mood/instrumentation) |
| `speech_language`       | Retrieve speech by language.            | speech ↔ language                            |
| `speech_paralinguistic` | Retrieve speech by speaker and emotion. | speech ↔ speaker/emotion desc                |

Weight `speech_content` and `speech_language` heavily in eval — that's MAEB's
known weak spot for contrastive audio-text models.

---

## 7. Staged execution plan

Regression guard every stage: **MMEB-V2 must stay unchanged** (base frozen). If
it moves, a base parameter leaked into training.

| Phase                            | Objective                                                                                                                            | Trains                                | Exit gate                                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **P0** Baseline & infra          | Stand up frozen base, reproduce its numbers, build eval harness + data pipeline + distributed training; record the audio-blind floor | nothing                               | Base MMEB-V2 subset reproduced within noise; harness runs end-to-end; loaders emit instruction-templated batches           |
| **P1** Connector alignment       | Get audio into the space                                                                                                             | connector + temp                      | AudioCaps & Clotho R@1 (A→T, T→A) **> 0 / climbing**; MMEB-V2 **unchanged**                                                |
| **P2** Contrastive + alignment   | Competitive on sound+music; harden geometry                                                                                          | + alignment modules + opt. audio LoRA | Competitive **MAEB** (sound+music); MRL holds; MMEB-V2 unchanged; modality gap small                                       |
| **P3** Speech parity + grounding | Close speech gap; add audio↔video/image                                                                                              | + (opt.) 2nd encoder                  | **MAEB multilingual speech retrieval non-random & competitive**; no domain near-random; cross-modal audio→video works      |
| **P4** Eval, soup, release       | Prove Gemini-competitor; finalize                                                                                                    | —                                     | Match/beat Gemini Embedding 2 on the wedge axes on a pre-registered all-5-modality benchmark; ablations support the design |

**8B scale-up:** after 2B validates P0–P4, repeat P1–P3 on
`Qwen3-VL-Embedding-8B` (`d≈4096`) — same connector scaled, same data, same
losses, re-tuned batch/lr. The recipe transfers; gains grow with scale.

**Cross-stage invariants:** base frozen → regression guard; EOS pooling + 2048
ladder fixed; instruction taxonomy from P1 (light) → P3 (full); dual-encoder
fallback is the one reserved escape hatch.

---

## 8. Data sourcing

**Scale comes from synthetic captioning, not human annotation.** The modern open
pipeline (audio-LLM extracts content → LLM writes caption → CLAP filters) is open
and is how you reach Gemini-scale; you can run it on any CC audio you control,
and **the captions you generate are yours** (sidesteps most licensing).

### 8.1 General sound ↔ text

| Dataset          | Pairs                                | Source / method                    | License note                             |
| ---------------- | ------------------------------------ | ---------------------------------- | ---------------------------------------- |
| AudioSetCaps     | 1.9M (+4.1M YT-8M/VGGSound ≈ **6M**) | audio-LLM + LLM, **open pipeline** | audio YouTube-sourced                    |
| Sound-VECaps     | 1.66M                                | audio+visual+LLM                   | enriched detail                          |
| Auto-ACD         | 1.5M                                 | audio+visual+LLM                   | from AudioSet+VGGSound                   |
| FusionAudio-1.2M | 1.2M                                 | multimodal fusion (2025)           | fine-grained                             |
| LAION-Audio-630K | 633K (4,325 h)                       | web-harvested, 8 sources           | per-clip CC (some NC/attrib); links only |
| WavCaps          | 403K (~2,056 h)                      | GPT-from-metadata                  | Freesound/BBC/SoundBible/AudioSet        |
| AudioCaps        | ~49–57K                              | human, from AudioSet/YouTube       | clips can vanish                         |
| Clotho v2        | ~30K cap / ~5K audio                 | human, Freesound                   | **CC, permissive**                       |
| AudioSet         | 1.9M, **labels only**                | YouTube                            | keyword→caption to make text             |

### 8.2 Music ↔ text (data-poor — ~100× smaller than sound)

| Dataset        | Pairs             | Note                                    |
| -------------- | ----------------- | --------------------------------------- |
| MusicCaps      | ~5.5K             | expert captions; the standard           |
| MusicBench     | ~52K              | MusicCaps ×11 via LLM                   |
| LP-MusicCaps   | large             | LLM pseudo-captions                     |
| JamendoMaxCaps | large (2025)      | **Jamendo CC music** — good for release |
| Song Describer | ~1.1K / 706 songs | **permissive** — eval                   |

Strategy: synthetic-caption CC music (Jamendo/FMA) for scale; MusicCaps/Song
Describer for quality/eval.

### 8.3 Speech ↔ text (required for parity; don't fail MAEB speech)

- ASR transcripts (content): LibriSpeech, Common Voice, GigaSpeech.
- Multilingual: **FLEURS (156 langs; CC-BY)**, **Common Voice (CC0)** — also
  MAEB's retrieval backbone.
- Paralinguistic/emotion/speaker: emotion/speaker datasets or synthetic desc.
- Both notions needed: content-match (transcript) and acoustic-match
  (description), under the §6 instructions.

### 8.4 Cross-modal (audio↔image/video) — P3 grounding

- VGGSound, AudioSet, YouTube-8M are video-sourced → free audio↔video/image pairs.

### 8.5 Licensing for an Apache release

Prefer permissive audio (Clotho, Song Describer, Jamendo, FMA, permissive
Freesound, FLEURS, Common Voice) + **self-generated synthetic captions**. Treat
YouTube-derived audio sets as research/ablation fuel. The happy accident: the
**multilingual speech data you need (FLEURS CC-BY, Common Voice CC0) is already
permissive**, so full speech parity and a clean release are compatible.

### 8.6 Stage data mix

- **Stage 1 (~4–6M):** sound ~50% (AudioSetCaps + Auto-ACD + WavCaps + LAION) /
  speech ~30% (FLEURS + Common Voice + LibriSpeech) / music ~20% (LP-MusicCaps +
  Jamendo + MusicCaps). Dedup vs all eval sets.
- **Stage 2:** AudioCaps + Clotho + Song Describer + mined hard negatives.
- **Stage 3:** + heavy multilingual speech; + audio↔video pairs.

---

## 9. Evaluation

- **Regression guard (every stage):** MMEB-V2 subset on the base's
  text/image/video paths must be **unchanged** vs P0.
- **Audio target: MAEB** (Massive Audio Embedding Benchmark) — 98 tasks across
  speech/music/environmental sound, multilingual, retrieval/classification/
  reranking. **Watch the speech/sound/music balance** — contrastive audio-text
  models notoriously go near-random on multilingual speech; the parity goal is
  precisely to not be that model.
- **Audio-text retrieval:** AudioCaps + Clotho R@1/R@10 (A→T, T→A).
- **P4 head-to-head:** a **pre-registered all-five-modality benchmark** vs Gemini
  Embedding 2 — text/image/video/audio/PDF, cross-modal pairs, **with hard
  negatives** (MMEB omits them). Report on the wedge axes (cross-modal precision,
  all-modality coverage, MRL compression, self-hostability).

---

## 10. Repo state (`fusion_embedding/`)

| file              | role                                                                                          | status                                          |
| ----------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `config.py`       | `FusionConfig` — confirmed dims + Stage-1 hyperparams (`d_resampler = 256`)                   | done; tested                                    |
| `model.py`        | `FusionResampler` (§4.2 bottleneck applied) + `FusionEmbeddingModel` (injection, EOS pool, freeze) | done; tested; HF-API seams marked          |
| `losses.py`       | `FusionContrastiveLoss` — InfoNCE over MRL + learnable temp + CORAL (+ debias/hard-neg knobs) | done; tested                                    |
| `data.py`         | instruction taxonomy + `FusionAudioTextManifest` + `FusionCollator` + synthetic backend       | done; tested; `load_audio`/processor seams      |
| `train_stage1.py` | P1 connector-only loop + retrieval eval + `RegressionGuard`                                    | done; tested; `load_components`/MMEB seams      |
| `_tiny.py`        | tiny frozen CPU stand-ins implementing the frozen-base contract (below)                        | done; tested                                    |
| `demo_stage1.py`  | `python -m fusion_embedding.demo_stage1` — whole P1 loop end to end                            | done                                            |
| `README.md`       | scaffold map + exit gate                                                                       | done                                            |

**The bottleneck (§4.2) is implemented**, not pending: `FusionResampler` runs
`in_proj 1280→256`, latent queries and `L=6` blocks at 256, `out_proj 256→2048` +
LayerNorm → **7,198,464 params (~0.36%)**. Verified by `tests/test_resampler.py`.

**Frozen-base contract (the seam every real-model wiring must satisfy).** The model
takes the frozen base as three injected callables, so the exact same code runs on
tiny CPU stand-ins (`_tiny.py`) or the real Qwen towers:

```
embed_tokens : nn.Module   ids [B,S]                       -> embeds [B,S,2048]
base_lm      : callable     (inputs_embeds, attention_mask)  -> hidden [B,S,2048]
audio_encoder: callable     (mel, mel_mask)                  -> (frames [B,T,1280], frame_mask [B,T])
```

**Tests (`tests/`):** 56 staged E2E tests, CPU-only (no `transformers`/GPU) — config
invariants → resampler shapes/masking → audio injection & EOS pooling → loss
correctness (debias→InfoNCE reduction, MRL tiling, CORAL) → data/collator contract →
optimizer/guard → **full P1 loop: loss↓, retrieval R@1→~1.0, base byte-frozen.**

**Integration seams to wire (the `# TODO(fusion):` markers) — the real-model work:**

1. `load_components` — load `Qwen3-VL-Embedding-2B`, grab the LM that takes
   `inputs_embeds` + does EOS pooling; extract the Omni audio tower; add the
   `<|audio_pad|>` special token; return the three-callable contract above.
2. Audio injection (`model.inject_audio` / `encode_audio`) — confirm the base
   accepts `inputs_embeds` and that placeholder splicing matches its image-token path.
3. Audio encoder call (`model.audio_tokens`) — match the Omni encoder signature
   (mel + mask; 2 s blocks).
4. `load_audio` + `RealAudioProcessor` (`data.py`) — 16 kHz mono + Omni feature
   extractor inputs.
5. `evaluate` — AudioCaps/Clotho R@1 **and** the MMEB-V2 regression guard (the
   param-level `RegressionGuard` already enforces base-frozen every run).

---

## 11. Risk register & live decisions

**Risks**

- **Speech weakness** (contrastive audio-text → near-random on speech): the #1
  risk to the parity claim. Mitigations: speech-capable encoder, multilingual
  speech data, heavy MAEB-speech weighting; lever = dual-encoder fallback.
- **Music scarcity:** synthetic-caption CC music.
- **Audio modality gap:** e5-omni alignment (temp/whitening/CORAL).
- **One-to-many false negatives:** debiased contrastive.
- **Encoder license** (Apache target): verify the Omni audio-encoder license
  early; fall back to Whisper-MIT + permissive general-audio encoder.
- **Base leakage:** keep base frozen; MMEB-V2 guard each run.
- **Long/variable audio:** chunk-and-fuse, fixed token budget.

**Live decisions**

- Connector `N`: 64 (P1) → 128–200 (P3 speech). Tune on MAEB speech.
- Dual-encoder trigger: pull only if P3 shows the single encoder weak on speech
  _or_ sound.
- Audio-encoder LoRA: default off; tiny LoRA only if frozen underfits.
- 8B output dim: ~4096 (confirm from its config).

---

## 12. References

**Models**

- Qwen3-VL-Embedding / Reranker — arXiv 2601.04720 (`Qwen/Qwen3-VL-Embedding-2B`,
  `-8B`); MMEB-V2 SOTA; dual-tower, EOS pooling, MRL, QAT.
- Qwen2.5-Omni — arXiv (tech report 2503.xxxxx); audio tower Whisper-large-v3-
  derived, d=1280, ~25 fps, 2 s blocks (`Qwen/Qwen2.5-Omni-7B`).
- jina-embeddings-v5-omni — arXiv 2605.08384; frozen-tower composition
  (SigLIP2 + Whisper onto a frozen text backbone, ~0.35% trained); the precedent.
- e5-omni — arXiv 2601.03666; explicit cross-modal alignment (modality-aware
  temperature, debiased negatives, batch whitening / CORAL). The alignment recipe.
- ImageBind — arXiv 2305.05665; bind every modality to an anchor → emergent
  alignment without all-pairs data. The "text is the bridge" justification.
- Gemini Embedding 2 — the proprietary target (5 modalities, 3072-dim, MRL).

**Method / theory**

- Matryoshka Representation Learning — Kusupati et al. 2022.
- On the Theoretical Limitations of Embedding-Based Retrieval (the sign-rank
  bottleneck) — Weller et al., arXiv 2508.21038.
- Uni-MoE (200 query tokens / 30 s captures paralinguistics) — arXiv 2511.12609.

**Data**

- AudioSetCaps — arXiv 2411.18953 (6M pairs, open pipeline).
- Auto-ACD (1.5M), Sound-VECaps (arXiv 2407.04416, 1.66M), WavCaps (403K),
  LAION-Audio-630K (arXiv 2211.06687), AudioCaps, Clotho.
- Music: MusicCaps, MusicBench, LP-MusicCaps, JamendoMaxCaps (arXiv 2502.07461),
  Song Describer.
- Speech: FLEURS (CC-BY), Common Voice (CC0), LibriSpeech.

**Benchmarks**

- MAEB — Massive Audio Embedding Benchmark, arXiv 2602.16008 (98 tasks;
  speech/music/sound; multilingual).
- MMEB-V2 — multimodal embedding benchmark (the regression guard + image/video).

---

_Build order: P0 (stand up base + harness) → wire the §10 seams → P1 (train
connector) → P2 → P3 → P4, then repeat P1–P3 on 8B. The contribution is
execution — frozen-base composition + cross-modal alignment + data — not a new
architecture._
