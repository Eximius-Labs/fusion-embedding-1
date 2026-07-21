"""Package Ember (the thermal sense pack, release seed 2) and upload to HF.

Converts /vol/checkpoints/thermal_release_seed2.pt (pulled beforehand to
thermal_pack_release/out/_raw_seed2.pt) to safetensors, writes the pack config,
the model card, and the FLIR strip list, then uploads everything to
EximiusLabs/fusion-embedding-2-ember.

Run: PYTHONUTF8=1 uv run --env-file .env --with huggingface_hub,safetensors python scripts/package_thermal_pack.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import torch

OUT = "thermal_pack_release/out"
REPO = "EximiusLabs/fusion-embedding-2-ember"
FE2 = "EximiusLabs/fusion-embedding-2-2b-preview"
FE2_REV = "v0.2-preview"
BASE = "Qwen/Qwen3-VL-Embedding-2B"

CARD = """---
license: cc-by-nc-4.0
pipeline_tag: feature-extraction
base_model: {fe2}
tags:
- embeddings
- retrieval
- multimodal
- thermal
- infrared
- adapters
---

# Ember — the thermal sense for fusion-embedding-2

Ember is the first sense pack for [fusion-embedding-2]({fe2_url}): it teaches the
model to embed thermal infrared images in the same vector space as its text,
image, video, and audio embeddings. Packs are named for the physical trace their
sensor reads; Ember reads heat.

Ember is strictly additive. Technically it is a 44M-parameter gated adapter pack
that attaches to the frozen decoder behind a thermal-only gate: when the gate is
closed (every non-thermal input), the model's outputs are bit-for-bit identical
to the model without the pack. This is verified, not aspirational; see
Correctness below.

![Ember architecture overview](assets/fe2_ember_overview.png)

## What it does

- Thermal image to text retrieval: R@10 0.785 on a held-out 2,000-caption gallery
  (frozen base: 0.224).
- Thermal zero-shot classification is preserved: LLVIP person/background 94.3
  (calibrated ensemble harness; frozen base reference 95.4).
- Cross-domain thermal-to-visible retrieval: R@10 0.348 on LLVIP registered pairs,
  above the frozen baseline 0.165, without training on any LLVIP data.
- Text, RGB image, video, and audio embeddings unchanged, bit-for-bit.

## Usage

Ember loads as an adapter pack through the multi-gate adapter registry in the
[fusion-embedding GitHub repository](https://github.com/Eximius-Labs/fusion-embedding)
(`fusion_embedding/adapters.py`). Thermal images are single-channel; replicate to
three channels and encode through the ordinary image path with the thermal scope open.

```python
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from fusion_embedding.adapters import AdapterPacks
from inference import FusionEmbedder          # fe2_release/inference.py (GitHub repo)

emb = FusionEmbedder.from_pretrained(
    "{fe2}", revision="{fe2_rev}", device="cuda")

packs = AdapterPacks()
adapters, gate = packs.add_pack("thermal", emb.model.base_lm, 2048, rank=384)
adapters.load_state_dict(load_file(hf_hub_download(
    "{repo}", "model.safetensors")))
packs.to("cuda")

# thermal encode: thermal readout template, thermal scope open (the scope
# spans forward and backward)
import torch.nn.functional as F
text = ("<|im_start|>system\\nRepresent this thermal infrared image.<|im_end|>\\n"
        "<|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\\n"
        "<|im_start|>assistant\\n")
inp = emb.proc(text=[text], images=[thermal_image_3ch], return_tensors="pt").to("cuda")
with torch.no_grad(), packs.scope("thermal"):
    h = emb.full(**inp).last_hidden_state
thermal_vec = F.normalize(h[0, inp["attention_mask"][0].sum() - 1].float(), dim=-1)

# everything else: leave the scope closed; outputs equal the pack-free model exactly
text_vec = emb.embed_text("a person crossing a dark road")
audio_vec = emb.embed_audio(wav, sr=16000)    # audio pack co-loaded, unaffected
```

The pack attaches equally to the raw base (`AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-2B")`,
attach at `.language_model`), which is the exact configuration it was trained in;
both loading paths are verified bitwise in the release smoke. The readout is the
standard fusion-embedding protocol: chat template, last non-pad token pooling,
L2 normalization (see `config.json` for the exact templates and the trained
temperature).

## Evaluation

Training: contrastive thermal-to-caption alignment on IR-TD (61,320 pairs after
FLIR exclusion and eval dedup), 3,900 steps, batch 16, 1,024 bank negatives,
frozen base, bf16 base precision with fp32 adapters. Three seeds; seed 2 shipped.

| release run (61K corpus) | holdout t2t R@10 | delta vs frozen | LLVIP-ZS | LLVIP twin R@10 |
|---|---|---|---|---|
| frozen base | 0.224 | - | 95.4 | 0.165 |
| seed 1 | 0.783 | +0.560 | 94.1 | 0.333 |
| **seed 2 (shipped)** | **0.785** | **+0.561** | **94.3** | **0.348** |
| seed 3 | 0.777 | +0.554 | 91.8 | 0.341 |

Seed 2 holdout detail: R@1 0.412, R@5 0.692, R@10 0.785 over a 2,000-item gallery.

Text to thermal retrieval on the release holdout (queries shortened for display;
retrieval used the full captions):

![Ember retrieval gallery](assets/fe2_ember_retrieval_gallery.png)

A caption-style ablation on the pre-exclusion corpus (82K pairs, 5,000 steps)
found that caption richness is a generalization lever, not just an in-domain fit
lever: training on full descriptive captions reached holdout R@10 0.843 and LLVIP
twin 0.343, while first-sentence captions reached 0.594 and collapsed cross-domain
transfer to 0.089, below the frozen baseline. Ember ships the full-caption arm.

Domain note: IR-TD spans 63 source collections but is still a finite domain mix.
The LLVIP numbers above are cross-domain signal (night pedestrian scenes never
seen in training), not a claim of parity with in-domain retrieval. Expect the gap
to vary with distance from the training domains.

## Correctness

The bit-for-bit preservation claim is tested at three levels:

1. Unit suite (GitHub repo, `tests/test_thermal_adapters.py`): closed-gate
   forwards equal the base exactly; gradients reach only the open pack; the gate
   must span forward and backward under gradient checkpointing.
2. Release checks with the trained weights: RGB-image and text forwards
   bit-for-bit equal to the pack-free base with the thermal gate closed, per seed.
3. Composability matrix (audio pack + thermal pack co-loaded on the same frozen
   decoder): audio through the registry vs the shipped single-gate path, audio
   co-loaded vs audio-only, thermal co-loaded vs thermal-only, and text / RGB /
   video vs the raw base, all bitwise; retrieval scores identical under co-load.

Mixed inputs that would open two gates in a single forward are outside the
guarantee and are not tested.

## Provenance

- Training corpus: IR-TD early access (IRGPT, ICCV 2025,
  [arXiv:2507.14449](https://arxiv.org/abs/2507.14449),
  [repository](https://github.com/WheatCao/ICCV2025-IRGPT)); 84,284 real thermal
  images with LLM-generated descriptive captions; academic research use only.
- FLIR exclusion: IR-TD includes FLIR-derived sources whose terms restrict
  redistribution of trained weights. All 20,964 images matching the FLIR capture
  signature (640x512) were excluded from training and holdout. This is a
  size-based heuristic, not an author-provided source mapping; the exclusion list
  ships in this repository (`release_strip_640x512.json`, sha256 `{strip_sha}`).
- LLVIP is used for evaluation only (zero-shot gate and cross-domain retrieval).
- Eval hygiene: perceptual-hash dedup between the training set and the LLVIP test
  set found 0 collisions (hamming distance <= 4).

## License

The Ember weights in this repository are released under CC-BY-NC-4.0 for research
use, reflecting the academic-use terms of the training corpus. The core
fusion-embedding-2 model is a separate artifact under its own license; this pack
is optional and separable, and does not modify the core model's weights.
"""


def main() -> None:
    from huggingface_hub import HfApi
    from safetensors.torch import save_file

    ck = torch.load(os.path.join(OUT, "_raw_seed2.pt"), map_location="cpu",
                    weights_only=False)
    sd = ck["thermal_adapters"]
    assert len(sd) == 112 and ck["seed"] == 2 and ck["rank"] == 384
    save_file(sd, os.path.join(OUT, "model.safetensors"),
              metadata={"format": "pt", "pack": "thermal", "rank": "384",
                        "seed": str(ck["seed"]), "steps": str(ck["step"])})

    strip_sha = hashlib.sha256(
        open(os.path.join(OUT, "release_strip_640x512.json"), "rb").read()).hexdigest()

    config = {
        "artifact": "fusion-embedding-2-ember",
        "version": "0.1-preview",
        "pack": {
            "gate": "thermal",
            "rank": 384,
            "adapters": 28,
            "d_model": 2048,
            "parameters": int(sum(v.numel() for v in sd.values())),
            "dtype": "float32",
            "init": "zero-initialized up projection (identity at attach time)",
            "attach_point": "decoder layers (language_model.layers), forward hooks",
        },
        "attaches_to": {
            "base": BASE,
            "fe2": FE2,
            "fe2_revision": FE2_REV,
            "note": "the decoder is byte-frozen and shared between the base and "
                    "fusion-embedding-2, so the pack attaches to either",
        },
        "readout": {
            "thermal_instruction": ck["instruction"],
            "doc_instruction": "Represent the user's input.",
            "chat_template": "<|im_start|>system\\n{instruction}<|im_end|>\\n"
                             "<|im_start|>user\\n{input}<|im_end|>\\n"
                             "<|im_start|>assistant\\n",
            "image_user_content": "<|vision_start|><|image_pad|><|vision_end|>",
            "pooling": "last non-pad token, L2-normalized",
            "thermal_input": "single-channel thermal replicated to 3 channels, "
                             "vision path, max 1310720 pixels",
            "trained_logit_scale": float(ck["logit_scale"]),
        },
        "training": {
            "objective": "contrastive thermal->caption (InfoNCE, 1024 bank negatives)",
            "corpus": "IR-TD early access minus FLIR-signature images",
            "train_pairs": ck["train_n"],
            "steps": ck["step"],
            "batch": 16,
            "seed": ck["seed"],
        },
        "provenance": {
            "flir_strip": {"rule": "size == 640x512", "excluded": 20964,
                           "of": 84284, "list": "release_strip_640x512.json",
                           "sha256": strip_sha},
            "llvip": "evaluation only",
            "dedup": "phash hamming<=4 vs LLVIP test: 0 hits",
        },
    }
    with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    card = CARD.format(fe2=FE2, fe2_url=f"https://huggingface.co/{FE2}",
                       fe2_rev=FE2_REV, repo=REPO, strip_sha=strip_sha)
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    info = api.upload_folder(
        repo_id=REPO, folder_path=OUT,
        ignore_patterns=["_raw_seed2.pt"],
        commit_message="thermal pack v0.1-preview: seed-2 release adapters, config, card, FLIR strip list",
    )
    print("uploaded:", info.commit_url)
    sha = api.model_info(REPO).sha
    print("revision:", sha)
    api.create_tag(REPO, tag="v0.1-preview", revision=sha, exist_ok=True)
    print("tagged v0.1-preview")


if __name__ == "__main__":
    main()
