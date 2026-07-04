"""Modal deployment for Fusion Embedding 1 — serverless GPU training on Linux.

Why Modal (recap): Linux containers make 4-bit (bitsandbytes) + FlashAttention "just
work" (no Windows/WSL pain); per-second billing fits the burst-y, connector-only stages;
a Volume caches the frozen Qwen weights + preprocessed mel features so the H100 is never
blocked on downloads or audio decode. The exact same `fusion_embedding` code runs here
and locally — Modal only provides the GPU + filesystem.

Pipeline (run in order):
    uv run --env-file .env modal run modal_app.py::smoke         # cheap T4 — prove the image + GPU path
    uv run --env-file .env modal run modal_app.py::warm_cache    # one-time: pull Qwen weights to the Volume
    uv run --env-file .env modal run modal_app.py::preprocess --shard demo   # CPU: audio -> mel on the Volume
    uv run --env-file .env modal run modal_app.py::train_p1      # GPU: connector training (Stage 1)

Status: image + Volume + Secret + the tiny-stand-in GPU smoke are REAL and runnable today.
`preprocess` and `train_p1` carry `# TODO(fusion)` markers where the real dataset and the
HLD §10 `load_components` seam plug in — they run end-to-end on synthetic data until then.
"""

from __future__ import annotations

import modal

APP_NAME = "fusion-embedding"

# --- Persistent storage: one Volume holds the HF weight cache, preprocessed features,
#     and connector checkpoints. Created on first use; survives across runs. ---
volume = modal.Volume.from_name("fusion-data", create_if_missing=True)
VOL = "/vol"
HF_CACHE = f"{VOL}/hf-cache"        # frozen Qwen weights (downloaded once)
FEATURES = f"{VOL}/features"        # preprocessed mel shards (WebDataset/pt)
CKPTS = f"{VOL}/checkpoints"        # connector + temperature checkpoints (~30MB each)

# --- HF token: stored in Modal's secret store as `huggingface` (you create it; see below).
#     Referenced by name — the token never appears in this file or the repo. ---
hf_secret = modal.Secret.from_name("huggingface")

# --- Container image: cu124 torch + the `hf` extra (transformers/bitsandbytes/librosa).
#     Built once and cached by Modal; bitsandbytes/flash-attn build cleanly on Linux. ---
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1",              # audio decode backends for librosa/soundfile
                 "zip", "unzip", "p7zip-full")         # WavCaps multi-part zip reassembly
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.24",
        "transformers>=4.46",
        "accelerate>=0.30",
        "bitsandbytes>=0.43",                          # 4-bit frozen base (Linux only)
        "soundfile>=0.12",
        "librosa>=0.10",
        "datasets>=3.0",                               # real audio-caption datasets
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # Ship the local package into the image so `import fusion_embedding` works in the container.
    .add_local_python_source("fusion_embedding")
)

app = modal.App(APP_NAME, image=image)

# Shared env so every function caches HF downloads onto the Volume, not the ephemeral disk.
# HF_XET_HIGH_PERFORMANCE is the current fast-transfer backend (Xet); the old
# HF_HUB_ENABLE_HF_TRANSFER flag is deprecated (hf_transfer is no longer used). hf_xet ships
# in the image, so this actually enables faster weight/dataset pulls.
# FUSION_DATA_ROOT makes fusion_embedding.paths resolve features/frames/checkpoints to the
# Volume here — and to a local dir / mounted bucket off Modal. One env var = provider-portable.
HF_ENV = {"HF_HOME": HF_CACHE, "HF_XET_HIGH_PERFORMANCE": "1", "FUSION_DATA_ROOT": VOL,
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

BASE_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
AUDIO_MODEL = "Qwen/Qwen2.5-Omni-7B"


# --------------------------------------------------------------------------- #
# 0. Smoke — cheapest possible proof the image, GPU, and our code line up.
#    No weights, no data: runs the tiny CPU stand-ins on a real Modal GPU.
# --------------------------------------------------------------------------- #
@app.function(gpu="T4", timeout=600)
def smoke() -> dict:
    import torch
    from fusion_embedding.config import FusionConfig
    from fusion_embedding.train_stage1 import build_tiny_training_setup, train_stage1
    from fusion_embedding.memory_bank import TextMemoryBank

    assert torch.cuda.is_available(), "no CUDA in the Modal container"
    dev = "cuda"
    cfg = FusionConfig.tiny(max_steps=120, d_resampler=32, use_bf16=True)
    s = build_tiny_training_setup(cfg, n_train=8, batch_size=8, seed=0)
    s.model.to(dev)
    bank = TextMemoryBank(dim=cfg.d_llm, capacity=32, device=dev)
    state = train_stage1(
        s.model, s.train_loader, s.loss_fn, cfg,
        steps=120, eval_fn=s.eval_fn, device=dev, log_every=60, memory_bank=bank,
    )
    out = {
        "gpu": torch.cuda.get_device_name(0),
        "a2t_R@1": state.final_eval["a2t_R@1"],
        "base_drift": state.final_eval["base_drift"],
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    }
    print("SMOKE:", out)
    return out


# --------------------------------------------------------------------------- #
# 0b. introspect — discover the REAL Qwen APIs before writing load_components.
#     CPU-only (no GPU cost): loads the cached configs + the small 2B model, and
#     builds the 7B on the meta device (zero memory) to read its module tree.
#     Prints exactly what the frozen-base contract needs: embed_tokens path,
#     inputs_embeds support, hidden size, audio-tower path + forward signature,
#     d_audio, and the tokenizer special-token ids.
# --------------------------------------------------------------------------- #
@app.function(volumes={VOL: volume}, secrets=[hf_secret], cpu=4.0, memory=16384, timeout=1800, env=HF_ENV)
def introspect() -> dict:
    import inspect
    import json

    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    report: dict = {}

    def safe(fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - discovery: capture, don't crash
            return f"ERR: {type(e).__name__}: {e}"

    # ---- Base: Qwen3-VL-Embedding-2B ----
    b: dict = {}
    cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    b["config_class"] = type(cfg).__name__
    b["hidden_size"] = getattr(cfg, "hidden_size", None) or safe(lambda: cfg.text_config.hidden_size)
    b["top_level_keys"] = [k for k in vars(cfg) if not k.startswith("_")][:40]

    tok = safe(lambda: AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True))
    if not isinstance(tok, str):
        b["eos_id"] = tok.eos_token_id
        b["pad_id"] = tok.pad_token_id
        b["eos_token"] = tok.eos_token
        b["has_audio_pad"] = "<|audio_pad|>" in tok.get_vocab()
        b["vocab_size"] = len(tok)
        # candidate placeholder tokens already in the vocab
        b["pad_like_tokens"] = [t for t in tok.get_vocab() if "pad" in t.lower() or "audio" in t.lower()][:20]

    def load_base():
        m = AutoModel.from_pretrained(
            BASE_MODEL, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        return m

    m = safe(load_base)
    if not isinstance(m, str):
        b["model_class"] = type(m).__name__
        b["forward_params"] = list(inspect.signature(m.forward).parameters)
        b["accepts_inputs_embeds"] = "inputs_embeds" in b["forward_params"]
        # find embed_tokens-like modules
        b["embed_modules"] = [n for n, _ in m.named_modules() if n.endswith("embed_tokens")][:5]
        # top-level children (orientation in the module tree)
        b["top_children"] = [n for n, _ in m.named_children()]
        b["get_input_embeddings"] = safe(lambda: type(m.get_input_embeddings()).__name__)
    report["base"] = b

    # ---- Audio: Qwen2.5-Omni-7B (meta device — no weight memory) ----
    a: dict = {}
    acfg = AutoConfig.from_pretrained(AUDIO_MODEL, trust_remote_code=True)
    a["config_class"] = type(acfg).__name__
    a["sub_configs"] = [k for k in vars(acfg) if not k.startswith("_")][:40]
    # hunt for the audio sub-config + its hidden dim (d_audio should be 1280)
    for key in vars(acfg):
        sub = getattr(acfg, key)
        if hasattr(sub, "to_dict") and "audio" in key.lower():
            a[f"audio_cfg::{key}"] = {
                kk: vv for kk, vv in sub.to_dict().items()
                if any(s in kk for s in ("hidden", "d_model", "num_mel", "layers", "dim"))
            }

    def build_meta():
        from accelerate import init_empty_weights

        with init_empty_weights():
            mm = AutoModel.from_config(acfg, trust_remote_code=True)
        return mm

    am = safe(build_meta)
    if not isinstance(am, str):
        a["model_class"] = type(am).__name__
        a["top_children"] = [n for n, _ in am.named_children()]
        # find the audio tower/encoder submodule by name
        audio_mods = [n for n, _ in am.named_modules() if ("audio" in n.lower() and n.count(".") <= 1)]
        a["audio_module_names"] = audio_mods[:15]
        for name, mod in am.named_modules():
            if name in audio_mods and hasattr(mod, "forward"):
                a[f"forward::{name}"] = list(inspect.signature(mod.forward).parameters)
    else:
        a["meta_build"] = am

    # feature extractor / processor (cheap, real)
    a["processor"] = safe(
        lambda: type(
            __import__("transformers").AutoProcessor.from_pretrained(AUDIO_MODEL, trust_remote_code=True)
        ).__name__
    )
    report["audio"] = a

    print("INTROSPECT REPORT:\n" + json.dumps(report, indent=2, default=str))
    return report


# --------------------------------------------------------------------------- #
# 0c. introspect_audio — targeted dig into the Omni thinker's audio tower.
# --------------------------------------------------------------------------- #
@app.function(volumes={VOL: volume}, secrets=[hf_secret], cpu=4.0, memory=16384, timeout=1800, env=HF_ENV)
def introspect_audio() -> dict:
    import inspect
    import json

    import torch
    import transformers
    from transformers import AutoConfig

    report: dict = {}

    def safe(fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            return f"ERR: {type(e).__name__}: {e}"

    # confirm base d_llm
    bcfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    report["base_hidden"] = safe(lambda: bcfg.text_config.hidden_size)
    report["base_text_cfg_class"] = safe(lambda: type(bcfg.text_config).__name__)

    acfg = AutoConfig.from_pretrained(AUDIO_MODEL, trust_remote_code=True)
    th = acfg.thinker_config
    report["thinker_cfg_class"] = type(th).__name__
    report["thinker_sub"] = [k for k in vars(th) if not k.startswith("_")][:40]
    audio_cfg = safe(lambda: th.audio_config)
    if not isinstance(audio_cfg, str):
        report["audio_config"] = {k: v for k, v in audio_cfg.to_dict().items()
                                  if any(s in k for s in ("hidden", "d_model", "mel", "layer", "dim", "output", "size"))}
        report["audio_config_class"] = type(audio_cfg).__name__

    # what Omni classes exist in this transformers build?
    report["omni_classes"] = [n for n in dir(transformers) if "Omni" in n][:30]

    # build the audio encoder alone, on meta (zero memory), to read its forward + tree
    def build_audio_encoder():
        from accelerate import init_empty_weights
        # the audio tower class
        cls = None
        for name in ("Qwen2_5OmniAudioEncoder",):
            cls = getattr(transformers, name, None)
            if cls is not None:
                break
        if cls is None:
            return "no audio encoder class found"
        with init_empty_weights():
            enc = cls(audio_cfg)
        return enc

    enc = safe(build_audio_encoder)
    if not isinstance(enc, str):
        report["audio_encoder_class"] = type(enc).__name__
        report["audio_encoder_forward"] = list(inspect.signature(enc.forward).parameters)
        report["audio_encoder_children"] = [n for n, _ in enc.named_children()]
    else:
        report["audio_encoder_build"] = enc

    # feature extractor (how mel is produced)
    def load_feat():
        from transformers import Qwen2_5OmniProcessor
        proc = Qwen2_5OmniProcessor.from_pretrained(AUDIO_MODEL, trust_remote_code=True)
        fe = getattr(proc, "feature_extractor", None) or getattr(proc, "omni_processor", None)
        return {
            "processor_class": type(proc).__name__,
            "feature_extractor_class": type(fe).__name__ if fe is not None else None,
            "fe_sampling_rate": getattr(fe, "sampling_rate", None),
            "fe_n_mels": getattr(fe, "feature_size", None),
            "fe_call_params": list(inspect.signature(fe.__call__).parameters) if fe is not None else None,
        }

    report["feature_extractor"] = safe(load_feat)

    print("AUDIO INTROSPECT:\n" + json.dumps(report, indent=2, default=str))
    return report


# --------------------------------------------------------------------------- #
# 0d. introspect_encoder — read the Omni audio encoder's forward + load the FE.
# --------------------------------------------------------------------------- #
@app.function(volumes={VOL: volume}, secrets=[hf_secret], cpu=4.0, memory=16384, timeout=1800, env=HF_ENV)
def introspect_encoder() -> dict:
    import inspect
    import json

    from transformers import AutoConfig
    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as mod

    report: dict = {}

    def safe(fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            return f"ERR: {type(e).__name__}: {e}"

    acfg = AutoConfig.from_pretrained(AUDIO_MODEL, trust_remote_code=True)
    audio_cfg = acfg.thinker_config.audio_config

    enc_cls = mod.Qwen2_5OmniAudioEncoder
    report["encoder_class"] = enc_cls.__name__
    report["encoder_forward_params"] = list(inspect.signature(enc_cls.forward).parameters)
    # the source tells us inputs, masking, and what it returns (1280 vs 3584)
    src = inspect.getsource(enc_cls.forward)
    report["encoder_forward_source"] = src[:3500]

    # feature extractor — load directly (the full processor trips on the image side)
    def load_fe():
        from transformers import AutoFeatureExtractor

        fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL, trust_remote_code=True)
        return {
            "class": type(fe).__name__,
            "sampling_rate": getattr(fe, "sampling_rate", None),
            "feature_size": getattr(fe, "feature_size", None),
            "n_mels": getattr(fe, "feature_size", None),
            "call_params": list(inspect.signature(fe.__call__).parameters),
            "hop_length": getattr(fe, "hop_length", None),
            "chunk_length": getattr(fe, "chunk_length", None),
        }

    report["feature_extractor"] = safe(load_fe)
    print("ENCODER INTROSPECT:\n" + json.dumps(report, indent=2, default=str))
    return report


# --------------------------------------------------------------------------- #
# 0e. find_audio_dataset — STREAMING peek at candidate caption datasets (no full
#     download): which load, their columns, and a sample caption. Picks the repo
#     + columns for `preprocess` without burning full-download runs.
# --------------------------------------------------------------------------- #
@app.function(secrets=[hf_secret], cpu=2.0, timeout=900, env=HF_ENV)
def find_audio_dataset() -> dict:
    import os
    from datasets import load_dataset

    from datasets import Audio

    token = os.environ.get("HF_TOKEN") or None
    candidates = [
        ("OpenSound/AudioCaps", "train"),
        ("OpenSound/AudioCaps", "test"),
    ]
    report = {}
    for repo, split in candidates:
        try:
            ds = load_dataset(repo, split=split, streaming=True, token=token)
            feats = list(ds.features) if ds.features else None
            if feats and "audio" in feats:                 # avoid torchcodec on peek
                ds = ds.cast_column("audio", Audio(decode=False))
            sample = next(iter(ds))
            # show non-audio fields (strings) so we can spot caption columns
            text_fields = {k: (str(v)[:80]) for k, v in sample.items()
                           if isinstance(v, (str, int, float, list)) and k != "audio"}
            report[repo] = {"split": split, "features": feats, "sample_text_fields": text_fields}
            print(f"OK  {repo}:{split}  features={feats}")
        except Exception as e:  # noqa: BLE001
            report[repo] = f"ERR: {type(e).__name__}: {str(e)[:160]}"
            print(f"FAIL {repo}: {str(e)[:120]}")
    import json
    print("DATASET REPORT:\n" + json.dumps(report, indent=2, default=str))
    return report


@app.function(secrets=[hf_secret], cpu=2.0, timeout=900, env=HF_ENV)
def peek_eval() -> dict:
    """Peek the eval sets for Step 1: how AudioCaps-test / Clotho structure their 5 captions/clip
    (one row per caption sharing a clip id, or a list column) + which repo ships decodable audio."""
    import json
    import os
    from itertools import islice

    from datasets import Audio, load_dataset

    token = os.environ.get("HF_TOKEN") or None
    candidates = [
        ("OpenSound/AudioCaps", None, "test"),
        ("confit/clotho", "2023", "test"),
        ("CLAPv2/Clotho", None, "test"),
    ]
    report = {}
    for repo, cfg, split in candidates:
        key = f"{repo}:{cfg}" if cfg else repo
        try:
            ds = (load_dataset(repo, cfg, split=split, streaming=True, token=token) if cfg
                  else load_dataset(repo, split=split, streaming=True, token=token))
            feats = list(ds.features) if ds.features else None
            if feats and "audio" in feats:
                ds = ds.cast_column("audio", Audio(decode=False))
            rows = []
            for r in islice(ds, 6):
                rows.append({k: str(v)[:70] for k, v in r.items()
                             if isinstance(v, (str, int, float, list)) and k != "audio"})
            report[key] = {"split": split, "features": feats, "first_rows": rows}
            print(f"OK  {key}  features={feats}")
        except Exception as e:  # noqa: BLE001
            report[key] = f"ERR: {type(e).__name__}: {str(e)[:180]}"
            print(f"FAIL {key}: {str(e)[:150]}")
    print("EVAL PEEK:\n" + json.dumps(report, indent=2, default=str))
    return report


@app.function(secrets=[hf_secret], cpu=2.0, timeout=1800, env=HF_ENV)
def find_clotho_5ref() -> dict:
    """Discover a Clotho source that carries all 5 reference captions (for the published min-rank-over-5
    A→T protocol). Searches HF for clotho datasets, then peeks each candidate's schema + first row —
    flags repos with caption_1..caption_5 columns, a captions/all_captions list, or per-caption rows
    groupable by an audio/file key. Cheap CPU probe; informs the 5-ref ingestion path."""
    import json
    import os
    from itertools import islice

    from datasets import load_dataset
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi()
    names = [d.id for d in api.list_datasets(search="clotho", limit=40, token=token)]
    print(f"candidates ({len(names)}): {names}")

    def _caption_cols(feats):
        return [f for f in feats if "caption" in f.lower() or f.lower() in ("text", "captions", "raw_text")]

    report = {"candidates": names, "schemas": {}}
    for repo in names:
        for split in ("test", "evaluation", "validation"):
            try:
                ds = load_dataset(repo, split=split, streaming=True, token=token)
                feats = list(ds.features) if ds.features else None
                row0 = next(islice(ds, 1), None)
                sample = {k: str(v)[:80] for k, v in (row0 or {}).items() if k != "audio"}
                report["schemas"][f"{repo}:{split}"] = {
                    "features": feats, "caption_like": _caption_cols(feats or []),
                    "has_audio": bool(feats and "audio" in feats), "sample": sample}
                print(f"OK  {repo}:{split}  caption_like={_caption_cols(feats or [])}  audio={'audio' in (feats or [])}")
                break                                            # first working split is enough
            except Exception as e:                               # noqa: BLE001
                report["schemas"].setdefault(f"{repo}:{split}", f"ERR: {type(e).__name__}: {str(e)[:90]}")
    print("CLOTHO 5REF SEARCH:\n" + json.dumps(report, indent=2, default=str))
    return report


@app.function(secrets=[hf_secret], cpu=2.0, timeout=1800, env=HF_ENV)
def list_clotho_files(repos: str = "confit/clotho,d0rj/clotho-v2.1,ZhangShiao/clotho") -> dict:
    """List raw files in candidate Clotho repos (bypasses dead loader scripts) — looking for an
    `..._captions_evaluation.csv` (file_name,caption_1..5) + evaluation audio we can build 5-ref from."""
    import json
    import os

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi()
    out = {}
    for repo in [r.strip() for r in repos.split(",") if r.strip()]:
        try:
            files = api.list_repo_files(repo, repo_type="dataset", token=token)
            interesting = [f for f in files if any(k in f.lower() for k in
                           ("eval", "caption", ".csv", ".7z", ".zip", ".parquet", "audio"))]
            out[repo] = {"n_files": len(files), "interesting": interesting[:60]}
            print(f"OK {repo}: {len(files)} files")
        except Exception as e:                                   # noqa: BLE001
            out[repo] = f"ERR: {type(e).__name__}: {str(e)[:100]}"
            print(f"FAIL {repo}: {str(e)[:100]}")
    print("CLOTHO FILES:\n" + json.dumps(out, indent=2, default=str))
    return out


@app.function(secrets=[hf_secret], cpu=2.0, timeout=1800, env=HF_ENV)
def verify_clotho_5ref(specs: str = "LakoreAI/clotho-dev-sample:test:caption_1|caption_2|caption_3|caption_4|caption_5;mteb/Clotho:test:text|raw_text") -> dict:
    """For each `repo:split:col1|col2|...` spec, count rows + unique clips + how many carry all listed
    caption fields non-empty, and dump one full row. Decides which source = canonical 1045×5 Clotho eval."""
    import json
    import os

    from datasets import Audio, load_dataset

    token = os.environ.get("HF_TOKEN") or None
    out = {}
    for spec in [s for s in specs.split(";") if s.strip()]:
        repo, split, cols = spec.split(":", 2)
        cols = cols.split("|")
        try:
            ds = load_dataset(repo, split=split, streaming=True, token=token)
            if ds.features and "audio" in ds.features:
                ds = ds.cast_column("audio", Audio(decode=False))
            n = full5 = 0
            names = set()
            first = None
            for r in ds:
                n += 1
                if first is None:
                    first = {k: (str(v)[:60] if k != "audio" else "<audio>") for k, v in r.items()}
                fn = r.get("file_name") or r.get("index") or r.get("id")
                if fn is not None:
                    names.add(str(fn))
                vals = [r.get(c) for c in cols]
                if all(isinstance(v, str) and v.strip() for v in vals):
                    full5 += 1
                if n >= 12000:
                    break
            out[f"{repo}:{split}"] = {"rows": n, "unique_clip_keys": len(names),
                                      "rows_with_all_cols_nonempty": full5, "cols": cols, "first_row": first}
            print(f"{repo}:{split}  rows={n}  unique={len(names)}  all_cols_nonempty={full5}")
        except Exception as e:                                   # noqa: BLE001
            out[f"{repo}:{split}"] = f"ERR: {type(e).__name__}: {str(e)[:120]}"
            print(f"FAIL {repo}:{split}: {str(e)[:110]}")
    print("VERIFY CLOTHO:\n" + json.dumps(out, indent=2, default=str))
    return out


@app.function(secrets=[hf_secret], cpu=2.0, timeout=1800, env=HF_ENV)
def peek_clotho_candidates(
    repos: str = "mteb/Clotho,zachz/Clotho-PC-T2A,LakoreAI/clotho-dev-sample,humanify/ARAG-clotho-test",
    split: str = "test") -> dict:
    """Probe audio-bearing Clotho repos with decode OFF (so schema reads don't trip torchcodec):
    report features + a sample row + whether a clip carries all 5 refs (caption_1..5 / a list / groupable)."""
    import json
    import os
    from itertools import islice

    from datasets import Audio, load_dataset

    token = os.environ.get("HF_TOKEN") or None
    out = {}
    for repo in [r.strip() for r in repos.split(",") if r.strip()]:
        try:
            ds = load_dataset(repo, split=split, streaming=True, token=token)
            if ds.features and "audio" in ds.features:
                ds = ds.cast_column("audio", Audio(decode=False))
            feats = list(ds.features) if ds.features else None
            rows = [{k: (str(v)[:70] if k != "audio" else "<audio>") for k, v in r.items()}
                    for r in islice(ds, 3)]
            cap_cols = [f for f in (feats or []) if "caption" in f.lower()
                        or f.lower() in ("text", "captions", "raw_text", "all_captions")]
            out[repo] = {"features": feats, "caption_cols": cap_cols,
                         "has_audio": bool(feats and "audio" in feats), "rows": rows}
            print(f"OK  {repo}  caption_cols={cap_cols}  audio={'audio' in (feats or [])}")
        except Exception as e:                                   # noqa: BLE001
            out[repo] = f"ERR: {type(e).__name__}: {str(e)[:120]}"
            print(f"FAIL {repo}: {str(e)[:110]}")
    print("CLOTHO CANDIDATES:\n" + json.dumps(out, indent=2, default=str))
    return out


@app.function(secrets=[hf_secret], cpu=2.0, timeout=1800, env=HF_ENV)
def peek_clotho_grouping(repo: str = "CLAPv2/Clotho", split: str = "test", limit: int = 6000) -> dict:
    """Decide whether CLAPv2/Clotho can drive a 5-ref eval: is each clip's audio DUPLICATED across
    its caption rows (groupable) or one-caption-per-clip (not a min-rank-over-5 set)? Reports the
    candidate grouping keys (audio path basename, index-minus-suffix) + their repeat distribution."""
    import json
    import os
    from collections import Counter
    from itertools import islice

    from datasets import Audio, load_dataset

    token = os.environ.get("HF_TOKEN") or None
    ds = load_dataset(repo, split=split, streaming=True, token=token).cast_column("audio", Audio(decode=False))
    path_counts, n = Counter(), 0
    sample_paths = []
    for r in islice(ds, limit):
        a = r.get("audio") or {}
        p = a.get("path") or ""
        base = os.path.basename(str(p))
        path_counts[base] += 1
        if len(sample_paths) < 8:
            sample_paths.append({"index": str(r.get("index"))[:60], "audio_path": str(p)[:90]})
        n += 1
    reps = Counter(path_counts.values())                        # {captions-per-clip: how many clips}
    out = {"repo": repo, "rows_scanned": n, "unique_audio_paths": len(path_counts),
           "captions_per_clip_distribution": dict(sorted(reps.items())),
           "max_captions_for_one_clip": max(path_counts.values()) if path_counts else 0,
           "groupable_by_audio_path": len(path_counts) > 0 and len(path_counts) < n,
           "sample": sample_paths}
    print("CLOTHO GROUPING:\n" + json.dumps(out, indent=2))
    return out


@app.function(secrets=[hf_secret], cpu=2.0, timeout=900, env=HF_ENV)
def peek_wavcaps() -> dict:
    """Peek streamable WavCaps AudioSet_SL mirrors: features + a sample (need an id field for
    blacklist matching + a decodable audio field). Informs the ingestion path."""
    import json
    import os

    from datasets import Audio, load_dataset

    token = os.environ.get("HF_TOKEN") or None
    # (repo, config, split)
    candidates = [
        ("totoluo/wavcaps", "audioset_sl", "train"),
        ("TwinkStart/wavcaps-audioset", None, "test"),
        ("TwinkStart/wavcaps-soundbible", None, "test"),
    ]
    report = {}
    for repo, cfg, split in candidates:
        key = f"{repo}:{cfg}" if cfg else repo
        try:
            ds = (load_dataset(repo, cfg, split=split, streaming=True, token=token) if cfg
                  else load_dataset(repo, split=split, streaming=True, token=token))
            feats = list(ds.features) if ds.features else None
            if feats and "audio" in feats:
                ds = ds.cast_column("audio", Audio(decode=False))
            sample = next(iter(ds))
            text_fields = {k: str(v)[:100] for k, v in sample.items()
                           if isinstance(v, (str, int, float)) and k != "audio"}
            report[key] = {"split": split, "features": feats, "sample_fields": text_fields}
            print(f"OK  {key}  features={feats}")
        except Exception as e:  # noqa: BLE001
            report[key] = f"ERR: {type(e).__name__}: {str(e)[:180]}"
            print(f"FAIL {key}: {str(e)[:150]}")
    print("WAVCAPS MIRRORS:\n" + json.dumps(report, indent=2, default=str))
    return report


# --------------------------------------------------------------------------- #
# 1. warm_cache — pull the frozen Qwen weights onto the Volume once, so every
#    later GPU run mounts them instantly instead of re-downloading.
# --------------------------------------------------------------------------- #
@app.function(volumes={VOL: volume}, secrets=[hf_secret], timeout=3600, env=HF_ENV)
def warm_cache() -> dict:
    from huggingface_hub import snapshot_download

    info = {}
    for repo in (BASE_MODEL, AUDIO_MODEL):
        # TODO(fusion): the Omni repo is large (~7B). If only the audio tower is needed,
        # narrow with allow_patterns to cut download time/space once the layout is known.
        path = snapshot_download(repo, cache_dir=HF_CACHE)
        info[repo] = path
        print(f"cached {repo} -> {path}")
    volume.commit()                                    # persist downloads to the Volume
    return info


# --------------------------------------------------------------------------- #
# 2. preprocess — CPU fan-out: decode audio -> 128-mel -> store on the Volume,
#    so the H100 never spends GPU-seconds on audio decode (the real bottleneck).
# --------------------------------------------------------------------------- #
@app.function(volumes={VOL: volume}, secrets=[hf_secret], cpu=4.0, memory=16384, timeout=3600, env=HF_ENV)
def preprocess(
    shard: str = "audiocaps",
    dataset_repo: str = "OpenSound/AudioCaps",
    split: str = "train",
    limit: int = 1200,
    audio_col: str = "audio",
    text_col: str = "caption",
    task: str = "sound",
) -> dict:
    """Decode a real audio↔text dataset -> Whisper mel -> per-clip ``.pt`` on the Volume.

    Default = AudioCaps (rich, UNIQUE per-clip captions -> clean contrastive, no class
    collisions). The GPU never decodes audio — that cost is paid here once. The real Omni
    WhisperFeatureExtractor is used so the mel matches what the frozen audio tower expects.
    """
    import io
    import itertools
    import os

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from datasets import load_dataset, Audio
    from transformers import AutoFeatureExtractor

    token = os.environ.get("HF_TOKEN") or None
    fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL, trust_remote_code=True, token=token)
    sr = fe.sampling_rate

    # STREAMING: pull only the `limit` clips we consume (the full split is ~49K clips / many GB).
    ds = load_dataset(dataset_repo, split=split, streaming=True, token=token)
    print(f"streaming {dataset_repo}:{split} | features: {list(ds.features)}")
    ds = ds.cast_column(audio_col, Audio(decode=False))    # raw bytes -> soundfile (no torchcodec)
    ds = itertools.islice(ds, limit)

    def decode_wav(a) -> np.ndarray:
        if a.get("bytes"):
            wav, sr0 = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        else:
            wav, sr0 = sf.read(a["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr0 != sr:
            wav = librosa.resample(wav, orig_sr=sr0, target_sr=sr)
        return wav

    out_dir = f"{FEATURES}/{shard}"
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for i, row in enumerate(ds):
        wav = decode_wav(row[audio_col])
        caption = str(row[text_col]).replace("_", " ").strip()
        feats = fe(wav, sampling_rate=sr, return_tensors="pt", return_attention_mask=True,
                   padding="max_length", truncation=True)
        mel = feats["input_features"][0]
        am = feats.get("attention_mask")
        if am is not None:                                     # trim padded mel to real length
            L = int(am[0].sum().item()); mel = mel[:, :L]
        torch.save({"mel": mel.contiguous(), "text": caption, "task": task},
                   f"{out_dir}/item-{i:05d}.pt")
        n += 1
        if i % 100 == 0:
            print(f"  {i}/{limit}  '{caption[:50]}'  mel{tuple(mel.shape)}")
    volume.commit()
    print(f"preprocessed {n} clips -> {out_dir}")
    return {"shard": shard, "count": n, "dir": out_dir, "dataset": dataset_repo}


# --------------------------------------------------------------------------- #
# 4b. preprocess_wavcaps — REAL DATA scale-up: WavCaps (cvssp/WavCaps) -> mel.
#     WavCaps ships caption JSONs + (multi-part) FLAC zips, NOT a streamable
#     parquet. We download the source JSON + audio, apply the repo's eval-leakage
#     BLACKLIST (AudioCaps/Clotho/ESC-50/VGGSound overlap), then mel like preprocess.
#     Start with source="SoundBible" (1,232 clips, single zip) to validate the path.
# --------------------------------------------------------------------------- #
_WAVCAPS_JSON = {
    "SoundBible": "json_files/SoundBible/sb_final.json",
    "AudioSet_SL": "json_files/AudioSet_SL/as_final.json",
    "BBC_Sound_Effects": "json_files/BBC_Sound_Effects/bbc_final.json",
    "FreeSound": "json_files/FreeSound/fsd_final.json",
}


def _wavcaps_flac_name(source: str, item_id: str) -> str:
    """JSON id -> extracted flac filename (AudioSet ids carry a `.wav` we swap to `.flac`)."""
    return item_id.replace(".wav", ".flac") if source == "AudioSet_SL" else f"{item_id}.flac"


def _flatten_ids(obj, acc: set) -> set:
    """Collect every leaf string (and its extension-stripped form) from a nested blacklist JSON."""
    if isinstance(obj, str):
        acc.add(obj); acc.add(obj.replace(".wav", "").replace(".flac", ""))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_ids(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_ids(v, acc)
    return acc


@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=6 * 3600,
              memory=32768, cpu=4.0, env=HF_ENV)
def ingest_clotho_eval(frame_shard: str = "clotho_eval5", zenodo_record: str = "4783391",
                       csv_name: str = "clotho_captions_evaluation.csv",
                       audio_7z: str = "clotho_audio_evaluation.7z",
                       audio_feature_layer: str = "post_proj", shard_size: int = 512,
                       limit: int = 0) -> dict:
    """Ingest the CANONICAL Clotho v2.1 EVALUATION set (1045 clips × 5 refs) straight from Zenodo —
    the only source that carries all 5 captions per clip. Downloads the 5-caption CSV + evaluation
    audio 7z, decodes each wav → Whisper mel → frozen tower → sharded frames + index with
    `captions_multi` (5/clip) + `clip_ids` (file_name). Enables the published min-rank-over-5 A→T."""
    import csv as _csv
    import glob
    import json
    import os
    import subprocess
    import urllib.request

    import librosa
    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor

    from fusion_embedding.data import write_frame_shard
    from fusion_embedding.hf_components import load_audio_tower
    from fusion_embedding.paths import frames_dir

    base = f"https://zenodo.org/records/{zenodo_record}/files"
    work = "/tmp/clotho"; os.makedirs(work, exist_ok=True)
    csv_path = os.path.join(work, csv_name); sevenz = os.path.join(work, audio_7z)

    print(f"downloading {csv_name} ...", flush=True)
    urllib.request.urlretrieve(f"{base}/{csv_name}?download=1", csv_path)
    caps: dict = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:     # file_name,caption_1..caption_5
        reader = _csv.DictReader(fh)
        for row in reader:
            fn = row["file_name"]
            caps[fn] = [row[f"caption_{i}"].strip() for i in range(1, 6)
                        if row.get(f"caption_{i}", "").strip()]
    print(f"csv: {len(caps)} clips, cols={reader.fieldnames}", flush=True)

    print(f"downloading {audio_7z} (~1.6GB) ...", flush=True)
    urllib.request.urlretrieve(f"{base}/{audio_7z}?download=1", sevenz)
    subprocess.run(["7z", "x", "-y", f"-o{work}", sevenz], check=True,
                   stdout=subprocess.DEVNULL)
    wav_by_name = {os.path.basename(p): p for p in glob.glob(f"{work}/**/*.wav", recursive=True)}
    print(f"extracted {len(wav_by_name)} wavs", flush=True)

    token = os.environ.get("HF_TOKEN") or None
    dev = "cuda"
    fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL, trust_remote_code=True, token=token)
    sr = fe.sampling_rate
    enc, _fe, d_audio = load_audio_tower(device=dev, dtype=torch.bfloat16,
                                         audio_feature_layer=audio_feature_layer)

    out_dir = frames_dir(frame_shard); os.makedirs(str(out_dir), exist_ok=True)
    shard_recs, captions_multi, clip_ids, shard_files = [], [], [], []
    n = n_missing = n_bad = 0
    names = list(caps.keys())[:limit] if limit else list(caps.keys())

    def _write():
        if not shard_recs:
            return
        name = f"shard-{len(shard_files):04d}.pt"
        write_frame_shard(out_dir / name, shard_recs, half=True)
        shard_files.append(name); shard_recs.clear(); volume.commit()

    for fn in names:
        path = wav_by_name.get(fn)
        if path is None:
            n_missing += 1; continue
        try:
            wav, sr0 = sf.read(path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr0 != sr:
                wav = librosa.resample(wav, orig_sr=sr0, target_sr=sr)
        except Exception:                                        # noqa: BLE001
            n_bad += 1; continue
        feats = fe(wav, sampling_rate=sr, return_tensors="pt", return_attention_mask=True,
                   padding="max_length", truncation=True)
        mel = feats["input_features"][0]
        am = feats.get("attention_mask")
        if am is not None:
            mel = mel[:, : int(am[0].sum().item())]
        with torch.no_grad():
            frames, fmask = enc(mel.unsqueeze(0).to(dev),
                                torch.ones(1, mel.shape[1], dtype=torch.bool, device=dev))
        t = int(fmask[0].sum().item())
        shard_recs.append({"frames": frames[0, :t].cpu().contiguous(), "text": caps[fn][0], "task": "sound"})
        captions_multi.append(caps[fn]); clip_ids.append(fn); n += 1
        if len(shard_recs) >= shard_size:
            _write()
        if n % 200 == 0:
            print(f"  {n} clips  missing={n_missing} bad={n_bad}", flush=True)
    _write()

    with open(str(out_dir / "index.json"), "w") as fh:
        json.dump({"d_audio": d_audio, "shard_size": shard_size, "n_total": n,
                   "shards": shard_files, "captions_multi": captions_multi, "clip_ids": clip_ids,
                   "source_repo": f"zenodo:{zenodo_record}", "group_key": "file_name",
                   "captions": [c[0] for c in captions_multi]}, fh)
    volume.commit()
    result = {"frame_shard": frame_shard, "clips": n, "missing_audio": n_missing, "decode_fail": n_bad,
              "total_captions": sum(len(c) for c in captions_multi),
              "avg_caps_per_clip": round(sum(len(c) for c in captions_multi) / max(n, 1), 2),
              "shards": len(shard_files), "d_audio": d_audio}
    print(f"INGEST_CLOTHO5: {result}")
    return result


@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=6 * 3600,
              memory=32768, cpu=4.0, env=HF_ENV)
def ingest_audiocaps_eval(frame_shard: str = "audiocaps_test816", repo: str = "OpenSound/AudioCaps",
                          split: str = "test", group_key: str = "youtube_id",
                          caption_col: str = "caption", audio_feature_layer: str = "post_proj",
                          shard_size: int = 512, limit: int = 0) -> dict:
    """Ingest a MULTI-CAPTION eval set (AudioCaps/Clotho test: ~5 captions/clip) as sharded frames.

    Groups rows by clip id → one audio + its list of reference captions. Writes frames (one per clip)
    + `index.json` with `captions_multi` (list-of-lists) so `rescore_816` can score min-rank-over-5.
    NO blacklist here — this IS the held-out eval; training already excludes these ids.
    """
    import io
    import itertools
    import json
    import os

    import librosa
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from transformers import AutoFeatureExtractor

    from fusion_embedding.data import write_frame_shard
    from fusion_embedding.hf_components import load_audio_tower
    from fusion_embedding.paths import frames_dir

    dev = "cuda"
    token = os.environ.get("HF_TOKEN") or None
    fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL, trust_remote_code=True, token=token)
    sr = fe.sampling_rate
    enc, _fe, d_audio = load_audio_tower(device=dev, dtype=torch.bfloat16,
                                         audio_feature_layer=audio_feature_layer)

    ds = load_dataset(repo, split=split, streaming=True, token=token).cast_column("audio", Audio(decode=False))
    if limit:
        ds = itertools.islice(ds, limit)

    # group rows by clip -> {audio ref (first seen), captions[]}
    clips: dict = {}
    order: list = []
    for row in ds:
        gid = str(row.get(group_key, "")) + "|" + str(row.get("start_time", ""))
        cap = str(row.get(caption_col, "")).strip()
        if gid not in clips:
            clips[gid] = {"audio": row["audio"], "captions": []}
            order.append(gid)
        if cap:
            clips[gid]["captions"].append(cap)
    print(f"{len(order)} unique clips, {sum(len(clips[g]['captions']) for g in order)} captions")

    def _decode(a):
        if a.get("bytes"):
            wav, sr0 = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        else:
            wav, sr0 = sf.read(a["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr0 != sr:
            wav = librosa.resample(wav, orig_sr=sr0, target_sr=sr)
        return wav

    out_dir = frames_dir(frame_shard)
    os.makedirs(str(out_dir), exist_ok=True)
    shard_recs: list = []
    captions_multi: list = []
    clip_ids: list = []
    shard_files: list = []
    n = n_bad = 0

    def _write():
        if not shard_recs:
            return
        name = f"shard-{len(shard_files):04d}.pt"
        write_frame_shard(out_dir / name, shard_recs, half=True)
        shard_files.append(name); shard_recs.clear(); volume.commit()

    for gid in order:
        caps = clips[gid]["captions"]
        if not caps:
            continue
        try:
            wav = _decode(clips[gid]["audio"])
        except Exception:                                        # noqa: BLE001
            n_bad += 1
            continue
        feats = fe(wav, sampling_rate=sr, return_tensors="pt", return_attention_mask=True,
                   padding="max_length", truncation=True)
        mel = feats["input_features"][0]
        am = feats.get("attention_mask")
        if am is not None:
            mel = mel[:, : int(am[0].sum().item())]
        with torch.no_grad():
            frames, fmask = enc(mel.unsqueeze(0).to(dev), torch.ones(1, mel.shape[1], dtype=torch.bool, device=dev))
        t = int(fmask[0].sum().item())
        shard_recs.append({"frames": frames[0, :t].cpu().contiguous(), "text": caps[0], "task": "sound"})
        captions_multi.append(caps); clip_ids.append(gid); n += 1     # gid = "<group_key>|<start_time>"
        if len(shard_recs) >= shard_size:
            _write()
        if n % 200 == 0:
            print(f"  {n} clips  bad={n_bad}", flush=True)
    _write()

    with open(str(out_dir / "index.json"), "w") as fh:
        json.dump({"d_audio": d_audio, "shard_size": shard_size, "n_total": n,
                   "shards": shard_files, "captions_multi": captions_multi, "clip_ids": clip_ids,
                   "source_repo": repo, "group_key": group_key,
                   "captions": [c[0] for c in captions_multi]}, fh)   # `captions`=first ref (train-format compat)
    volume.commit()
    result = {"frame_shard": frame_shard, "clips": n, "decode_fail": n_bad,
              "total_captions": sum(len(c) for c in captions_multi), "shards": len(shard_files),
              "d_audio": d_audio}
    print(f"INGEST_EVAL: {result}")
    return result


def _wavcaps_excluded(iid: str, excl: set) -> bool:
    """Blacklist check tolerant of the AudioSet `Y` prefix + `.wav/.flac` extension variants."""
    norm = iid.replace(".wav", "").replace(".flac", "")
    cands = {iid, norm}
    if norm.startswith("Y"):
        cands.add(norm[1:])                                     # AudioCaps ids are usually un-prefixed
    return bool(cands & excl)


@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=24 * 3600,
              memory=32768, cpu=4.0, env=HF_ENV)
def ingest_wavcaps_frames(repo: str = "TwinkStart/wavcaps-audioset", split: str = "test",
                          frame_shard: str = "wavcaps_audioset_sl", audio_feature_layer: str = "post_proj",
                          shard_size: int = 512, limit: int = 0, apply_blacklist: bool = True,
                          batch: int = 16, id_col: str = "id", caption_col: str = "caption") -> dict:
    """FUSED ingestion: stream a WavCaps mirror -> decode -> Whisper mel -> frozen audio tower ->
    SHARDED frames directly. No 108K intermediate mel files, no zip reassembly, one GPU pass.
    Applies the eval-leakage blacklist by id (AudioSet_SL heavily overlaps AudioCaps).
    ``id_col``/``caption_col`` generalise to other audio-caption mirrors (e.g. CLAPv2/FSD50K:
    id_col=index caption_col=text — FSD50K ids are Freesound ids, path prefix stripped, and the
    ub8k/esc50 blacklist entries ARE Freesound ids, so leakage dedup still applies)."""
    import io
    import itertools
    import json
    import os

    import librosa
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoFeatureExtractor

    from fusion_embedding.data import write_frame_shard
    from fusion_embedding.hf_components import load_audio_tower
    from fusion_embedding.paths import frames_dir

    dev = "cuda"
    token = os.environ.get("HF_TOKEN") or None
    fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL, trust_remote_code=True, token=token)
    sr = fe.sampling_rate
    enc, _fe, d_audio = load_audio_tower(device=dev, dtype=torch.bfloat16,
                                         audio_feature_layer=audio_feature_layer)

    excl: set = set()
    if apply_blacklist:
        for bl in ("blacklist_exclude_all_ac.json", "blacklist_exclude_ub8k_esc50_vggsound.json"):
            try:
                p = hf_hub_download("cvssp/WavCaps", f"json_files/blacklist/{bl}",
                                    repo_type="dataset", token=token)
                with open(p) as fh:
                    _flatten_ids(json.load(fh), excl)
            except Exception as e:                               # noqa: BLE001
                print(f"  blacklist {bl}: {e}")
        print(f"blacklist excludes {len(excl)} ids")

    ds = load_dataset(repo, split=split, streaming=True, token=token).cast_column("audio", Audio(decode=False))
    if limit:
        ds = itertools.islice(ds, limit)

    out_dir = frames_dir(frame_shard)
    os.makedirs(str(out_dir), exist_ok=True)

    def _decode(a):
        if a.get("bytes"):
            wav, sr0 = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        else:
            wav, sr0 = sf.read(a["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr0 != sr:
            wav = librosa.resample(wav, orig_sr=sr0, target_sr=sr)
        return wav

    mel_buf: list = []
    cap_buf: list = []
    shard_recs: list = []
    captions: list = []
    tasks: list = []
    shard_files: list = []
    n_seen = n_kept = n_bl = n_bad = 0

    def _run_tower():                                            # mel batch -> frames -> shard buffer
        if not mel_buf:
            return
        n_mels = mel_buf[0].shape[0]; Fmax = max(m.shape[1] for m in mel_buf)
        mb = torch.zeros(len(mel_buf), n_mels, Fmax, device=dev)
        mm = torch.zeros(len(mel_buf), Fmax, dtype=torch.bool, device=dev)
        for i, m in enumerate(mel_buf):
            mb[i, :, : m.shape[1]] = m.to(dev); mm[i, : m.shape[1]] = True
        frames, fmask = enc(mb, mm)
        for i, cap in enumerate(cap_buf):
            t = int(fmask[i].sum().item())
            shard_recs.append({"frames": frames[i, :t].cpu().contiguous(), "text": cap, "task": "sound"})
            captions.append(cap); tasks.append("sound")
        mel_buf.clear(); cap_buf.clear()

    def _write_shard():
        if not shard_recs:
            return
        name = f"shard-{len(shard_files):04d}.pt"
        write_frame_shard(out_dir / name, shard_recs, half=True)
        shard_files.append(name); shard_recs.clear(); volume.commit()

    for row in ds:
        n_seen += 1
        iid = os.path.basename(str(row.get(id_col, "")))         # "./train/10047" -> "10047"
        if apply_blacklist and _wavcaps_excluded(iid, excl):
            n_bl += 1
            continue
        cap = str(row.get(caption_col, "")).strip()
        if not cap:
            continue
        try:
            wav = _decode(row["audio"])
        except Exception:                                        # noqa: BLE001
            n_bad += 1
            continue
        feats = fe(wav, sampling_rate=sr, return_tensors="pt", return_attention_mask=True,
                   padding="max_length", truncation=True)
        mel = feats["input_features"][0]
        am = feats.get("attention_mask")
        if am is not None:
            mel = mel[:, : int(am[0].sum().item())]
        mel_buf.append(mel); cap_buf.append(cap); n_kept += 1
        if len(mel_buf) >= batch:
            _run_tower()
        if len(shard_recs) >= shard_size:
            _write_shard()
        if n_kept % 500 == 0:
            print(f"  seen={n_seen} kept={n_kept} bl={n_bl} bad={n_bad} shards={len(shard_files)}", flush=True)
    _run_tower(); _write_shard()

    with open(str(out_dir / "index.json"), "w") as fh:
        json.dump({"d_audio": d_audio, "shard_size": shard_size, "n_total": n_kept,
                   "captions": captions, "tasks": tasks, "shards": shard_files}, fh)
    volume.commit()
    result = {"frame_shard": frame_shard, "repo": repo, "seen": n_seen, "kept": n_kept,
              "blacklisted": n_bl, "decode_fail": n_bad, "shards": len(shard_files), "d_audio": d_audio}
    print(f"INGEST_WAVCAPS_FRAMES: {result}")
    return result


@app.function(cpu=2.0, timeout=600, env=HF_ENV)
def smoke_extraction() -> dict:
    """Integration test IN THE PROD IMAGE: build a real spanned zip with the image's own
    zip/unzip/7z binaries and run the extract_split_zip strategy chain against it.
    Run this after ANY change to the extraction code, BEFORE a 35GB ingest trusts it."""
    import os
    import subprocess
    import tempfile

    from fusion_embedding.ingest_utils import extract_split_zip

    src = tempfile.mkdtemp(); zdir = tempfile.mkdtemp(); audio = tempfile.mkdtemp()
    for i in range(8):
        with open(os.path.join(src, f"clip{i}.flac"), "wb") as fh:
            fh.write(os.urandom(64 * 1024))
    subprocess.run(["zip", "-q", "-s", "100k", "-r", os.path.join(zdir, "SRC.zip"), "."],
                   cwd=src, check=True)
    parts = sorted(os.listdir(zdir))
    assert any(p.endswith(".z01") for p in parts), f"not spanned: {parts}"
    label, n = extract_split_zip(zdir, "SRC", audio)
    result = {"parts": parts, "strategy": label, "extracted": n, "ok": n == 8}
    print(f"SMOKE_EXTRACTION: {result}")
    assert n == 8, result
    return result


@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=20 * 3600,
              memory=32768, cpu=8.0, ephemeral_disk=512 * 1024, env=HF_ENV)  # 512GiB = Modal's minimum
def ingest_wavcaps_zip(source: str = "AudioSet_SL", frame_shard: str = "",
                       audio_feature_layer: str = "post_proj", shard_size: int = 512,
                       limit: int = 0, apply_blacklist: bool = True, batch: int = 16) -> dict:
    """FULL WavCaps source via the canonical cvssp/WavCaps multi-part zip → fused frames.

    The streamable mirrors are partial (TwinkStart ASL = 9K of 108K) or id-less (totoluo), so the
    complete clean set only exists in the origin zips. Downloads the parts to EPHEMERAL disk (NOT
    the Volume — 35GB+), `zip -FF` reassembles, extracts, then joins caption JSON → blacklist →
    decode → Whisper mel → frozen tower → sharded frames + text-cache-ready index. Sources:
    AudioSet_SL (108K, ~18% blacklisted), BBC_Sound_Effects (31K), FreeSound (262K, ~640GB — NOT
    via this fn without a bigger disk).
    """
    import glob
    import json
    import os
    import shutil
    import subprocess
    import zipfile

    import librosa
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoFeatureExtractor

    from fusion_embedding.data import write_frame_shard
    from fusion_embedding.hf_components import load_audio_tower
    from fusion_embedding.paths import frames_dir

    if source not in _WAVCAPS_JSON:
        raise ValueError(f"source must be one of {list(_WAVCAPS_JSON)}")
    token = os.environ.get("HF_TOKEN") or None
    frame_shard = frame_shard or f"wavcaps_{source.lower()}_full"
    dev = "cuda"
    fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL, trust_remote_code=True, token=token)
    sr = fe.sampling_rate
    enc, _fe, d_audio = load_audio_tower(device=dev, dtype=torch.bfloat16,
                                         audio_feature_layer=audio_feature_layer)

    # 1) captions + blacklist
    jpath = hf_hub_download("cvssp/WavCaps", _WAVCAPS_JSON[source], repo_type="dataset", token=token)
    with open(jpath) as fh:
        data = json.load(fh).get("data", [])
    excl: set = set()
    if apply_blacklist:
        for bl in ("blacklist_exclude_all_ac.json", "blacklist_exclude_ub8k_esc50_vggsound.json"):
            try:
                p = hf_hub_download("cvssp/WavCaps", f"json_files/blacklist/{bl}",
                                    repo_type="dataset", token=token)
                with open(p) as fh:
                    _flatten_ids(json.load(fh), excl)
            except Exception as e:                               # noqa: BLE001
                print(f"  blacklist {bl}: {e}")
    print(f"{source}: {len(data)} captions | blacklist {len(excl)} ids", flush=True)

    # 2) zips -> EPHEMERAL disk (local_dir overrides HF_HOME which lives on the Volume)
    zroot = "/tmp/zips"; audio_dir = "/tmp/audio"
    os.makedirs(audio_dir, exist_ok=True)
    print("downloading zip parts ...", flush=True)
    from fusion_embedding.ingest_utils import retry
    # retried: multi-GB HF snapshots die on transient disconnects; resume makes retries cheap
    d = retry(lambda: snapshot_download("cvssp/WavCaps", repo_type="dataset", local_dir=zroot,
                                        allow_patterns=[f"Zip_files/{source}/*"], token=token),
              attempts=4, wait_s=60, log=lambda m: print(m, flush=True))
    zdir = os.path.join(d, "Zip_files", source)
    merged = f"/tmp/{source}_merged.zip"
    from fusion_embedding.ingest_utils import extract_split_zip   # unit+smoke-tested strategy chain
    strategy, n_extracted = extract_split_zip(zdir, source, audio_dir, merged_path=merged,
                                              log=lambda m: print(m, flush=True))
    print(f"extraction strategy={strategy}: {n_extracted} flacs", flush=True)
    shutil.rmtree(zroot, ignore_errors=True)                     # free parts only AFTER success
    if os.path.exists(merged):
        os.remove(merged)
    flac_index = {os.path.basename(p): p
                  for p in glob.glob(os.path.join(audio_dir, "**", "*.flac"), recursive=True)}

    # 3) join -> decode -> tower -> shards (same fused shape as ingest_wavcaps_frames)
    out_dir = frames_dir(frame_shard)
    os.makedirs(str(out_dir), exist_ok=True)
    mel_buf: list = []; cap_buf: list = []
    shard_recs: list = []; captions: list = []; tasks: list = []; shard_files: list = []
    kept = n_bl = n_missing = n_bad = 0

    def _run_tower():
        if not mel_buf:
            return
        n_mels = mel_buf[0].shape[0]; Fmax = max(m.shape[1] for m in mel_buf)
        mb = torch.zeros(len(mel_buf), n_mels, Fmax, device=dev)
        mm = torch.zeros(len(mel_buf), Fmax, dtype=torch.bool, device=dev)
        for i, m in enumerate(mel_buf):
            mb[i, :, : m.shape[1]] = m.to(dev); mm[i, : m.shape[1]] = True
        with torch.no_grad():
            frames, fmask = enc(mb, mm)
        for i, cap in enumerate(cap_buf):
            t = int(fmask[i].sum().item())
            shard_recs.append({"frames": frames[i, :t].cpu().contiguous(), "text": cap, "task": "sound"})
            captions.append(cap); tasks.append("sound")
        mel_buf.clear(); cap_buf.clear()

    def _write_shard():
        if not shard_recs:
            return
        name = f"shard-{len(shard_files):04d}.pt"
        write_frame_shard(out_dir / name, shard_recs, half=True)
        shard_files.append(name); shard_recs.clear(); volume.commit()

    for item in data:
        iid = str(item.get("id", ""))
        if apply_blacklist and _wavcaps_excluded(iid, excl):
            n_bl += 1
            continue
        cap = str(item.get("caption", "")).strip()
        if not cap:
            continue
        fp = flac_index.get(_wavcaps_flac_name(source, iid))
        if fp is None:
            n_missing += 1
            continue
        try:
            wav, sr0 = sf.read(fp, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr0 != sr:
                wav = librosa.resample(wav, orig_sr=sr0, target_sr=sr)
        except Exception:                                        # noqa: BLE001
            n_bad += 1
            continue
        feats = fe(wav, sampling_rate=sr, return_tensors="pt", return_attention_mask=True,
                   padding="max_length", truncation=True)
        mel = feats["input_features"][0]
        am = feats.get("attention_mask")
        if am is not None:
            mel = mel[:, : int(am[0].sum().item())]
        mel_buf.append(mel); cap_buf.append(cap); kept += 1
        if len(mel_buf) >= batch:
            _run_tower()
        if len(shard_recs) >= shard_size:
            _write_shard()
        if kept % 1000 == 0:
            print(f"  kept={kept} bl={n_bl} missing={n_missing} bad={n_bad} "
                  f"shards={len(shard_files)}", flush=True)
        if limit and kept >= limit:
            break
    _run_tower(); _write_shard()

    with open(str(out_dir / "index.json"), "w") as fh:
        json.dump({"d_audio": d_audio, "shard_size": shard_size, "n_total": kept,
                   "captions": captions, "tasks": tasks, "shards": shard_files}, fh)
    volume.commit()
    result = {"frame_shard": frame_shard, "source": source, "kept": kept, "blacklisted": n_bl,
              "missing_audio": n_missing, "decode_fail": n_bad, "shards": len(shard_files),
              "d_audio": d_audio}
    print(f"INGEST_WAVCAPS_ZIP: {result}")
    return result


@app.function(volumes={VOL: volume}, secrets=[hf_secret], timeout=12 * 3600,
              memory=32768, env=HF_ENV)
def preprocess_wavcaps(source: str = "SoundBible", shard: str = "", limit: int = 0,
                       apply_blacklist: bool = True, extract_dir: str = "/tmp/wavcaps") -> dict:
    """Download a WavCaps source, dedup against eval sets, and cache Whisper mel per clip."""
    import glob
    import json
    import os
    import subprocess
    import zipfile

    import librosa
    import numpy as np  # noqa: F401  (kept parallel to preprocess)
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoFeatureExtractor

    if source not in _WAVCAPS_JSON:
        raise ValueError(f"source must be one of {list(_WAVCAPS_JSON)}")
    token = os.environ.get("HF_TOKEN") or None
    shard = shard or f"wavcaps_{source.lower()}"
    fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL, trust_remote_code=True, token=token)
    sr = fe.sampling_rate

    # 1) captions
    jpath = hf_hub_download("cvssp/WavCaps", _WAVCAPS_JSON[source], repo_type="dataset", token=token)
    with open(jpath) as fh:
        data = json.load(fh).get("data", [])
    print(f"{source}: {len(data)} caption entries")

    # 2) eval-leakage blacklist (AudioCaps + UrbanSound8K/ESC-50/VGGSound overlap)
    excl: set = set()
    if apply_blacklist:
        for bl in ("blacklist_exclude_all_ac.json", "blacklist_exclude_ub8k_esc50_vggsound.json"):
            try:
                p = hf_hub_download("cvssp/WavCaps", f"json_files/blacklist/{bl}",
                                    repo_type="dataset", token=token)
                with open(p) as fh:
                    _flatten_ids(json.load(fh), excl)
            except Exception as e:                       # noqa: BLE001
                print(f"  blacklist {bl}: {e}")
        print(f"  blacklist excludes {len(excl)} ids")

    # 3) audio: download + extract to a flat flac index
    audio_dir = os.path.join(extract_dir, source)
    os.makedirs(audio_dir, exist_ok=True)
    if source == "SoundBible":                            # single zip -> pure-python extract
        zpath = hf_hub_download("cvssp/WavCaps", "Zip_files/SoundBible/SoundBible.zip",
                                repo_type="dataset", token=token)
        with zipfile.ZipFile(zpath) as zf:
            for m in zf.namelist():
                if m.endswith(".flac"):
                    dst = os.path.join(audio_dir, os.path.basename(m))
                    with zf.open(m) as s, open(dst, "wb") as d:
                        d.write(s.read())
    else:                                                 # multi-part spanned zip -> reassemble
        d = snapshot_download("cvssp/WavCaps", repo_type="dataset",
                              allow_patterns=[f"Zip_files/{source}/*"], token=token)
        zdir = os.path.join(d, "Zip_files", source)
        main_zip = os.path.join(zdir, f"{source}.zip")
        merged = os.path.join(extract_dir, f"{source}_merged.zip")
        # `zip -FF` reassembles the .z01.. parts; it may ask to confirm single-disk -> answer 'y'.
        subprocess.run(["zip", "-FF", main_zip, "--out", merged], input=b"y\ny\n", check=True)
        subprocess.run(["unzip", "-o", "-q", merged, "-d", audio_dir], check=True)

    flac_index = {os.path.basename(p): p
                  for p in glob.glob(os.path.join(audio_dir, "**", "*.flac"), recursive=True)}
    print(f"  extracted {len(flac_index)} flac files")

    # 4) mel per clip (skip blacklisted ids + missing/empty)
    out_dir = f"{FEATURES}/{shard}"
    os.makedirs(out_dir, exist_ok=True)
    n = kept = skipped_bl = skipped_missing = 0
    for item in data:
        iid = str(item.get("id", ""))
        norm = iid.replace(".wav", "").replace(".flac", "")
        if norm in excl or iid in excl:
            skipped_bl += 1
            continue
        fp = flac_index.get(_wavcaps_flac_name(source, iid))
        if fp is None:
            skipped_missing += 1
            continue
        caption = str(item.get("caption", "")).strip()
        if not caption:
            continue
        try:
            wav, sr0 = sf.read(fp, dtype="float32")
        except Exception:                                # noqa: BLE001
            skipped_missing += 1
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr0 != sr:
            wav = librosa.resample(wav, orig_sr=sr0, target_sr=sr)
        feats = fe(wav, sampling_rate=sr, return_tensors="pt", return_attention_mask=True,
                   padding="max_length", truncation=True)
        mel = feats["input_features"][0]
        am = feats.get("attention_mask")
        if am is not None:
            L = int(am[0].sum().item()); mel = mel[:, :L]
        torch.save({"mel": mel.contiguous(), "text": caption, "task": "sound"},
                   f"{out_dir}/item-{kept:05d}.pt")
        kept += 1; n += 1
        if limit and kept >= limit:
            break
        if kept % 200 == 0:
            print(f"  {kept} clips  '{caption[:50]}'")
            volume.commit()
    volume.commit()
    result = {"shard": shard, "source": source, "count": kept, "excluded_blacklist": skipped_bl,
              "missing_audio": skipped_missing, "n_captions": len(data), "dir": out_dir}
    print(f"WAVCAPS preprocessed {kept} clips ({source}) -> {out_dir} | {result}")
    return result


# --------------------------------------------------------------------------- #
# 3. train_p1 — the GPU function: Stage-1 connector training (HLD §5.2, §7).
#    Runs end-to-end on synthetic data now; the real-model seam is load_components.
# --------------------------------------------------------------------------- #
@app.function(gpu="H100", volumes={VOL: volume}, secrets=[hf_secret], timeout=24 * 3600, env=HF_ENV)
def train_p1(steps: int = 500, use_real_base: bool = False) -> dict:
    """Connector-only P1 loop on a Modal GPU, checkpointing to the Volume.

    use_real_base=False (default): tiny CPU stand-ins on the GPU — validates the full
        loop + checkpoint/resume + Volume wiring without needing the Qwen weights.
    use_real_base=True: wire the frozen Qwen 2B + Omni tower via load_components
        (HLD §10) — the one remaining seam. Build the real manifest from the Volume's
        preprocessed features, attach a large TextMemoryBank, and train for real.
    """
    import os
    import torch
    from fusion_embedding.config import FusionConfig
    from fusion_embedding.memory_bank import TextMemoryBank
    from fusion_embedding.train_stage1 import build_tiny_training_setup, train_stage1

    os.makedirs(CKPTS, exist_ok=True)
    dev = "cuda"

    if use_real_base:
        # TODO(fusion) — the real-model path (HLD §10 seams):
        #   from fusion_embedding.train_stage1 import load_components
        #   cfg, embed_tokens, base_lm, audio_encoder = load_components(
        #       FusionConfig(), BASE_MODEL, AUDIO_MODEL, device=dev)  # 4-bit base + grad-ckpt
        #   model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder).to(dev)
        #   manifest, collator = <build from FEATURES on the Volume>
        #   bank = precompute_text_bank(model, bank_manifest, collator)  # frozen text, once
        raise NotImplementedError(
            "Set use_real_base=True after implementing load_components (HLD §10). "
            "The loop, checkpointing, bank, and Volume wiring below are already proven."
        )

    cfg = FusionConfig.tiny(max_steps=steps, d_resampler=32, use_bf16=True)
    s = build_tiny_training_setup(cfg, n_train=8, batch_size=8, seed=0)
    s.model.to(dev)
    bank = TextMemoryBank(dim=cfg.d_llm, capacity=4096, device=dev)

    state = train_stage1(
        s.model, s.train_loader, s.loss_fn, cfg,
        steps=steps, eval_fn=s.eval_fn, device=dev, log_every=max(1, steps // 10),
        memory_bank=bank,
    )

    # Trained-params-only checkpoint (connector + temperature) — ~MBs, per HLD §5.3.
    ckpt_path = f"{CKPTS}/p1_connector_step{steps}.pt"
    torch.save(
        {
            "resampler": s.model.resampler.state_dict(),
            "logit_scale": s.model.logit_scale.detach().cpu(),
            "config": cfg.__dict__,
            "step": steps,
        },
        ckpt_path,
    )
    volume.commit()
    result = {
        "ckpt": ckpt_path,
        "a2t_R@1": state.final_eval["a2t_R@1"],
        "base_drift": state.final_eval["base_drift"],
        "regression_ok": state.final_eval["regression_ok"],
        "final_loss": state.history[-1]["loss"],
    }
    print("TRAIN_P1:", result)
    return result


# --------------------------------------------------------------------------- #
# 4. train_real — wire the REAL frozen Qwen base + Omni audio tower and run a
#    short connector-overfit to validate the whole real path end-to-end.
#    Defaults to L4 (24GB, cheap) for validation BEFORE committing H100 time.
# --------------------------------------------------------------------------- #
@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=3600, env=HF_ENV)
def train_real(steps: int = 60, n_items: int = 8, load_in_4bit: bool = True, use_bank: bool = True) -> dict:
    """Real-base validation: frozen Qwen3-VL-Embedding-2B + Omni audio tower, connector trains.

    Uses synthetic audio (real WhisperFeatureExtractor mel from noise) + synthetic captions so
    we exercise the true encoder/inject/pool/loss path without needing the real dataset yet.
    Asserts: it runs, loss is finite, and the frozen base does not move (base_drift == 0).
    """
    import torch

    from fusion_embedding.config import FusionConfig
    from fusion_embedding.model import FusionEmbeddingModel
    from fusion_embedding.losses import FusionContrastiveLoss
    from fusion_embedding.memory_bank import TextMemoryBank
    from fusion_embedding.train_stage1 import (
        RegressionGuard, build_optimizer, encode_dataset, retrieval_report,
    )
    from fusion_embedding.data import FusionCollator, FusionAudioTextManifest, make_synthetic_records
    from fusion_embedding.hf_components import load_components

    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg0 = FusionConfig(n_query=64, d_resampler=256, lambda_coral=0.05)
    cfg, embed_tokens, base_lm, audio_encoder, tokenizer, feat_extractor = load_components(
        cfg0, device=dev, dtype=torch.bfloat16, load_in_4bit=load_in_4bit, gradient_checkpointing=True,
    )
    print(f"resolved dims: d_llm={cfg.d_llm} d_audio={cfg.d_audio} audio_pad_id={cfg.audio_pad_id} eos={cfg.eos_id}")

    model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder)
    model.resampler.to(dev).float()                       # connector trains in fp32 (HLD §5.3)
    if isinstance(model.logit_scale, torch.nn.Parameter):
        model.logit_scale.data = model.logit_scale.data.to(dev)

    # real mel via WhisperFeatureExtractor over synthetic waveforms (validates the FE path)
    class RealMelProcessor:
        def __init__(self, fe, sr): self.fe, self.sr = fe, sr
        def __call__(self, record):
            import hashlib
            seed = int(hashlib.sha256(record["id"].encode()).hexdigest()[:8], 16) % (2**31)
            g = torch.Generator().manual_seed(seed)
            secs = 2.0 + float(torch.rand(1, generator=g).item()) * 3.0   # 2-5 s clips
            wav = torch.randn(int(self.sr * secs), generator=g).numpy()
            feats = self.fe(wav, sampling_rate=self.sr, return_tensors="pt",
                            return_attention_mask=True, padding="max_length", truncation=True)
            mel = feats["input_features"][0]
            am = feats.get("attention_mask")
            if am is not None:                            # trim padded mel frames to real length
                L = int(am[0].sum().item()); mel = mel[:, :L]
            return mel

    proc = RealMelProcessor(feat_extractor, cfg.sample_rate)
    # DISTINCT captions (not the near-identical "caption number i" template) so Qwen's text
    # embeddings are separable — a fair test of whether the connector can memorize the map
    # from each (real-encoder) audio to its text through the real frozen base.
    topics = [
        "a dog barking loudly in an empty parking garage at night",
        "gentle piano melody with soft rain on a window",
        "a woman explaining quantum physics in a lecture hall",
        "heavy traffic with honking cars and a distant siren",
        "ocean waves crashing on rocks with seagulls calling",
        "an electric guitar solo over a fast drum beat",
        "a child laughing while playing in a crowded playground",
        "footsteps on gravel followed by a creaking wooden door",
    ]
    records = [
        {"id": f"item-{i}", "audio": f"synthetic://item-{i}", "text": topics[i % len(topics)],
         "task": "sound"}
        for i in range(n_items)
    ]
    manifest = FusionAudioTextManifest(records, proc)
    collator = FusionCollator(cfg, _HFTok(tokenizer, cfg))

    from torch.utils.data import DataLoader
    import itertools
    loader = DataLoader(manifest, batch_size=min(2, n_items), collate_fn=collator, shuffle=True)
    loss_fn = FusionContrastiveLoss(cfg)
    opt = build_optimizer(model, cfg)
    bank = TextMemoryBank(dim=cfg.d_llm, capacity=256, device=dev)
    guard = RegressionGuard(model)

    model.train()
    di = itertools.cycle(loader)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in next(di).items()}
        bt = bank.get() if use_bank else None
        out = model(batch)
        loss, m = loss_fn(out["audio"], out["text"], out["logit_scale"], bank_text=bt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), cfg.grad_clip)
        opt.step()
        if use_bank:
            bank.enqueue(out["text"].detach())
        losses.append(float(loss))
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:>3} loss {float(loss):.4f} acc {float(m['acc_a2t']):.3f}")

    a, t = encode_dataset(model, manifest, collator, device=dev)
    rep = retrieval_report(a, t)
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "d_audio": cfg.d_audio, "d_llm": cfg.d_llm,
        "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
        "a2t_R@1": rep["a2t_R@1"],
        "base_drift": guard.max_drift(model),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    assert result["base_drift"] == 0.0, f"BASE LEAKED: drift {result['base_drift']}"
    assert losses[-1] == losses[-1], "NaN loss"
    print("TRAIN_REAL:", result)
    return result


class _HFTok:
    """Adapt a HF tokenizer to the FusionCollator's tokenizer interface."""

    def __init__(self, hf, cfg):
        self.hf = hf
        self.pad_id = cfg.pad_id
        self.eos_id = cfg.eos_id
        self.audio_pad_id = cfg.audio_pad_id

    def encode(self, text: str):
        return self.hf.encode(text, add_special_tokens=False)


# --------------------------------------------------------------------------- #
# 5. train_p1_real — REAL DATA: train the connector on preprocessed features
#    from the Volume against the frozen Qwen base. L4 first (cheap), checkpoints.
# --------------------------------------------------------------------------- #
@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=6 * 3600, env=HF_ENV)
def train_p1_real(
    shard: str = "audiocaps",
    steps: int = 800,
    batch_size: int = 8,
    bank_capacity: int = 4096,
    eval_size: int = 200,
    use_bank: bool = True,
    audio_feature_layer: str = "post_proj",
    load_in_4bit: bool = True,
    gpu_note: str = "",
) -> dict:
    """Stage-1 connector training on real cached audio↔text features (HLD §7 P1 gate).

    Held-out eval = up to ``eval_size`` clips with DISTINCT captions (so diagonal-GT
    retrieval is well-posed); training uses every remaining clip. R@1 = "does a held-out
    clip's audio retrieve its own caption among the eval set". Checkpoints connector+temp.
    """
    import glob
    import itertools
    import os

    import torch
    from torch.utils.data import DataLoader

    from fusion_embedding.config import FusionConfig
    from fusion_embedding.model import FusionEmbeddingModel
    from fusion_embedding.losses import FusionContrastiveLoss
    from fusion_embedding.memory_bank import TextMemoryBank
    from fusion_embedding.data import CachedFeatureDataset, FusionCollator
    from fusion_embedding.hf_components import load_components
    from fusion_embedding.train_stage1 import (
        RegressionGuard, build_optimizer, build_scheduler, encode_dataset, retrieval_report,
    )

    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} {gpu_note}")

    feat_dir = f"{FEATURES}/{shard}"
    paths = sorted(glob.glob(os.path.join(feat_dir, "*.pt")))
    if not paths:
        raise RuntimeError(f"no features in {feat_dir} — run preprocess first")
    print(f"{len(paths)} cached clips in {feat_dir}")

    # Held-out eval: up to `eval_size` clips with DISTINCT captions (well-posed diagonal GT);
    # everything else trains. Works for unique-caption (AudioCaps) and label-caption (ESC-50).
    eval_caps, eval_paths, train_paths = set(), [], []
    for p in paths:
        cap = torch.load(p, map_location="cpu", weights_only=False)["text"]
        if len(eval_paths) < eval_size and cap not in eval_caps:
            eval_paths.append(p); eval_caps.add(cap)
        else:
            train_paths.append(p)
    print(f"train={len(train_paths)} eval(held-out, unique-caption)={len(eval_paths)}")

    # --- real frozen base ---
    cfg0 = FusionConfig(n_query=64, d_resampler=256, lambda_coral=0.05, max_steps=steps)
    cfg, embed_tokens, base_lm, audio_encoder, tokenizer, _fe = load_components(
        cfg0, device=dev, dtype=torch.bfloat16, load_in_4bit=load_in_4bit, gradient_checkpointing=True,
        audio_feature_layer=audio_feature_layer,
    )
    print(f"dims: d_llm={cfg.d_llm} d_audio={cfg.d_audio} feat={audio_feature_layer} audio_pad_id={cfg.audio_pad_id}")

    model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder)
    model.resampler.to(dev).float()
    if isinstance(model.logit_scale, torch.nn.Parameter):
        model.logit_scale.data = model.logit_scale.data.to(dev)

    collator = FusionCollator(cfg, _HFTok(tokenizer, cfg))
    train_ds = CachedFeatureDataset(train_paths)
    eval_ds = CachedFeatureDataset(eval_paths)
    loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collator, shuffle=True, drop_last=True)

    loss_fn = FusionContrastiveLoss(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, steps)
    bank = TextMemoryBank(dim=cfg.d_llm, capacity=bank_capacity, device=dev)
    guard = RegressionGuard(model)

    model.train()
    di = itertools.cycle(loader)
    torch.cuda.reset_peak_memory_stats()
    hist = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in next(di).items()}
        bt = bank.get() if use_bank else None
        out = model(batch)
        loss, m = loss_fn(out["audio"], out["text"], out["logit_scale"], bank_text=bt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), cfg.grad_clip)
        opt.step()
        sched.step()
        if use_bank:
            bank.enqueue(out["text"].detach())
        if step % 25 == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(loss), "acc": float(m["acc_a2t"])})
            print(f"step {step:>4} loss {float(loss):.4f} acc {float(m['acc_a2t']):.3f} lr {sched.get_last_lr()[0]:.2e}")

    # eval: R@1 over the unique-caption set + the regression guard
    a, t = encode_dataset(model, eval_ds, collator, device=dev)
    rep = retrieval_report(a, t)
    drift = guard.max_drift(model)

    ckpt = f"{CKPTS}/p1_{shard}_{audio_feature_layer}_step{steps}.pt"
    os.makedirs(CKPTS, exist_ok=True)
    torch.save({"resampler": model.resampler.state_dict(),
                "logit_scale": model.logit_scale.detach().cpu(),
                "config": cfg.__dict__, "shard": shard, "steps": steps,
                "audio_feature_layer": audio_feature_layer}, ckpt)
    volume.commit()

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "shard": shard, "feat": audio_feature_layer, "d_audio": cfg.d_audio,
        "n_train": len(train_paths), "n_eval": len(eval_paths),
        "loss_first": hist[0]["loss"], "loss_last": hist[-1]["loss"],
        "a2t_R@1": rep["a2t_R@1"], "a2t_R@10": rep["a2t_R@10"],
        "t2a_R@1": rep["t2a_R@1"],
        "base_drift": drift, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "ckpt": ckpt,
    }
    # persist the result to the Volume so detached runs are retrievable via read_results
    import json
    res_path = f"{CKPTS}/result_{shard}_{audio_feature_layer}_step{steps}.json"
    with open(res_path, "w") as fh:
        json.dump(result, fh, indent=2)
    volume.commit()

    assert drift == 0.0, f"BASE LEAKED: {drift}"
    print("TRAIN_P1_REAL:", result)
    return result


@app.function(volumes={VOL: volume}, timeout=300)
def read_results() -> list:
    """Print every training result JSON on the Volume (for retrieving detached runs)."""
    import glob
    import json
    import os

    out = []
    for p in sorted(glob.glob(f"{CKPTS}/result_*.json")):
        with open(p) as fh:
            r = json.load(fh)
        out.append(r)
        print(f"{os.path.basename(p)}: R@1={r.get('a2t_R@1')} R@10={r.get('a2t_R@10')} "
              f"feat={r.get('feat')} loss {r.get('loss_first')}->{r.get('loss_last')} drift={r.get('base_drift')}")
    if not out:
        print("no results yet")
    return out


# --------------------------------------------------------------------------- #
# 6. precompute_frames — run the FROZEN audio tower ONCE over cached mel; cache
#    the frames. Training then skips the encoder entirely (the big Option 2 win).
# --------------------------------------------------------------------------- #
@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=6 * 3600, env=HF_ENV)
def precompute_frames(mel_shard: str = "audiocaps4k", frame_shard: str = "",
                      audio_feature_layer: str = "post_proj", batch: int = 16,
                      shard_size: int = 512) -> dict:
    """Frozen audio tower over cached mel -> frames packed into ``shard-NNNN.pt`` files.

    Sharded output (``shard_size`` clips/file) so training streams big sequential reads instead of
    one network round-trip per clip. ``index.json`` carries d_audio + shard_size + a flat caption/
    task list (order == global clip index) so the eval split needs no tensor loads.
    """
    import glob
    import json
    import torch
    from fusion_embedding.data import write_frame_shard
    from fusion_embedding.hf_components import load_audio_tower
    from fusion_embedding.paths import features_dir, frames_dir

    dev = "cuda"
    frame_shard = frame_shard or f"{mel_shard}_{audio_feature_layer}"
    enc, _fe, d_audio = load_audio_tower(device=dev, dtype=torch.bfloat16,
                                         audio_feature_layer=audio_feature_layer)
    paths = sorted(glob.glob(str(features_dir(mel_shard) / "*.pt")))
    out_dir = frames_dir(frame_shard)
    print(f"encoding {len(paths)} clips -> frames d_audio={d_audio}, shard_size={shard_size} -> {out_dir}")

    n = 0
    buf: list = []                                              # records pending write to the current shard
    captions: list = []                                        # flat, order == global clip index
    tasks: list = []
    shard_files: list = []

    def _flush():
        if not buf:
            return
        sf = f"shard-{len(shard_files):04d}.pt"
        write_frame_shard(out_dir / sf, buf, half=True)         # frames stored fp16
        shard_files.append(sf)
        buf.clear()
        volume.commit()

    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        recs = [torch.load(p, map_location="cpu", weights_only=False) for p in chunk]
        mels = [r["mel"] for r in recs]
        n_mels = mels[0].shape[0]
        Fmax = max(m.shape[1] for m in mels)
        mb = torch.zeros(len(mels), n_mels, Fmax, device=dev)
        mm = torch.zeros(len(mels), Fmax, dtype=torch.bool, device=dev)
        for i, m in enumerate(mels):
            mb[i, :, : m.shape[1]] = m.to(dev); mm[i, : m.shape[1]] = True
        frames, fmask = enc(mb, mm)                              # [B,T,d_audio], [B,T]
        for i, r in enumerate(recs):
            t = int(fmask[i].sum().item())
            buf.append({"frames": frames[i, :t].cpu().contiguous(), "text": r["text"], "task": r["task"]})
            captions.append(r["text"]); tasks.append(r["task"]); n += 1
            if len(buf) >= shard_size:
                _flush()
        if start % (batch * 20) == 0:
            print(f"  {n}/{len(paths)}  ({len(shard_files)} shards)")
    _flush()

    with open(str(out_dir / "index.json"), "w") as fh:          # captions without loading tensors
        json.dump({"d_audio": d_audio, "shard_size": shard_size, "n_total": n,
                   "captions": captions, "tasks": tasks, "shards": shard_files}, fh)
    volume.commit()
    print(f"cached {n} clips in {len(shard_files)} shards (+ index.json) -> {out_dir}")
    return {"frame_shard": frame_shard, "count": n, "shards": len(shard_files),
            "d_audio": d_audio, "dir": str(out_dir)}


# --------------------------------------------------------------------------- #
# 7. train_frames — train the connector on PRECOMPUTED frames (base only, no 7B
#    tower). Much faster/step -> bigger batch + longer schedule for real P1.
# --------------------------------------------------------------------------- #
@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=12 * 3600,
              memory=32768, env=HF_ENV,
              retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0))
def train_frames(frame_shard: str = "audiocaps4k_post_proj", steps: int = 2000,
                 batch_size: int = 32, eval_size: int = 200, lambda_coral: float = 0.05,
                 load_in_4bit: bool = True, gpu_note: str = "", whiten_text: bool = True,
                 run_tag: str = "", eval_816_shard: str = "audiocaps_test816",
                 use_text_cache: bool = False, accum_steps: int = 1,
                 bank_negatives: bool = False, peak_lr: float = 0.0,
                 d_resampler: int = 256, n_query: int = 64,
                 fn_mask_threshold: float = 0.0, soft_label_beta: float = 0.0) -> dict:
    """L4 wrapper — see _train_frames_impl."""
    return _train_frames_impl(frame_shard, steps, batch_size, eval_size, lambda_coral,
                              load_in_4bit, gpu_note, whiten_text, run_tag, eval_816_shard,
                              use_text_cache, accum_steps, bank_negatives, peak_lr,
                              d_resampler, n_query, fn_mask_threshold, soft_label_beta)


@app.function(gpu="H100", volumes={VOL: volume}, secrets=[hf_secret], timeout=16 * 3600,
              memory=131072, env=HF_ENV,
              # Preemption resilience half 2: auto-retry + the every-100-step resume ckpt means a
              # preempted run relaunches itself and loses <=100 steps — no human in the loop.
              # (A preemption at step 850 once cost 4.5h/$11 because neither half existed.)
              retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0))
def train_frames_a100(frame_shard: str = "audiocaps4k_post_proj", steps: int = 4000,
                      batch_size: int = 128, eval_size: int = 200, lambda_coral: float = 0.05,
                      load_in_4bit: bool = True, gpu_note: str = "A100", whiten_text: bool = True,
                      run_tag: str = "", eval_816_shard: str = "audiocaps_test816",
                      use_text_cache: bool = False, accum_steps: int = 1,
                      bank_negatives: bool = False, peak_lr: float = 0.0,
                      d_resampler: int = 256, n_query: int = 64,
                      fn_mask_threshold: float = 0.0, soft_label_beta: float = 0.0) -> dict:
    """A100-80GB wrapper — bigger batch (more negatives) + more RAM for larger frame sets."""
    return _train_frames_impl(frame_shard, steps, batch_size, eval_size, lambda_coral,
                              load_in_4bit, gpu_note, whiten_text, run_tag, eval_816_shard,
                              use_text_cache, accum_steps, bank_negatives, peak_lr,
                              d_resampler, n_query, fn_mask_threshold, soft_label_beta)


# --------------------------------------------------------------------------- #
# 7b. floor_audit — ZERO-training diagnosis of the Step-3 loss saturation
#     (docs/next_steps.md revised step (a)). Needs only the text caches + the
#     probe ckpt: no base model, no frames, no audio. Answers, for ~$1:
#       * what loss would a PERFECT connector get at the probe's exact config?
#         (if that floor ~= the observed ~4.0-5.0, semantic near-dups explain
#         the saturation; if far below, the residual is capacity/optimization)
#       * how much of the floor is the unmasked full-corpus bank's own doing?
#       * what would whitened-cosine FN-masking buy at each threshold?
#       * is the learned temperature pinned at its clamp?
#       * corpus-wide census of semantic near-dup captions, with examples.
# --------------------------------------------------------------------------- #
@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=2 * 3600,
              memory=65536, cpu=4.0, env=HF_ENV)
def floor_audit(frame_shard: str = "audiocaps10k_sharded,fsd50k_train,wavcaps_audioset_sl_full",
                ckpt_name: str = ("p1frames_audiocaps10k_sharded,fsd50k_train,"
                                  "wavcaps_audioset_sl_full_step1800_probe131k_bf16.pt"),
                batch_size: int = 128, n_batches: int = 32,
                fn_thresholds: str = "0.99,0.97,0.95,0.9,0.85",
                stats_thresholds: str = "0.85,0.9,0.95,0.99",
                observed_loss: float = 4.5, limit_shards: int = 0, run_tag: str = "") -> dict:
    """Perfect-connector loss floor + semantic near-dup census on the probe's own corpus/config."""
    import dataclasses
    import json

    import torch

    from fusion_embedding.config import FusionConfig
    from fusion_embedding.losses import FusionContrastiveLoss
    from fusion_embedding.memory_bank import build_corpus_bank_from_cache
    from fusion_embedding.model import TextWhitening
    from fusion_embedding.paths import checkpoints_dir, frames_dir
    from fusion_embedding.train_stage1 import bank_neardup_stats, predict_loss_floor

    dev = "cuda"
    # 1) captions + shard paths in trainer order (mirrors _train_frames_impl's concatenation).
    shard_names = [s.strip() for s in str(frame_shard).split(",") if s.strip()]
    shard_paths, captions = [], []
    for nm in shard_names:
        fd = frames_dir(nm)
        with open(str(fd / "index.json")) as fh:
            idx = json.load(fh)
        shards, caps, ssz = idx["shards"], idx["captions"], int(idx["shard_size"])
        if limit_shards and limit_shards < len(shards):    # smoke mode: first k FULL shards/source
            shards = shards[:limit_shards]
            caps = caps[: len(shards) * ssz]
        shard_paths += [str(fd / s) for s in shards]
        captions += caps
    print(f"{len(shard_names)} source(s), {len(shard_paths)} shards, {len(captions)} captions")

    # 2) the probe ckpt: fitted whitening + final temperature + the run's exact loss config.
    ck = torch.load(str(checkpoints_dir() / ckpt_name), map_location="cpu", weights_only=False)
    flds = {f.name for f in dataclasses.fields(FusionConfig)}
    cfg = FusionConfig(**{k: v for k, v in ck["config"].items() if k in flds})
    whitening = TextWhitening(cfg.d_llm)
    whitening.load_state_dict(ck["text_whitening"])
    loss_fn = FusionContrastiveLoss(cfg)
    ls_raw = float(ck["logit_scale"].float().reshape(()))
    ls_clamped = min(ls_raw, cfg.logit_scale_max)          # training always passed the CLAMPED value
    at_clamp = ls_raw >= cfg.logit_scale_max - 1e-3
    print(f"logit_scale: raw {ls_raw:.4f} clamped {ls_clamped:.4f} (max {cfg.logit_scale_max:.4f}, "
          f"at_clamp={at_clamp}) | temp {1.0 / float(torch.tensor(ls_clamped).exp()):.5f}")

    # 3) whitened corpus bank straight from the Step-2 text caches (no base forwards at all).
    bank = build_corpus_bank_from_cache(shard_paths, captions, whitening, device=dev)
    print(f"bank: {len(bank)} rows, {bank.n_duplicate_captions} in exact-dup caption groups")

    def _floor(tag, **kw):
        r = predict_loss_floor(bank.embs, captions, loss_fn, torch.tensor(ls_clamped),
                               batch_size=batch_size, n_batches=n_batches, **kw)
        print(f"floor[{tag}]: {r['floor_mean']:.4f} ±{r['floor_std']:.4f} "
              f"(masked rows/anchor {r['mean_masked_bank_rows_per_anchor']:.0f})")
        return r

    floors = {
        "observed_config_bank_exactdup_mask": _floor("observed", use_bank=True),
        "inbatch_only_no_bank": _floor("no-bank", use_bank=False),
    }
    for t in (float(x) for x in fn_thresholds.split(",") if x.strip()):
        floors[f"fn_masked@{t:g}"] = _floor(f"fn@{t:g}", use_bank=True, fn_mask_threshold=t)
    # context: the same observed config at the INIT temperature (how much the learned temp matters)
    r_init = predict_loss_floor(bank.embs, captions, loss_fn, torch.tensor(cfg.logit_scale_init),
                                batch_size=batch_size, n_batches=n_batches, use_bank=True)
    floors["observed_config_at_init_scale"] = r_init

    stats = bank_neardup_stats(bank.embs, captions, dim=cfg.mrl_default,
                               thresholds=tuple(float(x) for x in stats_thresholds.split(",") if x.strip()))
    print("near-dup census:", json.dumps({k: v for k, v in stats.items() if k != "examples"}, indent=2))
    for ex in stats["examples"][:10]:
        print(f"  cos {ex['cos']:.3f} | {ex['a'][:70]!r} <> {ex['b'][:70]!r}")

    out = {"frame_shard": frame_shard, "ckpt": ckpt_name, "n_corpus": len(bank),
           "n_exact_dup_rows": bank.n_duplicate_captions, "limit_shards": limit_shards,
           "batch_size": batch_size, "n_batches": n_batches,
           "logit_scale_raw": ls_raw, "logit_scale_clamped": ls_clamped, "at_clamp": at_clamp,
           "observed_train_loss_reference": observed_loss,
           "floors": floors, "neardup_stats": stats}
    name = f"floor_audit_{run_tag or 'probe131k'}.json"
    with open(str(checkpoints_dir() / name), "w") as fh:
        json.dump(out, fh, indent=2)
    volume.commit()
    print(f"FLOOR_AUDIT saved -> checkpoints/{name}")
    return out


# --------------------------------------------------------------------------- #
# 8. rescore_frames — EVAL HYGIENE: re-score a saved connector checkpoint with
#    the richer protocol (R@1/5/10 + mAP@10, diagonal AND semantic-duplicate-
#    aware). Quantifies how much of the low R@1 is a multi-relevant metric artifact.
# --------------------------------------------------------------------------- #
@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=3 * 3600,
              memory=16384, env=HF_ENV)
def rescore_frames(frame_shard: str = "audiocaps10k_post_proj", steps: int = 4000,
                   eval_size: int = 200, load_in_4bit: bool = True,
                   sem_thresholds: str = "0.9,0.95", run_tag: str = "") -> dict:
    """Reproduce the training eval split, load the saved connector, report R@k + mAP@10
    both diagonally and crediting near-duplicate captions as relevant. No training."""
    import json
    import torch

    from fusion_embedding.config import FusionConfig
    from fusion_embedding.model import FusionEmbeddingModel
    from fusion_embedding.data import (
        InMemoryFrameDataset, FrameCollator, load_frame_clips, shard_starts_from,
    )
    from fusion_embedding.hf_components import load_base
    from fusion_embedding.paths import frames_dir, checkpoints_dir
    from fusion_embedding.train_stage1 import (
        encode_dataset, lexical_relevance, retrieval_report, semantic_relevance,
    )

    dev = "cuda"
    fdir = frames_dir(frame_shard)
    with open(str(fdir / "index.json")) as fh:
        index = json.load(fh)
    d_audio = int(index["d_audio"])

    # Reproduce the EXACT held-out eval split _train_frames_impl used (first N unique captions).
    eval_caps, eval_captions = set(), []
    if "shards" in index:                                       # sharded format
        shard_size = int(index["shard_size"])
        shard_paths = [str(fdir / s) for s in index["shards"]]
        eval_gidx = []
        for gi, cap in enumerate(index["captions"]):
            if len(eval_gidx) < eval_size and cap not in eval_caps:
                eval_gidx.append(gi); eval_caps.add(cap); eval_captions.append(cap)
        starts = shard_starts_from(len(shard_paths), shard_size, len(index["captions"]))
        eval_ds = InMemoryFrameDataset(load_frame_clips(shard_paths, starts, eval_gidx))
    else:                                                       # legacy per-clip format
        eval_paths = []
        for it in index["items"]:
            cap = it["caption"]
            if len(eval_paths) < eval_size and cap not in eval_caps:
                eval_paths.append(str(fdir / it["file"])); eval_caps.add(cap); eval_captions.append(cap)
        eval_ds = InMemoryFrameDataset.from_paths(eval_paths, half=True)
    print(f"eval={len(eval_captions)} unique-caption clips | d_audio={d_audio}")

    cfg0 = FusionConfig(n_query=64, d_resampler=256)
    cfg, embed_tokens, base_lm, tokenizer = load_base(
        cfg0, device=dev, dtype=torch.bfloat16, load_in_4bit=load_in_4bit, d_audio=d_audio)
    model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder=None)
    model.resampler.to(dev).float()

    ckpt_path = str(checkpoints_dir() / f"p1frames_{frame_shard}_step{steps}{run_tag}.pt")
    ckpt = torch.load(ckpt_path, map_location=dev)
    model.resampler.load_state_dict(ckpt["resampler"])
    if isinstance(model.logit_scale, torch.nn.Parameter):
        model.logit_scale.data = ckpt["logit_scale"].to(dev)
    if "text_whitening" in ckpt:            # reproduce the whitened text geometry at eval
        model.text_whitening.load_state_dict(ckpt["text_whitening"])
        print(f"loaded text_whitening (fitted={int(model.text_whitening.fitted)})")
    print(f"loaded {ckpt_path}")

    collator = FrameCollator(cfg, _HFTok(tokenizer, cfg))
    a, t = encode_dataset(model, eval_ds, collator, device=dev)

    out = {"frame_shard": frame_shard, "steps": steps, "n_eval": len(eval_paths),
           "diagonal": retrieval_report(a, t)}
    # embedding-cosine grouping (confounded by frozen-text anisotropy — kept for diagnosis)
    for thr in [float(x) for x in sem_thresholds.split(",")]:
        rel = semantic_relevance(t, threshold=thr)
        n_extra = int(rel.sum().item() - rel.size(0))     # off-diagonal relevant pairs credited
        rep = retrieval_report(a, t, relevance=rel)
        rep["_off_diagonal_relevant_pairs"] = n_extra
        out[f"grouped_cosine@{thr}"] = rep
    # lexical (word-overlap) grouping — anisotropy-free, the honest metric-artifact estimate
    for jthr in (0.5, 0.7):
        rel = lexical_relevance(eval_captions, threshold=jthr).to(a.device)
        rep = retrieval_report(a, t, relevance=rel)
        rep["_off_diagonal_relevant_pairs"] = int(rel.sum().item() - rel.size(0))
        out[f"grouped_lexical@{jthr}"] = rep
    out["run_tag"] = run_tag
    out["text_whitening_fitted"] = int(model.text_whitening.fitted)
    with open(str(checkpoints_dir() / f"rescore_{frame_shard}_step{steps}{run_tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    volume.commit()
    print("RESCORE:", json.dumps(out, indent=2))
    return out


def _score_816_protocol(model, cfg, collator, eval_shard, *, device, dim=0, task="sound",
                        id_allowlist=None) -> dict:
    """Score a LOADED model on a multi-caption eval set (min-rank over the N refs per clip).

    Shared by ``rescore_816`` (post-hoc, from a ckpt) and ``_train_frames_impl`` (end-of-run, so every
    training job reports the paper-comparable number). Encodes eval audio (one/clip) and ALL reference
    captions SEPARATELY, builds the ``[n_clips, n_caps]`` relevance via ``multicaption_relevance``.
    Read-only: never touches the frozen base params (drift stays 0).

    ``id_allowlist`` (a set of clip-id strings): restrict scoring to the exact canonical split, e.g.
    the standard AudioCaps-957/816 test ids — for published-claim comparability vs our fuller pool.
    """
    import json
    import torch

    from fusion_embedding.model import mrl_truncate_normalize
    from fusion_embedding.data import (
        InMemoryFrameDataset, load_frame_clips, shard_starts_from, instruction_for,
    )
    from fusion_embedding.paths import frames_dir
    from fusion_embedding.train_stage1 import (
        encode_dataset, filter_clips_by_allowlist, flatten_caption_groups, multicaption_relevance,
        retrieval_report,
    )

    fdir = frames_dir(eval_shard)
    with open(str(fdir / "index.json")) as fh:
        index = json.load(fh)
    caps_multi = index["captions_multi"]                        # list-of-lists, one per clip
    clip_indices = list(range(len(caps_multi)))
    n_total_clips = len(caps_multi)
    if id_allowlist is not None:                                # restrict to the exact canonical split
        clip_ids = index.get("clip_ids")
        if not clip_ids:
            raise RuntimeError(f"{eval_shard} index has no clip_ids — re-ingest to enable split filtering")
        clip_indices, caps_multi = filter_clips_by_allowlist(caps_multi, clip_ids, id_allowlist)
        if not clip_indices:
            raise RuntimeError("id_allowlist matched 0 clips — check the id format vs index clip_ids")
    n_clips = len(caps_multi)
    shard_size = int(index["shard_size"])
    shard_paths = [str(fdir / s) for s in index["shards"]]
    starts = shard_starts_from(len(shard_paths), shard_size, index["n_total"])
    eval_ds = InMemoryFrameDataset(load_frame_clips(shard_paths, starts, clip_indices))
    flat_caps, group_ids = flatten_caption_groups(caps_multi)   # (caption text, owning clip index)
    dim = int(dim) or cfg.mrl_default

    audio_emb, _ = encode_dataset(model, eval_ds, collator, dim=dim, device=device)   # [n_clips, dim]
    instr = instruction_for(task)
    was_training = model.training
    model.eval()
    t_chunks, bs = [], 64
    with torch.no_grad():                                       # ALL refs: encode_text -> whiten -> MRL
        for i in range(0, len(flat_caps), bs):
            ids, mask = collator._text_ids(
                [{"instruction": instr, "text": c} for c in flat_caps[i:i + bs]])
            raw = model.encode_text(ids.to(device), mask.to(device))
            t_chunks.append(mrl_truncate_normalize(model.text_whitening(raw), dim).cpu())
    if was_training:
        model.train()
    text_emb = torch.cat(t_chunks)                              # [n_caps, dim]

    rel = multicaption_relevance(group_ids, n_audio=n_clips)
    report = retrieval_report(audio_emb, text_emb, relevance=rel)
    return {"eval_shard": eval_shard, "n_clips": n_clips, "n_captions": len(flat_caps), "dim": dim,
            "n_total_clips": n_total_clips, "split_filtered": id_allowlist is not None,
            "protocol": "min-rank-over-refs", **report}


@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=3 * 3600,
              memory=32768, env=HF_ENV)
def rescore_816(eval_shard: str = "audiocaps_test816", ckpt_shard: str = "audiocaps10k_post_proj",
                steps: int = 4000, run_tag: str = "", load_in_4bit: bool = True,
                task: str = "sound", dim: int = 0, id_allowlist_file: str = "") -> dict:
    """Score a saved connector on the PUBLISHED multi-caption protocol (min-rank over the 5 refs).

    ``eval_shard`` is a multi-caption eval set built by ``ingest_audiocaps_eval`` (index has
    ``captions_multi``: one caption-list per clip). Directly comparable to AudioCaps/Clotho tables.
    ``id_allowlist_file`` (optional): a JSON list of canonical clip-ids on the Volume — restricts
    scoring to the exact standard split for published claims (needs an index with ``clip_ids``).
    """
    import json
    import torch

    from fusion_embedding.config import FusionConfig
    from fusion_embedding.model import FusionEmbeddingModel
    from fusion_embedding.data import FrameCollator
    from fusion_embedding.hf_components import load_base
    from fusion_embedding.paths import frames_dir, checkpoints_dir

    dev = "cuda"
    with open(str(frames_dir(eval_shard) / "index.json")) as fh:
        d_audio = int(json.load(fh)["d_audio"])
    id_allowlist = None
    if id_allowlist_file:
        with open(id_allowlist_file) as fh:
            id_allowlist = set(json.load(fh))
        print(f"exact-split filter: {len(id_allowlist)} canonical ids from {id_allowlist_file}")

    cfg0 = FusionConfig(n_query=64, d_resampler=256)
    cfg, embed_tokens, base_lm, tokenizer = load_base(
        cfg0, device=dev, dtype=torch.bfloat16, load_in_4bit=load_in_4bit, d_audio=d_audio)
    model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder=None)
    model.resampler.to(dev).float()

    ckpt_path = str(checkpoints_dir() / f"p1frames_{ckpt_shard}_step{steps}{run_tag}.pt")
    ckpt = torch.load(ckpt_path, map_location=dev)
    model.resampler.load_state_dict(ckpt["resampler"])
    if isinstance(model.logit_scale, torch.nn.Parameter):
        model.logit_scale.data = ckpt["logit_scale"].to(dev)
    if "text_whitening" in ckpt:
        model.text_whitening.load_state_dict(ckpt["text_whitening"])
    print(f"loaded {ckpt_path} | whitening fitted={int(model.text_whitening.fitted)}")

    collator = FrameCollator(cfg, _HFTok(tokenizer, cfg))
    out = {"ckpt_shard": ckpt_shard, "steps": steps, "run_tag": run_tag,
           "text_whitening_fitted": int(model.text_whitening.fitted),
           **_score_816_protocol(model, cfg, collator, eval_shard, device=dev, dim=dim, task=task,
                                 id_allowlist=id_allowlist)}
    with open(str(checkpoints_dir() / f"score816_{eval_shard}__{ckpt_shard}_step{steps}{run_tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    volume.commit()
    print("SCORE816:", json.dumps(out, indent=2))
    return out


@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=12 * 3600,
              memory=32768, env=HF_ENV)
def precompute_text_cache(frame_shard: str = "audiocaps10k_sharded", batch: int = 64,
                          load_in_4bit: bool = True) -> dict:
    """STEP 2: cache the frozen-base RAW pooled text embedding for every clip, beside its frame shard.

    Because the base + captions are frozen, text embeddings are deterministic — precompute them ONCE so
    training skips the text-side base forward each step (~2× fewer base forwards) and can later hold a
    large text-negative bank. Writes ``shard-NNNN.txtemb.pt`` (fp16 ``[n_clips, d_llm]``, RAW/pre-
    whitening) next to each frame shard and sets ``text_cache=True`` in the index. Comma-sep = multi-src.
    """
    import json

    import torch

    from fusion_embedding.config import FusionConfig
    from fusion_embedding.model import FusionEmbeddingModel
    from fusion_embedding.data import FrameCollator, instruction_for, write_text_emb_shard
    from fusion_embedding.hf_components import load_base
    from fusion_embedding.paths import frames_dir
    from fusion_embedding.data import TASK_INSTRUCTIONS

    dev = "cuda"
    shard_names = [s.strip() for s in str(frame_shard).split(",") if s.strip()]
    fd0 = frames_dir(shard_names[0])
    with open(str(fd0 / "index.json")) as fh:
        d_audio = int(json.load(fh)["d_audio"])

    cfg0 = FusionConfig(n_query=64, d_resampler=256)
    cfg, embed_tokens, base_lm, tokenizer = load_base(
        cfg0, device=dev, dtype=torch.bfloat16, load_in_4bit=load_in_4bit, d_audio=d_audio)
    model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder=None).eval()
    collator = FrameCollator(cfg, _HFTok(tokenizer, cfg))
    print(f"d_llm={cfg.d_llm} | caching text for {len(shard_names)} source(s)")

    total = 0
    for nm in shard_names:
        fd = frames_dir(nm)
        with open(str(fd / "index.json")) as fh:
            idx = json.load(fh)
        for sname in idx["shards"]:
            sp = fd / sname
            shard = torch.load(str(sp), map_location="cpu", weights_only=False)
            texts, tasks = shard["text"], shard["task"]
            embs = []
            for i in range(0, len(texts), batch):
                items = [{"instruction": instruction_for(t if t in TASK_INSTRUCTIONS else "sound"),
                          "text": tx} for tx, t in zip(texts[i:i + batch], tasks[i:i + batch])]
                ids, mask = collator._text_ids(items)
                with torch.no_grad():
                    raw = model.encode_text(ids.to(dev), mask.to(dev))   # RAW pooled [B, d_llm]
                embs.append(raw.float().cpu())
            write_text_emb_shard(sp, torch.cat(embs))
            total += len(texts)
            volume.commit()
            print(f"  {nm}/{sname}: cached {len(texts)} (total {total})", flush=True)
        idx["text_cache"] = True; idx["d_llm"] = cfg.d_llm
        with open(str(fd / "index.json"), "w") as fh:
            json.dump(idx, fh)
        volume.commit()
    result = {"frame_shard": frame_shard, "cached_clips": total, "d_llm": cfg.d_llm,
              "sources": len(shard_names)}
    print(f"TEXT_CACHE: {result}")
    return result


def _infinite_loader(loader):
    """Yield batches forever, re-iterating (a fresh reshuffled epoch) each pass.

    Unlike ``itertools.cycle``, this does NOT cache — required for streaming IterableDatasets
    (cycle would buffer the whole first epoch in RAM, defeating the point of sharding).
    """
    while True:
        for b in loader:
            yield b


def _train_frames_impl(frame_shard, steps, batch_size, eval_size, lambda_coral,
                       load_in_4bit, gpu_note, whiten_text=True, run_tag="",
                       eval_816_shard="audiocaps_test816", use_text_cache=False,
                       accum_steps=1, bank_negatives=False, peak_lr=0.0,
                       d_resampler=256, n_query=64, fn_mask_threshold=0.0,
                       soft_label_beta=0.0) -> dict:
    import glob
    import itertools
    import json
    import os

    # Set BEFORE torch initialises its CUDA allocator. The frozen base LM's audio forward keeps
    # full activations (grads must reach the resampler), so variable-length batches fragment the
    # 80GB pool until a mid-run allocation fails ("CUDA OOM ... reserved but unallocated", probe
    # crash 2026-07-02 step 50). expandable_segments lets the allocator grow/reclaim segments
    # instead of fragmenting — the fix the OOM error itself recommends.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from torch.utils.data import DataLoader

    # Multi-worker DataLoaders ship batch tensors between workers and the main process through
    # shared memory. Modal's default /dev/shm is tiny (~64 MB), so at 100K+ scale the prefetched
    # frame tensors overflow it and workers die with "Bus error ... out of shared memory"
    # (probe crash 2026-07-02, step 50). The 'file_system' strategy moves that transfer to /tmp
    # (container-RAM-sized, tens of GB) instead — the canonical fix for this exact bus error.
    import torch.multiprocessing as _torch_mp
    _torch_mp.set_sharing_strategy("file_system")

    from fusion_embedding.config import FusionConfig
    from fusion_embedding.model import FusionEmbeddingModel
    from fusion_embedding.losses import FusionContrastiveLoss
    from fusion_embedding.data import InMemoryFrameDataset, FrameCollator
    from fusion_embedding.hf_components import load_base
    from fusion_embedding.paths import frames_dir, checkpoints_dir
    from fusion_embedding.train_stage1 import (
        RegressionGuard, build_optimizer, build_scheduler, encode_dataset, fit_text_whitening,
        fit_text_whitening_from_cache, retrieval_report, save_resume_ckpt, load_resume_ckpt,
    )

    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} {gpu_note}")
    # frame_shard may be a comma-separated list of shards to MIX (all must be the sharded format).
    shard_names = [s.strip() for s in str(frame_shard).split(",") if s.strip()]
    loaded = []
    for nm in shard_names:
        fd = frames_dir(nm); ip = fd / "index.json"
        if not ip.exists():
            raise RuntimeError(f"no index.json in {fd} — run precompute_frames / ingest first")
        with open(str(ip)) as fh:
            loaded.append((nm, fd, json.load(fh)))
    d_audio = int(loaded[0][2]["d_audio"])
    sharded = all("shards" in idx for _, _, idx in loaded)
    if len(shard_names) > 1 and not sharded:
        raise RuntimeError("multi-shard training requires every shard in the new sharded format")
    if use_text_cache:
        if not sharded:
            raise RuntimeError("use_text_cache requires the sharded frame format")
        missing = [nm for nm, _, idx in loaded if not idx.get("text_cache")]
        if missing:
            raise RuntimeError(f"text cache not built for {missing} — run precompute_text_cache first")
    if bank_negatives and not use_text_cache:
        raise RuntimeError("bank_negatives requires use_text_cache (the bank IS the text cache)")
    accum_steps = max(1, int(accum_steps))

    # peak_lr>0 overrides the config default (1e-4) — large-batch runs need it scaled up
    # (measured 2026-07-01: eff-1024 at lr 1e-4 with 4× fewer opt steps was LR-starved).
    lr_kw = {"lr": float(peak_lr)} if peak_lr else {}
    # d_resampler/n_query are the HLD §4.2 capacity dials — parameterized for the capacity A/B.
    cfg0 = FusionConfig(n_query=int(n_query), d_resampler=int(d_resampler),
                        fn_mask_threshold=float(fn_mask_threshold),
                        soft_label_beta=float(soft_label_beta),
                        lambda_coral=lambda_coral, max_steps=steps,
                        **lr_kw)
    cfg, embed_tokens, base_lm, tokenizer = load_base(
        cfg0, device=dev, dtype=torch.bfloat16, load_in_4bit=load_in_4bit,
        gradient_checkpointing=True, d_audio=d_audio)
    print(f"dims: d_llm={cfg.d_llm} d_audio={cfg.d_audio} | {'SHARDED' if sharded else 'legacy'} "
          f"| {len(shard_names)} source(s)")

    model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder=None)  # no tower at train time
    model.resampler.to(dev).float()
    if isinstance(model.logit_scale, torch.nn.Parameter):
        model.logit_scale.data = model.logit_scale.data.to(dev)
    collator = FrameCollator(cfg, _HFTok(tokenizer, cfg))

    if sharded:
        # Streaming: one big sequential read per shard, no full-RAM preload. Concatenate sources with
        # a running global offset (partial last shards handled by explicit shard_starts). Eval = first
        # N unique captions (by global index), materialised into RAM; everything else streams.
        from fusion_embedding.data import ShardedFrameDataset, load_frame_clips
        shard_paths, captions, shard_starts, running = [], [], [], 0
        for nm, fd, idx in loaded:
            ssz = int(idx["shard_size"]); caps = idx["captions"]; shards = idx["shards"]
            for p, s in enumerate(shards):
                shard_paths.append(str(fd / s)); shard_starts.append(running)
                cnt = ssz if p < len(shards) - 1 else (len(caps) - ssz * (len(shards) - 1))
                running += cnt
            captions += caps
        eval_caps, eval_gidx = set(), []
        for gi, cap in enumerate(captions):
            if len(eval_gidx) < eval_size and cap not in eval_caps:
                eval_gidx.append(gi); eval_caps.add(cap)
        n_train_report = len(captions) - len(eval_gidx)
        print(f"sharded: {len(shard_names)} source(s), {len(shard_paths)} shards, {len(captions)} clips "
              f"| train~{n_train_report} eval={len(eval_gidx)} | text_cache={use_text_cache}")
        eval_ds = InMemoryFrameDataset(
            load_frame_clips(shard_paths, shard_starts, eval_gidx, with_text_emb=use_text_cache))
        train_ds = ShardedFrameDataset(shard_paths, shard_starts, exclude=set(eval_gidx),
                                       shuffle_buffer=4096, seed=0, use_text_emb=use_text_cache)
        # num_workers=2, prefetch_factor=2 keeps at most ~4 batches resident (was 16 -> shm blowout);
        # combined with the file_system sharing strategy above this holds at 100K+ scale.
        loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collator, num_workers=2,
                            persistent_workers=True, prefetch_factor=2, drop_last=True)
    else:
        # Legacy per-clip .pt files: preload frames into RAM once (still supported for old shards).
        _, fdir, index = loaded[0]
        eval_caps, eval_paths, train_paths = set(), [], []
        for it in index["items"]:
            p = str(fdir / it["file"]); cap = it["caption"]
            if len(eval_paths) < eval_size and cap not in eval_caps:
                eval_paths.append(p); eval_caps.add(cap)
            else:
                train_paths.append(p)
        print(f"legacy: train={len(train_paths)} eval={len(eval_paths)} — preloading into RAM...", flush=True)
        train_ds = InMemoryFrameDataset.from_paths(train_paths, half=True, log_every=500)
        eval_ds = InMemoryFrameDataset.from_paths(eval_paths, half=True)
        loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collator,
                            shuffle=True, drop_last=True)
        n_train_report = len(train_paths)

    # Fit per-dim text whitening BEFORE training (anisotropy fix): the connector then learns to align
    # to whitened text targets. Stats are saved in the ckpt so eval reproduces. With the Step 2 cache
    # this reads precomputed RAW text vectors (no base forward) via a short pass over the loader.
    wstats = None
    if whiten_text:
        if use_text_cache:
            fit_loader = DataLoader(train_ds, batch_size=64, collate_fn=collator, num_workers=2)
            wstats = fit_text_whitening_from_cache(model, fit_loader, device=dev, max_samples=4096)
            del fit_loader
        else:
            wstats = fit_text_whitening(model, train_ds, collator, device=dev, max_samples=4096)
        print(f"text whitening fitted: {wstats}")

    # Step 3: full-corpus frozen-text negative bank (A→T denominator). Zero staleness — text is
    # frozen and cached; whitened once (fixed diagonal after fit). Eval clips excluded so eval
    # captions are never training negatives; each anchor's own caption is masked per batch.
    bank = None
    if bank_negatives:
        from fusion_embedding.memory_bank import build_corpus_bank_from_cache
        bank = build_corpus_bank_from_cache(shard_paths, captions, model.text_whitening,
                                            exclude=set(eval_gidx), device=dev)
        print(f"text-negative bank: {len(bank)} entries "
              f"({bank.n_duplicate_captions} rows in duplicate-caption groups)")
    loss_fn = FusionContrastiveLoss(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, steps)
    guard = RegressionGuard(model)

    # Preemption resilience: Modal restarts a preempted function from the top with the same input,
    # so we persist the trainable state every SAVE_EVERY steps and resume from the last save instead
    # of step 0 (a preemption at step 850 previously wiped ~4.5h — 2026-07-02). Base/whitening/bank
    # are re-derived deterministically above, so only resampler+optim+sched+step are checkpointed.
    SAVE_EVERY = 100
    resume_path = str(checkpoints_dir() / f"resume_{frame_shard.replace(',', '-')}{run_tag}.pt")
    # Fingerprint the arm-defining config: a resume must NEVER cross A/B arms (different
    # d_resampler = shape error; same shapes but different lr/batch = silent corruption).
    resume_key = (f"{frame_shard}|d{cfg.d_resampler}|N{cfg.n_query}|b{batch_size}x{accum_steps}"
                  f"|lr{peak_lr or cfg.lr}|bank{int(bank_negatives)}|4bit{int(load_in_4bit)}"
                  f"|wh{int(whiten_text)}|fn{cfg.fn_mask_threshold}|sl{cfg.soft_label_beta}")
    start_step = load_resume_ckpt(resume_path, model, opt, sched, total_steps=steps,
                                  config_key=resume_key)
    print(f"RESUME: continuing from step {start_step}/{steps}" if start_step > 0
          else "no resume checkpoint — training from step 0")

    model.train()
    di = _infinite_loader(loader)
    torch.cuda.reset_peak_memory_stats()
    hist = []
    diverged_at = None
    import time as _time
    torch.cuda.synchronize()
    _t0 = _time.time()
    oom_error = None
    for step in range(start_step, steps):
        opt.zero_grad(set_to_none=True)
        # grad accumulation: effective batch = batch_size × accum_steps for optimizer quality.
        # NOTE accumulation does NOT add in-batch negatives — the bank is what scales negatives.
        try:
            for _micro in range(accum_steps):
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in next(di).items()}
                out = model(batch)
                if bank is not None:
                    loss, m = loss_fn(out["audio"], out["text"], out["logit_scale"],
                                      bank_text=bank.embs,
                                      bank_exclude_mask=bank.exclude_mask(batch["texts"], device=dev))
                else:
                    loss, m = loss_fn(out["audio"], out["text"], out["logit_scale"])
                # Divergence guard: a non-finite loss never recovers — stop NOW instead of burning
                # the remaining steps (and never backprop NaNs into the resampler/optimizer state).
                if not torch.isfinite(loss):
                    diverged_at = step
                    break
                (loss / accum_steps).backward()
        except torch.cuda.OutOfMemoryError as e:
            # OOM guard: a config that doesn't fit never will — report a RESULT instead of
            # raising, so Modal's auto-retry doesn't rerun a deterministic failure 3x (the
            # d512@b128 incident, 2026-07-03: 2 wasted retry attempts ~$3).
            oom_error = str(e).split("\n")[0]
            print(f"OOM at step {step}: {oom_error}", flush=True)
            break
        if diverged_at is not None:
            print(f"DIVERGED: non-finite loss at step {step} (lr {sched.get_last_lr()[0]:.2e}) — "
                  f"stopping early; resume ckpt kept at step {((step // SAVE_EVERY) * SAVE_EVERY) - 1}"
                  f" for post-mortem", flush=True)
            break
        gn = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), cfg.grad_clip)
        opt.step(); sched.step()
        if step % 50 == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(loss), "acc": float(m["acc_a2t"]),
                         "grad_norm": float(gn)})
            print(f"step {step:>4} loss {float(loss):.4f} acc {float(m['acc_a2t']):.3f} "
                  f"gnorm {float(gn):.2f} lr {sched.get_last_lr()[0]:.2e}")
        # Periodic resume checkpoint (skip the final step — the end-of-run ckpt covers it).
        if (step + 1) % SAVE_EVERY == 0 and step + 1 < steps:
            save_resume_ckpt(resume_path, model, opt, sched, step, steps, config_key=resume_key)
            volume.commit()
            print(f"  [resume ckpt saved @ step {step}]", flush=True)

    torch.cuda.synchronize()
    train_seconds = round(_time.time() - _t0, 1)
    ran_steps = steps - start_step                             # this session's steps (resume-aware)
    steps_per_min = round(ran_steps / (train_seconds / 60), 1) if train_seconds > 0 and ran_steps else None
    print(f"train loop: {train_seconds}s for {ran_steps} steps (resumed@{start_step}, {steps_per_min} steps/min)")

    if diverged_at is not None or oom_error is not None:
        # Report divergence/OOM as a RESULT, not an exception: raising would make Modal's
        # auto-retry rerun a deterministic failure; a status return stops cleanly. The resume
        # ckpt is kept for post-mortem (an OOM run relaunched with a SMALLER batch gets a
        # different config_key and correctly starts fresh).
        result = {"status": "oom" if oom_error is not None else "diverged",
                  "oom_error": oom_error, "diverged_at_step": diverged_at,
                  "gpu": torch.cuda.get_device_name(0), "frame_shard": frame_shard,
                  "d_resampler": cfg.d_resampler,
                  "batch_size": batch_size, "accum_steps": accum_steps,
                  "peak_lr": peak_lr or cfg.lr, "bank_negatives": len(bank) if bank is not None else 0,
                  "hist_tail": hist[-5:], "resumed_from_step": start_step,
                  "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  "train_seconds": train_seconds}
        with open(str(checkpoints_dir() / f"result_frames_{frame_shard}_step{steps}{run_tag}.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        volume.commit()
        print(f"TRAIN_FRAMES {result['status'].upper()}:", result)
        return result

    a, t = encode_dataset(model, eval_ds, collator, device=dev)
    rep = retrieval_report(a, t)

    # EXIT GATE: also score the PAPER-COMPARABLE min-rank-over-refs protocol on the held-out
    # multi-caption eval set (e.g. AudioCaps-883), so every run reports the number we track vs SOTA.
    # Read-only (frozen base untouched); a missing/bad eval shard warns but never fails the run.
    score816 = None
    if eval_816_shard and (frames_dir(eval_816_shard) / "index.json").exists():
        try:
            score816 = _score_816_protocol(model, cfg, collator, eval_816_shard, device=dev)
            print(f"816-protocol ({eval_816_shard}, {score816['n_clips']} clips): "
                  f"a2t R@1 {score816['a2t_R@1']:.3f} R@5 {score816['a2t_R@5']:.3f} "
                  f"R@10 {score816['a2t_R@10']:.3f} mAP@10 {score816['a2t_mAP@10']:.3f}")
        except Exception as e:                                  # noqa: BLE001
            print(f"WARN: 816-protocol scoring failed ({eval_816_shard}): {e}")
    elif eval_816_shard:
        print(f"WARN: eval_816_shard '{eval_816_shard}' has no index.json — skipping comparable score")

    drift = guard.max_drift(model)
    ckpt = str(checkpoints_dir() / f"p1frames_{frame_shard}_step{steps}{run_tag}.pt")
    torch.save({"resampler": model.resampler.state_dict(),
                "logit_scale": model.logit_scale.detach().cpu(),
                "text_whitening": model.text_whitening.state_dict(),
                "config": cfg.__dict__, "frame_shard": frame_shard, "steps": steps}, ckpt)
    result = {
        "gpu": torch.cuda.get_device_name(0), "frame_shard": frame_shard, "d_audio": d_audio,
        "sharded": sharded, "n_train": n_train_report, "n_eval": len(eval_ds),
        "whiten_text": whiten_text, "whiten_stats": wstats, "text_cache": use_text_cache,
        "batch_size": batch_size, "accum_steps": accum_steps,
        "effective_batch": batch_size * accum_steps,
        "bank_negatives": len(bank) if bank is not None else 0,
        "loss_first": hist[0]["loss"] if hist else None,
        "loss_last": hist[-1]["loss"] if hist else None,
        "resumed_from_step": start_step,
        "a2t_R@1": rep["a2t_R@1"], "a2t_R@5": rep["a2t_R@5"], "a2t_R@10": rep["a2t_R@10"],
        "a2t_mAP@10": rep["a2t_mAP@10"], "t2a_R@1": rep["t2a_R@1"], "t2a_mAP@10": rep["t2a_mAP@10"],
        "score816": score816,                                  # paper-comparable min-rank-over-refs
        "train_seconds": train_seconds, "steps_per_min": steps_per_min,
        "base_drift": drift, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "ckpt": ckpt,
    }
    with open(str(checkpoints_dir() / f"result_frames_{frame_shard}_step{steps}{run_tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    # Run finished cleanly — drop the resume checkpoint so a later run of the same tag starts fresh.
    if os.path.exists(resume_path):
        os.remove(resume_path)
    volume.commit()
    assert drift == 0.0, f"BASE LEAKED: {drift}"
    print("TRAIN_FRAMES:", result)
    return result


# --------------------------------------------------------------------------- #
# Local entrypoint: `modal run modal_app.py` runs the smoke by default.
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main():
    print(smoke.remote())
