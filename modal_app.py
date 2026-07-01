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
    .apt_install("ffmpeg", "libsndfile1")              # audio decode backends for librosa/soundfile
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
                      audio_feature_layer: str = "post_proj", batch: int = 16) -> dict:
    import glob
    import torch
    from fusion_embedding.hf_components import load_audio_tower
    from fusion_embedding.paths import features_dir, frames_dir

    dev = "cuda"
    frame_shard = frame_shard or f"{mel_shard}_{audio_feature_layer}"
    enc, _fe, d_audio = load_audio_tower(device=dev, dtype=torch.bfloat16,
                                         audio_feature_layer=audio_feature_layer)
    paths = sorted(glob.glob(str(features_dir(mel_shard) / "*.pt")))
    out_dir = frames_dir(frame_shard)
    print(f"encoding {len(paths)} clips -> frames d_audio={d_audio} -> {out_dir}")

    n = 0
    index = []                                                  # lightweight [{file, caption}] for fast splits
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
            fname = f"item-{start + i:05d}.pt"
            torch.save({"frames": frames[i, :t].cpu().contiguous(), "text": r["text"], "task": r["task"]},
                       str(out_dir / fname))
            index.append({"file": fname, "caption": r["text"], "task": r["task"]})
            n += 1
        if start % (batch * 20) == 0:
            print(f"  {n}/{len(paths)}")
    import json
    with open(str(out_dir / "index.json"), "w") as fh:          # captions without loading tensors
        json.dump({"d_audio": d_audio, "items": index}, fh)
    volume.commit()
    print(f"cached {n} frame files (+ index.json) -> {out_dir}")
    return {"frame_shard": frame_shard, "count": n, "d_audio": d_audio, "dir": str(out_dir)}


# --------------------------------------------------------------------------- #
# 7. train_frames — train the connector on PRECOMPUTED frames (base only, no 7B
#    tower). Much faster/step -> bigger batch + longer schedule for real P1.
# --------------------------------------------------------------------------- #
@app.function(gpu="L4", volumes={VOL: volume}, secrets=[hf_secret], timeout=12 * 3600,
              memory=32768, env=HF_ENV)
def train_frames(frame_shard: str = "audiocaps4k_post_proj", steps: int = 2000,
                 batch_size: int = 32, eval_size: int = 200, lambda_coral: float = 0.05,
                 load_in_4bit: bool = True, gpu_note: str = "") -> dict:
    import glob
    import itertools
    import json

    import torch
    from torch.utils.data import DataLoader

    from fusion_embedding.config import FusionConfig
    from fusion_embedding.model import FusionEmbeddingModel
    from fusion_embedding.losses import FusionContrastiveLoss
    from fusion_embedding.data import InMemoryFrameDataset, FrameCollator
    from fusion_embedding.hf_components import load_base
    from fusion_embedding.paths import frames_dir, checkpoints_dir
    from fusion_embedding.train_stage1 import (
        RegressionGuard, build_optimizer, build_scheduler, encode_dataset, retrieval_report,
    )

    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} {gpu_note}")
    fdir = frames_dir(frame_shard)
    idx_path = fdir / "index.json"
    if not idx_path.exists():
        raise RuntimeError(f"no index.json in {fdir} — run precompute_frames first")
    with open(str(idx_path)) as fh:
        index = json.load(fh)
    d_audio = int(index["d_audio"])
    print(f"{len(index['items'])} frame clips | d_audio={d_audio}")

    # held-out unique-caption eval; rest trains — using the lightweight index (no tensor loads)
    eval_caps, eval_paths, train_paths = set(), [], []
    for it in index["items"]:
        p = str(fdir / it["file"]); cap = it["caption"]
        if len(eval_paths) < eval_size and cap not in eval_caps:
            eval_paths.append(p); eval_caps.add(cap)
        else:
            train_paths.append(p)
    print(f"train={len(train_paths)} eval={len(eval_paths)}")

    cfg0 = FusionConfig(n_query=64, d_resampler=256, lambda_coral=lambda_coral, max_steps=steps)
    cfg, embed_tokens, base_lm, tokenizer = load_base(
        cfg0, device=dev, dtype=torch.bfloat16, load_in_4bit=load_in_4bit,
        gradient_checkpointing=True, d_audio=d_audio)
    print(f"dims: d_llm={cfg.d_llm} d_audio={cfg.d_audio}")

    model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder=None)  # no tower at train time
    model.resampler.to(dev).float()
    if isinstance(model.logit_scale, torch.nn.Parameter):
        model.logit_scale.data = model.logit_scale.data.to(dev)

    collator = FrameCollator(cfg, _HFTok(tokenizer, cfg))
    # Preload frames into RAM once (frames are ~7x mel; per-step Volume reads starve the GPU).
    print("preloading frames into RAM...", flush=True)
    train_ds = InMemoryFrameDataset.from_paths(train_paths, half=True, log_every=500)
    eval_ds = InMemoryFrameDataset.from_paths(eval_paths, half=True)
    print(f"preloaded train={len(train_ds)} eval={len(eval_ds)}")
    loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collator,
                        shuffle=True, drop_last=True)
    loss_fn = FusionContrastiveLoss(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, steps)
    guard = RegressionGuard(model)

    model.train()
    di = itertools.cycle(loader)
    torch.cuda.reset_peak_memory_stats()
    hist = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in next(di).items()}
        out = model(batch)
        loss, m = loss_fn(out["audio"], out["text"], out["logit_scale"])   # big batch = many real negs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), cfg.grad_clip)
        opt.step(); sched.step()
        if step % 50 == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(loss), "acc": float(m["acc_a2t"])})
            print(f"step {step:>4} loss {float(loss):.4f} acc {float(m['acc_a2t']):.3f} lr {sched.get_last_lr()[0]:.2e}")

    a, t = encode_dataset(model, eval_ds, collator, device=dev)
    rep = retrieval_report(a, t)
    drift = guard.max_drift(model)
    ckpt = str(checkpoints_dir() / f"p1frames_{frame_shard}_step{steps}.pt")
    torch.save({"resampler": model.resampler.state_dict(),
                "logit_scale": model.logit_scale.detach().cpu(),
                "config": cfg.__dict__, "frame_shard": frame_shard, "steps": steps}, ckpt)
    result = {
        "gpu": torch.cuda.get_device_name(0), "frame_shard": frame_shard, "d_audio": d_audio,
        "n_train": len(train_paths), "n_eval": len(eval_paths),
        "loss_first": hist[0]["loss"], "loss_last": hist[-1]["loss"],
        "a2t_R@1": rep["a2t_R@1"], "a2t_R@10": rep["a2t_R@10"], "t2a_R@1": rep["t2a_R@1"],
        "base_drift": drift, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "ckpt": ckpt,
    }
    with open(str(checkpoints_dir() / f"result_frames_{frame_shard}_step{steps}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
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
