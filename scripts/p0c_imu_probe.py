"""P0c — IMU kill-probe (Tremor gate 0): can IMU enter the frozen space cheaply?

Redesigned per docs/research_imu_deep.md §5.2 (supersedes the plan's single-arm
tokenizer bet, which the literature says is uninformative at 36K):

  * P0c-0 (zero-train floor): UCI-HAR test windows rendered as (a) line-plot
    grids and (b) STFT tile grids through the FROZEN image path, and (c) a naive
    mel-packed pseudo-spectrogram through the FROZEN released audio path.
    Zero-shot 6-class HAR via calibrated class-name ensembles (P0a harness
    lesson) + retrieval vs a random-embedding control.
  * P0c-A (audio-tower arm): tiny learned conv front-end mapping 6-ch IMU
    windows onto the 128-mel grid of the FROZEN released FE2 audio stack
    (tower + trained resampler + audio-gated adapters + LM all frozen; only the
    front-end + logit_scale train). Contrastive vs frozen whitened text targets
    of SensorCaps captions. GW-Whisper pattern in spirit: a small front-end
    feeding a tower the decoder already reads natively.
  * P0c-B (from-scratch tokenizer at Ego4D scale): DEFERRED — blocked on the
    Ego4D license signature (user action, ~48 h); running it at 36K would be
    uninformative (research §1.3).

Data (probe tier; licenses documented in the RESULT):
  * SensorCaps (BASH-Lab, HF; Hippocratic HL3): train = hhar_v2 + uci_v2 +
    shoaib_v2 + pamap2_v2; DISJOINT holdout = motion_v2 (MotionSense held out
    entirely -> cross-dataset AND participant-disjoint by construction).
  * UCI-HAR (research-classic) for the zero-train floor renders.
  * WISDM v1.1 (never in SensorCaps) as the style-independent ZS check
    (plan named CAPTURE-24; WISDM substituted -- tiny, form-free, truly
    never-trained; 3-axis accel only, gyro zero-padded, documented).

Pre-registered gates (plan §P0c, verbatim thresholds):
  * GO:   holdout text->IMU R@10 >= 10x random control AND MRR >= 0.09.
  * KILL: MRR < 0.05 (one architecture fallback allowed before killing).
  * WISDM ZS class-name eval reported as the style-independence check.
  * Ego4D 108-scenario ZS: not runnable (license pending) — recorded as blocked.

Run (deploy + spawn; each mode is one self-contained job):
    PYTHONUTF8=1 uv run modal deploy scripts/p0c_imu_probe.py
    ... Function.from_name("fusion-p0c-imu", "prep").spawn()          # CPU
    ... Function.from_name("fusion-p0c-imu", "run").spawn(mode="p0c0")
    ... Function.from_name("fusion-p0c-imu", "run").spawn(mode="armA", smoke=True)
    ... Function.from_name("fusion-p0c-imu", "run").spawn(mode="armA")
"""

from __future__ import annotations

import modal

app = modal.App("fusion-p0c-imu")
hf_secret = modal.Secret.from_name("huggingface")
volume = modal.Volume.from_name("fusion-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libsndfile1")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "numpy>=1.24",
        "transformers>=4.46", "accelerate>=0.30", "pillow>=10.0",
        "matplotlib>=3.8", "datasets>=2.19", "librosa>=0.10", "soundfile>=0.12",
        "scipy>=1.11",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"PYTHONUTF8": "1"})
    .add_local_dir(".", "/root/fe", copy=True,
                   ignore=["**/.git/**", "**/.venv/**", "**/_render/**",
                           "**/__pycache__/**", "**/results/**", "**/docs/**",
                           "release/**", "fe2_release/out/**", "assets/**",
                           "dist/**", "submission/**", "_staging_space/**",
                           "**/*.egg-info/**", "**/*.pt", "**/*.safetensors",
                           "**/*.zip", "**/*.png", "**/*.jpg"])
)

OUT = "/vol/p0c_imu"
CKPT_DIR = "/vol/checkpoints"
FE2_CKPT = "/vol/checkpoints/p1frames_audiocaps_train_full,ac2_new_step1600_ac2ft_1600.pt"
SR = 10          # nominal rate for STFT rendering (SensorCaps rate varies per source)
WIN_S = 5
TARGET_T = 250   # every window is resampled to this length before the front-end
UCI_URL = ("https://archive.ics.uci.edu/static/public/240/"
           "human+activity+recognition+using+smartphones.zip")
WISDM_URL = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
UCI_CLASSES = ["walking", "walking upstairs", "walking downstairs",
               "sitting", "standing", "laying down"]
WISDM_CLASSES = ["walking", "jogging", "walking upstairs", "walking downstairs",
                 "sitting", "standing"]
_WISDM_MAP = {"Walking": 0, "Jogging": 1, "Upstairs": 2, "Downstairs": 3,
              "Sitting": 4, "Standing": 5}

# class-name prompt ensembles (P0a lesson: ensemble + prior calibration)
def _prompts(cls: str):
    return [f"accelerometer and gyroscope recording of a person {cls}",
            f"wearable motion sensor signal of someone {cls}",
            f"IMU data captured while {cls}",
            f"body movement sensor readings of a person {cls}"]


# ---------------------------------------------------------------------------- #
# prep — CPU: SensorCaps parse, UCI-HAR + WISDM download/window, all -> Volume
# ---------------------------------------------------------------------------- #
@app.function(image=image, volumes={"/vol": volume}, timeout=3 * 3600,
              memory=16384, cpu=8, secrets=[hf_secret])
def prep() -> dict:
    import io
    import json
    import os
    import re
    import tarfile
    import urllib.request
    import zipfile

    import numpy as np

    os.makedirs(OUT, exist_ok=True)
    report = {}

    # ---------------- SensorCaps ----------------
    sc_path = os.path.join(OUT, "sensorcaps.npz")
    if not os.path.exists(sc_path):
        from datasets import load_dataset
        splits = {"train": ["hhar_v2", "uci_v2", "shoaib_v2", "pamap2_v2"],
                  "holdout": ["motion_v2"]}
        num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

        def _sensor_block(user, name):
            """Grab the [T,3] array serialized after `name:` — tolerant of both
            JSON-ish [[a,b,c],...] and numpy-repr [[a b c] ...] styles."""
            m = re.search(name + r"\s*:?\s*(\[\[.*?\]\])", user, re.S)
            if not m:
                return None
            vals = [float(x) for x in num_re.findall(m.group(1))]
            if len(vals) < 30 or len(vals) % 3 != 0:
                return None
            return np.array(vals, dtype=np.float32).reshape(-1, 3).T   # [3,T]

        def _resample(arr, T=None):
            T = T or TARGET_T
            idx_new = np.linspace(0, arr.shape[1] - 1, T)
            idx_old = np.arange(arr.shape[1])
            return np.stack([np.interp(idx_new, idx_old, ch) for ch in arr]).astype(np.float32)

        def parse_record(rec):
            """messages: user content embeds serialized 6-axis IMU; assistant = caption."""
            user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
            asst = next((m["content"] for m in rec["messages"] if m["role"] == "assistant"), "")
            acc = _sensor_block(user, "Accelerometer")
            gyr = _sensor_block(user, "Gyroscope")
            if acc is None or gyr is None or not asst.strip():
                return None
            T = min(acc.shape[1], gyr.shape[1])
            arr = np.concatenate([acc[:, :T], gyr[:, :T]], 0)          # [6,T]
            return _resample(arr), asst.replace("**", "").strip()

        data = {}
        for part, names in splits.items():
            xs, caps, srcs = [], [], []
            for name in names:
                ds = load_dataset("BASH-Lab/SensorCaps", split=name)
                ok = bad = 0
                if len(ds):
                    print(f"P0C_PREP {name}: n={len(ds)} sample_keys={list(ds[0])}",
                          flush=True)
                for rec in ds:
                    out = parse_record(rec)
                    if out is None:
                        bad += 1
                        continue
                    xs.append(out[0]); caps.append(out[1]); srcs.append(name); ok += 1
                print(f"P0C_PREP {name}: parsed {ok}, dropped {bad}", flush=True)
            data[part] = (np.stack(xs), caps, srcs)
        np.savez_compressed(
            sc_path,
            x_train=data["train"][0], x_hold=data["holdout"][0],
            cap_train=np.array(data["train"][1], dtype=object),
            cap_hold=np.array(data["holdout"][1], dtype=object),
            src_train=np.array(data["train"][2], dtype=object),
            src_hold=np.array(data["holdout"][2], dtype=object))
        volume.commit()
    z = np.load(sc_path, allow_pickle=True)
    report["sensorcaps"] = {"train": int(z["x_train"].shape[0]),
                            "holdout_motionsense": int(z["x_hold"].shape[0])}

    # ---------------- UCI-HAR (zero-train floor renders) ----------------
    uci_path = os.path.join(OUT, "ucihar_test.npz")
    if not os.path.exists(uci_path):
        buf = io.BytesIO(urllib.request.urlopen(UCI_URL, timeout=300).read())
        with zipfile.ZipFile(buf) as zf:
            inner = zf.read("UCI HAR Dataset.zip")
        with zipfile.ZipFile(io.BytesIO(inner)) as zf:
            def sig(name):
                raw = zf.read(f"UCI HAR Dataset/test/Inertial Signals/{name}_test.txt")
                return np.loadtxt(io.StringIO(raw.decode())).astype(np.float32)
            chans = [sig(f"total_acc_{a}") for a in "xyz"] + \
                    [sig(f"body_gyro_{a}") for a in "xyz"]
            x = np.stack(chans, axis=1)                       # [N,6,128] @50 Hz
            y = np.loadtxt(io.StringIO(
                zf.read("UCI HAR Dataset/test/y_test.txt").decode())).astype(int) - 1
        np.savez_compressed(uci_path, x=x, y=y)
        volume.commit()
    z = np.load(uci_path)
    report["ucihar_test"] = {"n": int(z["x"].shape[0])}

    # ---------------- WISDM v1.1 (never-trained ZS check; 3-ch accel) ----------
    wisdm_path = os.path.join(OUT, "wisdm.npz")
    if not os.path.exists(wisdm_path):
        buf = io.BytesIO(urllib.request.urlopen(WISDM_URL, timeout=300).read())
        rows = []
        with tarfile.open(fileobj=buf, mode="r:gz") as tf:
            raw = tf.extractfile("WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt").read().decode(
                "utf-8", errors="replace")
        for line in raw.replace(";", "\n").splitlines():
            p = [s for s in line.strip().split(",") if s != ""]
            if len(p) >= 6 and p[1] in _WISDM_MAP:
                try:
                    rows.append((int(p[0]), _WISDM_MAP[p[1]],
                                 float(p[3]), float(p[4]), float(p[5])))
                except ValueError:
                    continue
        arr = np.array(rows, dtype=np.float32)                 # [M,(subj,cls,x,y,z)] @20 Hz
        xs, ys = [], []
        step = 20 * WIN_S
        for subj in np.unique(arr[:, 0]):
            for cls in range(6):
                seg = arr[(arr[:, 0] == subj) & (arr[:, 1] == cls)][:, 2:5]
                for s0 in range(0, len(seg) - step + 1, step):
                    w20 = seg[s0:s0 + step].T                  # [3,100] @20 Hz
                    idx_new = np.linspace(0, w20.shape[1] - 1, TARGET_T)
                    idx_old = np.arange(w20.shape[1])
                    w = np.stack([np.interp(idx_new, idx_old, ch)
                                  for ch in w20]).astype(np.float32)   # [3,TARGET_T]
                    xs.append(np.concatenate([w, np.zeros_like(w)], 0))  # gyro zero-pad
                    ys.append(cls)
        x = np.stack(xs); y = np.array(ys)
        keep = np.random.RandomState(0).permutation(len(x))[:1800]
        np.savez_compressed(wisdm_path, x=x[keep], y=y[keep])
        volume.commit()
    z = np.load(wisdm_path)
    report["wisdm"] = {"n": int(z["x"].shape[0])}

    print("P0C_PREP_RESULT:", json.dumps(report), flush=True)
    return report


# ---------------------------------------------------------------------------- #
# helpers shared by both GPU modes
# ---------------------------------------------------------------------------- #
def _render_lineplot(w, sr):
    """[6,T] -> 2x3 line-plot grid PIL image (By-My-Eyes pattern)."""
    import matplotlib
    matplotlib.use("Agg")
    import io as _io

    import matplotlib.pyplot as plt
    from PIL import Image
    names = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
    fig, axes = plt.subplots(2, 3, figsize=(6, 3.6), dpi=80)
    t = [i / sr for i in range(w.shape[1])]
    for i, ax in enumerate(axes.flat):
        ax.plot(t, w[i], lw=0.9, color="black")
        ax.set_title(names[i], fontsize=7)
        ax.tick_params(labelsize=5)
    fig.tight_layout(pad=0.4)
    b = _io.BytesIO(); fig.savefig(b, format="png"); plt.close(fig)
    b.seek(0)
    return Image.open(b).convert("RGB")


def _render_stft_tiles(w, sr):
    """[6,T] -> 2x3 STFT log-magnitude tile grid PIL image."""
    import matplotlib
    matplotlib.use("Agg")
    import io as _io

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal as sps
    from PIL import Image
    fig, axes = plt.subplots(2, 3, figsize=(6, 3.6), dpi=80)
    for i, ax in enumerate(axes.flat):
        f, t, S = sps.stft(w[i], fs=sr, nperseg=min(16, w.shape[1]), noverlap=8)
        ax.imshow(np.log1p(np.abs(S)), aspect="auto", origin="lower", cmap="magma")
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    b = _io.BytesIO(); fig.savefig(b, format="png"); plt.close(fig)
    b.seek(0)
    return Image.open(b).convert("RGB")


def _naive_melpack(w, out_bins=128, out_t=500):
    """[6,T]@10Hz -> pseudo-mel [128,out_t]: per-channel STFT stacked into
    6 x ~21-bin bands, interpolated to the Whisper grid, log-scaled, standardized
    to mel-ish stats. The floor measurement (research doc: 'expected weak')."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy import signal as sps
    bands = []
    per = out_bins // 6
    for i in range(6):
        _f, _t, S = sps.stft(w[i], fs=SR, nperseg=16, noverlap=12)
        m = np.log1p(np.abs(S)).astype(np.float32)             # [9, T']
        t9 = torch.tensor(m)[None, None]
        band = F.interpolate(t9, size=(per, out_t), mode="bilinear",
                             align_corners=False)[0, 0]
        bands.append(band)
    mel = torch.cat(bands, 0)
    mel = torch.cat([mel, mel[: out_bins - mel.shape[0]]], 0) if mel.shape[0] < out_bins else mel[:out_bins]
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel * 0.8


def _load_embedder(dev):
    import sys
    sys.path.insert(0, "/root/fe")
    sys.path.insert(0, "/root/fe/fe2_release")
    from inference import FusionEmbedder
    emb = FusionEmbedder(FE2_CKPT, device=dev)
    return emb


def _batched_text(emb, texts, bs=48):
    """Whitened frozen text embeddings via the released text path, batched."""
    import torch
    import torch.nn.functional as F
    sys_ins = "Represent the user's input."
    outs = []
    tok = emb.tok
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            ids = [tok.encode(f"<|im_start|>system\n{sys_ins}<|im_end|>\n"
                              f"<|im_start|>user\n{t}<|im_end|>\n"
                              f"<|im_start|>assistant\n", add_special_tokens=False)[:512]
                   for t in chunk]
            L = max(len(x) for x in ids)
            pad = tok.pad_token_id or 0
            batch = torch.full((len(ids), L), pad, dtype=torch.long)
            mask = torch.zeros((len(ids), L), dtype=torch.long)
            for r, x in enumerate(ids):                        # LEFT pad: last token pools
                batch[r, L - len(x):] = torch.tensor(x)
                mask[r, L - len(x):] = 1
            pooled = emb.model.encode_text(batch.to(emb.device), mask.to(emb.device))
            pooled = emb.model.text_whitening(pooled)
            outs.append(F.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(outs, 0)


def _encoder_frames_grad(adapter, mel):
    """Grad-ENABLED replica of OmniAudioAdapter.forward (post_proj path). The released
    adapter is @torch.no_grad (built for the cached-frames inference path), which
    detaches a front-end placed BEFORE the encoder. The encoder PARAMS stay frozen;
    enabling grad here only lets the gradient flow THROUGH to the front-end's mel
    (the GW-Whisper design). post_proj only; single length T per batch (our pseudo-mel
    is fixed-length)."""
    import torch
    B, n_mels, Fdim = mel.shape
    enc = adapter.encoder
    dtype = next(enc.parameters()).dtype
    per = []
    for i in range(B):
        feats = mel[i].to(dtype)                                    # [n_mels, F]
        out = enc(input_features=feats,
                  feature_lens=torch.tensor([Fdim], device=mel.device))
        frames = out.last_hidden_state
        if frames.dim() == 3:
            frames = frames[0]
        per.append(frames.float())
    T = max(f.shape[0] for f in per)
    fo = mel.new_zeros(B, T, adapter.d_audio)
    fm = torch.zeros(B, T, dtype=torch.bool, device=mel.device)
    for i, f in enumerate(per):
        fo[i, :f.shape[0]] = f
        fm[i, :f.shape[0]] = True
    return fo, fm


def _imu_through_audio(emb, mels, grad=False):
    """pseudo-mel [B,128,T] -> pooled embedding via the frozen released audio path.
    grad=True routes through a grad-enabled encoder replica so a front-end placed
    before the encoder actually receives gradient (the no_grad adapter would zero it)."""
    import torch
    import torch.nn.functional as F
    dev = emb.device
    B = mels.shape[0]
    mels = mels.to(dev, dtype=torch.float32)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        if grad:
            frames, frame_mask = _encoder_frames_grad(emb.model.audio_encoder, mels)
            audio_tok = emb.model.resampler(frames, frame_mask)
        else:
            mask = torch.ones(B, mels.shape[2], dtype=torch.bool, device=dev)
            audio_tok = emb.model.audio_tokens(mels, mask)
        ids = torch.tensor([[emb.cfg.audio_pad_id] * emb.cfg.n_query + [emb.cfg.eos_id]],
                           device=dev).expand(B, -1).contiguous()
        pooled = emb.model.encode_audio(ids, torch.ones_like(ids), audio_tok)
        return F.normalize(pooled.float(), dim=-1)


def _calibrated_zs(img_embs, class_texts_fn, embed_text_fn, n_cls, y):
    """P0a harness: per-class prompt-ensemble embedding + prior calibration."""
    import torch
    cls_embs = []
    for c in range(n_cls):
        e = embed_text_fn(class_texts_fn(c))
        e = e.mean(0, keepdim=True)
        cls_embs.append(e / e.norm(dim=-1, keepdim=True))
    te = torch.cat(cls_embs, 0)                                # [C,D]
    logits = img_embs @ te.T                                   # [N,C]
    raw = (logits.argmax(1) == y).float().mean().item()
    cal = ((logits - logits.mean(0, keepdim=True)).argmax(1) == y).float().mean().item()
    return {"zs_raw": round(raw, 4), "zs_calibrated": round(cal, 4)}


# ---------------------------------------------------------------------------- #
# run — GPU: mode="p0c0" (zero-train floor) | mode="armA" (front-end train)
# ---------------------------------------------------------------------------- #
@app.function(gpu="A100-80GB", image=image, secrets=[hf_secret],
              volumes={"/vol": volume}, timeout=8 * 3600, memory=32768)
def run(mode: str = "p0c0", smoke: bool = False, seed: int = 1,
        fallback: bool = False) -> dict:
    import json
    import os
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    t0 = time.time()
    dev = "cuda"
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"{mode}{'_smoke' if smoke else ''}_s{seed}{'_fb' if fallback else ''}"

    result = {"probe": "p0c_imu", "mode": mode, "smoke": smoke, "seed": seed,
              "fallback": fallback, "fe2_ckpt": os.path.basename(FE2_CKPT),
              "sensorcaps_rate_hz": SR}

    # ---------------- P0c-0: zero-train floor ----------------
    if mode == "p0c0":
        z = np.load(os.path.join(OUT, "ucihar_test.npz"))
        x, y = z["x"], z["y"]
        n = 60 if smoke else 600
        sel = []
        rng = np.random.RandomState(0)
        for c in range(6):                                    # class-balanced
            idx = np.where(y == c)[0]
            sel.extend(rng.choice(idx, n // 6, replace=False).tolist())
        x, y = x[sel], torch.tensor(y[sel])
        print(f"P0C0 pool {x.shape} @50Hz", flush=True)

        # image path (base model, frozen)
        from transformers import AutoModel, AutoProcessor
        BASE = "Qwen/Qwen3-VL-Embedding-2B"
        base = AutoModel.from_pretrained(BASE, trust_remote_code=True,
                                         dtype=torch.bfloat16).to(dev).eval()
        proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)

        def _pool(h, m):
            i = m.long().cumsum(1).argmax(1)
            return h[torch.arange(h.shape[0], device=h.device), i]

        def embed_pils(pils, bs=16):
            outs = []
            txt = ("<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
                   "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                   "<|im_end|>\n<|im_start|>assistant\n")
            with torch.no_grad():
                for i in range(0, len(pils), bs):
                    inp = proc(text=[txt] * len(pils[i:i + bs]),
                               images=pils[i:i + bs], padding=True,
                               return_tensors="pt").to(dev)
                    h = base(**inp).last_hidden_state
                    outs.append(F.normalize(
                        _pool(h, inp["attention_mask"]).float(), dim=-1).cpu())
                    if i % 80 == 0:
                        print(f"P0C0_EMB img {i}/{len(pils)}", flush=True)
            return torch.cat(outs, 0)

        def embed_txts_base(txts):
            tok_emb = base.get_input_embeddings()
            lm = base.language_model if hasattr(base, "language_model") else base.model.language_model
            enc = proc.tokenizer(
                [f"<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
                 f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n" for t in txts],
                return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                o = lm(inputs_embeds=tok_emb(enc["input_ids"]),
                       attention_mask=enc["attention_mask"])
                h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
            return F.normalize(_pool(h, enc["attention_mask"]).float(), dim=-1).cpu()

        routes = {}
        for route, renderer in (("image_lineplot", _render_lineplot),
                                ("image_stft", _render_stft_tiles)):
            pils = [renderer(w, 50) for w in x]
            e = embed_pils(pils)
            routes[route] = _calibrated_zs(
                e, lambda c: _prompts(UCI_CLASSES[c]), embed_txts_base, 6, y)
            print(f"P0C0 {route}: {routes[route]}", flush=True)
        del base, proc
        torch.cuda.empty_cache()

        # audio path floor (released FE2 stack, naive mel-pack)
        emb = _load_embedder(dev)
        mels = torch.stack([_naive_melpack(w) for w in x])
        chunks = []
        for i in range(0, len(mels), 16):
            chunks.append(_imu_through_audio(emb, mels[i:i + 16]).cpu())
        ea = torch.cat(chunks, 0)
        routes["audio_naive_melpack"] = _calibrated_zs(
            ea, lambda c: _prompts(UCI_CLASSES[c]),
            lambda ts: _batched_text(emb, ts), 6, y)
        # random-embedding control (shared for all routes)
        er = F.normalize(torch.randn(len(y), ea.shape[1]), dim=-1)
        routes["random_control"] = _calibrated_zs(
            er, lambda c: _prompts(UCI_CLASSES[c]),
            lambda ts: _batched_text(emb, ts), 6, y)
        result.update({"pool_n": int(len(y)), "chance": round(1 / 6, 4),
                       "routes": routes})

    # ---------------- P0c-A: train the conv front-end ----------------
    else:
        ckpt_path = os.path.join(CKPT_DIR, f"p0c_imu_frontend_{tag}.pt")
        torch.save({"preflight": torch.zeros(4)}, ckpt_path)
        volume.commit()
        assert torch.load(ckpt_path)["preflight"].shape == (4,)
        print(f"P0C_PREFLIGHT ckpt OK {ckpt_path}", flush=True)

        z = np.load(os.path.join(OUT, "sensorcaps.npz"), allow_pickle=True)
        x_tr, cap_tr = z["x_train"], list(z["cap_train"])
        x_ho, cap_ho = z["x_hold"], list(z["cap_hold"])
        if smoke:
            x_tr, cap_tr = x_tr[:400], cap_tr[:400]
            x_ho, cap_ho = x_ho[:80], cap_ho[:80]
        gal_n = min(1500, len(x_ho))
        rng = np.random.RandomState(0)
        gal = rng.permutation(len(x_ho))[:gal_n]
        x_ho, cap_ho = x_ho[gal], [cap_ho[i] for i in gal]
        # per-channel standardization stats from TRAIN only
        mu = x_tr.mean(axis=(0, 2), keepdims=True)
        sd = x_tr.std(axis=(0, 2), keepdims=True) + 1e-6
        x_tr = (x_tr - mu) / sd
        x_ho = (x_ho - mu) / sd
        print(f"P0C_A train {x_tr.shape} holdout(motionsense) {x_ho.shape}", flush=True)

        emb = _load_embedder(dev)

        # frozen text targets (whitened, matching FE2 training/eval geometry)
        txt_cache = os.path.join(OUT, f"text_{'smoke' if smoke else 'full'}.pt")
        if os.path.exists(txt_cache):
            blob = torch.load(txt_cache)
            t_tr, t_ho = blob["train"], blob["holdout"]
            assert t_tr.shape[0] == len(cap_tr) and t_ho.shape[0] == len(cap_ho)
        else:
            print(f"P0C_A text precompute {len(cap_tr)}+{len(cap_ho)}", flush=True)
            t_tr = _batched_text(emb, cap_tr)
            t_ho = _batched_text(emb, cap_ho)
            torch.save({"train": t_tr, "holdout": t_ho}, txt_cache)
            volume.commit()

        class FrontEnd(torch.nn.Module):
            """[B,6,TARGET_T] -> mel-like [B,128,500] for the frozen tower."""
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Conv1d(6, 64, 7, padding=3), torch.nn.GELU(),
                    torch.nn.Conv1d(64, 128, 7, padding=3), torch.nn.GELU(),
                    torch.nn.Conv1d(128, 128, 3, padding=1))
                self.norm = torch.nn.LayerNorm(128)
                self.gain = torch.nn.Parameter(torch.tensor(0.8))

            def forward(self, w):                              # w [B,6,TARGET_T]
                u = F.interpolate(w, size=500, mode="linear", align_corners=False)
                m = self.net(u)                                # [B,128,500]
                m = self.norm(m.transpose(1, 2)).transpose(1, 2)
                return m * self.gain

        def naive_pack_torch(w):
            """[B,6,T] -> [B,128,500] torch mirror of _naive_melpack (the working
            zero-train floor configuration: per-channel STFT bands stacked)."""
            B = w.shape[0]
            u = F.interpolate(w, size=500, mode="linear", align_corners=False)
            bands = []
            per = 128 // 6
            for c in range(6):
                S = torch.stft(u[:, c], n_fft=32, hop_length=8, return_complex=True,
                               window=torch.hann_window(32, device=w.device))
                m = torch.log1p(S.abs())                       # [B,17,T']
                bands.append(F.interpolate(m.unsqueeze(1), size=(per, 500),
                                           mode="bilinear", align_corners=False)[:, 0])
            mel = torch.cat(bands, 1)                          # [B,126,500]
            mel = torch.cat([mel, mel[:, :128 - mel.shape[1]]], 1)
            mu_ = mel.mean(dim=(1, 2), keepdim=True)
            sd_ = mel.std(dim=(1, 2), keepdim=True) + 1e-6
            return (mel - mu_) / sd_ * 0.8

        class FrontEndResidual(torch.nn.Module):
            """fallback arm (evidence-driven): start AT the zero-train floor —
            mel = naive_pack(w) + alpha * net(w). The floor probe showed the
            naive configuration carries real signal (0.37 ZS vs 0.167 chance);
            random-init front-ends collapse the frozen pipeline (arm A result),
            so the fallback learns a residual on the working configuration.
            Deviation from the plan's CNN+GRU fallback, noted in the RESULT."""
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Conv1d(6, 64, 7, padding=3), torch.nn.GELU(),
                    torch.nn.Conv1d(64, 128, 7, padding=3), torch.nn.GELU(),
                    torch.nn.Conv1d(128, 128, 3, padding=1))
                self.alpha = torch.nn.Parameter(torch.tensor(0.05))

            def forward(self, w):
                base_mel = naive_pack_torch(w)
                u = F.interpolate(w, size=500, mode="linear", align_corners=False)
                return base_mel + self.alpha * self.net(u)

        fe = (FrontEndResidual() if fallback else FrontEnd()).to(dev)
        n_par = sum(p.numel() for p in fe.parameters())
        logit_scale = torch.nn.Parameter(torch.tensor(float(np.log(1 / 0.07)), device=dev))
        opt = torch.optim.AdamW(list(fe.parameters()) + [logit_scale],
                                lr=(1e-4 if fallback else 3e-4))
        steps = 20 if smoke else 2000
        bs = 12
        bank_k = 64 if smoke else 512
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        print(f"P0C_A_TRAIN fe_params={n_par/1e3:.0f}K steps={steps} bs={bs} "
              f"bank={bank_k} fallback={fallback}", flush=True)

        xt = torch.tensor(x_tr)
        order = np.random.permutation(len(xt)); ptr = 0
        t_tr0 = time.time()
        for step in range(steps):
            if ptr + bs > len(order):
                order = np.random.permutation(len(xt)); ptr = 0
            idx = order[ptr:ptr + bs]; ptr += bs
            mel = fe(xt[idx].to(dev))
            iemb = _imu_through_audio(emb, mel, grad=True)
            txt = t_tr[idx].to(dev)
            neg_pool = np.setdiff1d(np.arange(len(xt)), idx)
            neg = t_tr[np.random.choice(neg_pool, bank_k, replace=False)].to(dev)
            scale = logit_scale.exp().clamp(max=100.0)
            logits = scale * iemb @ torch.cat([txt, neg], 0).T
            labels = torch.arange(bs, device=dev)
            loss = F.cross_entropy(logits, labels) + \
                F.cross_entropy(logits[:, :bs].T, labels)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % (5 if smoke else 100) == 0:
                sps_ = (step + 1) / max(time.time() - t_tr0, 1e-6)
                fe_gn = sum(pp.grad.norm().item() for pp in fe.parameters()
                            if pp.grad is not None)
                mel_std = mel.std().item()
                print(f"P0C_A_TRAIN step {step}/{steps} loss {loss.item():.4f} "
                      f"fe_gradnorm {fe_gn:.3e} mel_std {mel_std:.3f} "
                      f"{sps_:.2f} step/s eta {(steps - step) / max(sps_, 1e-6) / 60:.0f}m",
                      flush=True)
            if step and step % 500 == 0:
                torch.save({"frontend": fe.state_dict(), "step": step}, ckpt_path)
                volume.commit()

        torch.save({"frontend": fe.state_dict(), "logit_scale": logit_scale.detach().cpu(),
                    "step": steps, "seed": seed, "fallback": fallback,
                    "params": n_par, "mu": mu, "sd": sd}, ckpt_path)
        volume.commit()
        print(f"P0C_CKPT saved {ckpt_path}", flush=True)

        # ---- naive floor retrieval on the SAME holdout (context row) ----
        if fallback:
            with torch.no_grad():
                chunks = []
                for i in range(0, len(x_ho), 16):
                    mel = naive_pack_torch(torch.tensor(x_ho[i:i + 16]).to(dev))
                    chunks.append(_imu_through_audio(emb, mel).cpu())
                inaive = torch.cat(chunks, 0)
            simsn = t_ho @ inaive.T
            ranksn = (simsn >= simsn.gather(1, torch.arange(len(inaive))[:, None])).sum(1).float()
            result["floor_text2imu"] = {
                "MRR": round((1.0 / ranksn).mean().item(), 4),
                "R@10": round((ranksn <= 10).float().mean().item(), 4)}

        # ---- GATE: holdout (MotionSense, cross-dataset) text->IMU retrieval ----
        fe.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(x_ho), 16):
                mel = fe(torch.tensor(x_ho[i:i + 16]).to(dev))
                chunks.append(_imu_through_audio(emb, mel).cpu())
            ih = torch.cat(chunks, 0)                          # [N,D] IMU gallery
        sims = t_ho @ ih.T                                     # text query -> IMU gallery
        diag = sims.gather(1, torch.arange(len(ih))[:, None])
        ranks = (sims >= diag).sum(1).float()
        mrr = (1.0 / ranks).mean().item()
        r_at = {f"R@{k}": round((ranks <= k).float().mean().item(), 4)
                for k in (1, 5, 10)}
        # random-embedding control on the identical pool
        er = F.normalize(torch.randn_like(ih), dim=-1)
        simsr = t_ho @ er.T
        ranksr = (simsr >= simsr.gather(1, torch.arange(len(er))[:, None])).sum(1).float()
        r10_ctrl = (ranksr <= 10).float().mean().item()
        mrr_ctrl = (1.0 / ranksr).mean().item()

        # ---- style-independent ZS: WISDM (never trained, real labels) ----
        zw = np.load(os.path.join(OUT, "wisdm.npz"))
        xw = (zw["x"] - mu) / sd
        yw = torch.tensor(zw["y"])
        if smoke:
            xw, yw = xw[:120], yw[:120]
        with torch.no_grad():
            chunks = []
            for i in range(0, len(xw), 16):
                mel = fe(torch.tensor(xw[i:i + 16], dtype=torch.float32).to(dev))
                chunks.append(_imu_through_audio(emb, mel).cpu())
            ew = torch.cat(chunks, 0)
        wisdm = _calibrated_zs(ew, lambda c: _prompts(WISDM_CLASSES[c]),
                               lambda ts: _batched_text(emb, ts), 6, yw)

        gate_pass = (r_at["R@10"] >= 10 * max(r10_ctrl, 1e-9)) and mrr >= 0.09
        kill = mrr < 0.05
        result.update({
            "frontend_params": n_par, "steps": steps,
            "holdout_pool": int(len(ih)),
            "text2imu": {"MRR": round(mrr, 4), **r_at},
            "random_control": {"MRR": round(mrr_ctrl, 4),
                               "R@10": round(r10_ctrl, 4)},
            "r10_vs_control_x": round(r_at["R@10"] / max(r10_ctrl, 1e-9), 1),
            "wisdm_zs_neverTrained": wisdm,
            "gate_MRR>=0.09_and_R10>=10x": bool(gate_pass),
            "kill_MRR<0.05": bool(kill),
            "ego4d_108_scenario_zs": "BLOCKED — Ego4D license signature pending (user)",
            "ckpt": ckpt_path})

    result["runtime_s"] = round(time.time() - t0, 1)
    out_path = os.path.join(OUT, f"result_{tag}.json")
    json.dump(result, open(out_path, "w", encoding="utf-8"), indent=1)
    volume.commit()
    print("P0C_RESULT:", json.dumps(result), flush=True)
    return result
