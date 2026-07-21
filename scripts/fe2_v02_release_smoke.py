"""Pre-push smoke for the FE2 v0.2 release artifact (the ARM_1600 AC2 fine-tune).

Loads the PACKAGED weights (staged on the Volume) through the released
inference.py — the end-user path — and verifies, before anything touches HF:
  (a) bitwise preservation on the NEW weights: text, image, and video outputs
      with adapter hooks attached vs removed (torch.equal), i.e. the gate-closed
      exact-preservation claim holds for v0.2;
  (b) AudioCaps-883 a2t through the RELEASE path (raw audio -> embed_audio;
      native-template queries via embed_text) reproduces the trainer-protocol
      0.759 R@10 within protocol noise;
  (c) the artifact loads strictly (adapters r384 present; loader hard-fails on
      mismatch by design).

Deploy + spawn:
    PYTHONUTF8=1 uv run modal deploy scripts/fe2_v02_release_smoke.py
    uv run python -c "import modal; print(modal.Function.from_name('fusion-fe2-v02-smoke','smoke').spawn().object_id)"
"""
from __future__ import annotations

import modal

app = modal.App("fusion-fe2-v02-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchcodec==0.2.1", "numpy>=1.24", "transformers>=4.46", "accelerate>=0.30",
        "soundfile>=0.12", "librosa>=0.10", "pillow>=10.0", "datasets>=2.19",
        "safetensors>=0.4",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_file("fe2_release/inference.py", "/root/fe2_inference.py")
    # the reference inference.py builds the model from the local package
    .add_local_dir("fusion_embedding", "/root/fusion_embedding")
)

hf_secret = modal.Secret.from_name("huggingface")
volume = modal.Volume.from_name("fusion-data")

CKPT = "/vol/ac2/fe2_v02_stage.pt"
SOUND_QUERY_INSTRUCTION = "Retrieve audio by sound description."


@app.function(gpu="L4", image=image, secrets=[hf_secret], volumes={"/vol": volume},
              timeout=4 * 3600, memory=32768,
              env={"HF_HOME": "/vol/hf-cache", "HF_HUB_DOWNLOAD_TIMEOUT": "60"})
def smoke(limit: int = 0) -> dict:
    import importlib.util
    import json
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    spec = importlib.util.spec_from_file_location("fe2_inference", "/root/fe2_inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("SMOKE_SRC_V4 (tensor video input)", flush=True)
    fe = mod.FusionEmbedder.from_pretrained(CKPT, device="cuda")
    assert fe.model.audio_adapters is not None and fe.cfg.adapter_rank == 384, \
        f"adapters missing: rank={fe.cfg.adapter_rank}"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert ck["version"] == "0.2-preview" and ck["source_run"]["run_tag"] == "_ac2ft_1600", ck["source_run"]

    # ---- (b) AudioCaps-883 a2t, release path, min-rank-over-refs ----
    with open("/vol/frames/audiocaps_test816/index.json") as fh:
        idx = json.load(fh)
    caps_multi = idx["captions_multi"]
    clip_ids = idx.get("clip_ids")
    assert clip_ids and len(clip_ids) == len(caps_multi)
    want = {str(c): k for k, c in enumerate(clip_ids)}

    from datasets import Audio, load_dataset
    import io as _io
    import soundfile as sf
    import librosa
    ds = load_dataset("OpenSound/AudioCaps", split="test", streaming=True).cast_column(
        "audio", Audio(decode=False))
    audio_emb = {}
    n_seen = 0
    for row in ds:
        gid = f"{row['youtube_id']}|{row['start_time']}"
        k = want.get(gid)
        if k is None or k in audio_emb:
            continue
        raw = row["audio"]["bytes"]
        try:
            wav, sr0 = sf.read(_io.BytesIO(raw), dtype="float32")
        except Exception:                                        # noqa: BLE001
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        audio_emb[k] = fe.embed_audio(wav, sr=sr0)
        n_seen += 1
        if n_seen % 100 == 0:
            print(f"embedded {n_seen}/{len(want)} clips", flush=True)
        if limit and n_seen >= limit:
            break
    print(f"audio embedded: {len(audio_emb)}/{len(want)}", flush=True)

    kept = sorted(audio_emb)
    texts, owner = [], []
    for j, k in enumerate(kept):
        for c in caps_multi[k]:
            texts.append(c)
            owner.append(j)
    T = torch.stack([torch.as_tensor(
        fe.embed_text(c, instruction=SOUND_QUERY_INSTRUCTION)) for c in texts])
    A = torch.stack([torch.as_tensor(audio_emb[k]) for k in kept])
    S = A @ T.T                                                    # [n_clips, n_caps]
    owner_t = torch.tensor(owner)
    ranks = torch.zeros(len(kept))
    for j in range(len(kept)):
        order = S[j].argsort(descending=True)
        pos = (owner_t[order] == j).nonzero()[0, 0].item() + 1     # min rank over refs
        ranks[j] = pos
    rep = {f"a2t_R@{k}": round((ranks <= k).float().mean().item(), 4) for k in (1, 5, 10)}

    # ---- (a) bitwise preservation LAST (hook removal is one-way in this session) ----
    rng = np.random.default_rng(0)
    from PIL import Image as PILImage
    img = PILImage.fromarray((rng.random((224, 224, 3)) * 255).astype("uint8"))
    frames = torch.from_numpy((rng.random((8, 3, 224, 224)) * 255).astype("uint8"))  # [T,C,H,W] decoded-frames tensor
    t_on = fe.embed_text("a dog barks in the distance")
    i_on = fe.embed_image(img)
    print("video input type:", type(frames).__name__, flush=True)
    v_on = fe.embed_video(frames, fps=2.0)
    for h in fe.model._adapter_handles:
        h.remove()
    t_off = fe.embed_text("a dog barks in the distance")
    i_off = fe.embed_image(img)
    v_off = fe.embed_video(frames, fps=2.0)
    bitwise = {
        "text": bool(torch.equal(torch.as_tensor(t_on), torch.as_tensor(t_off))),
        "image": bool(torch.equal(torch.as_tensor(i_on), torch.as_tensor(i_off))),
        "video": bool(torch.equal(torch.as_tensor(v_on), torch.as_tensor(v_off))),
    }

    out = {"ckpt": CKPT, "version": ck["version"], "bitwise": bitwise,
           "n_clips": len(kept), "n_captions": len(texts), **rep,
           "trainer_protocol_ref": {"a2t_R@10": 0.7588, "a2t_R@1": 0.3103}}
    print("FE2_V02_SMOKE:", json.dumps(out), flush=True)
    assert all(bitwise.values()), f"bitwise preservation FAILED: {bitwise}"
    if not limit:
        assert rep["a2t_R@10"] >= 0.74, f"release-path R@10 {rep['a2t_R@10']} too far from 0.759"
    return out
