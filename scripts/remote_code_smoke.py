"""GPU smoke for the Hub remote-code (AutoModel + trust_remote_code) loading path.

Standalone Modal app (deliberately not part of modal_app.py): loads each released
repo through plain transformers AutoModel, embeds text/audio/image/video, and asserts
outputs are bitwise-equal to the same inputs embedded through the repository's
reference inference.py loader in the same process. Video additionally gets a
manual-replication check: the identical frames and template pushed through the raw
base model directly must be bitwise-equal to embed_video's output. For
fusion-embedding-2 it also repeats the hooks-removed bitwise-equality checks
(text/image/video) and asserts the gate guards raise on non-audio embeds.

Two modes per repo:
  local  — pre-push: the modeling/configuration files baked from the working tree
           plus the repo's published weights; this is what MUST pass before upload.
  hub    — post-push: everything from the public repo, the end-user path.

Run:
    PYTHONUTF8=1 uv run modal run --detach scripts/remote_code_smoke.py::smoke_fe1_local
    PYTHONUTF8=1 uv run modal run --detach scripts/remote_code_smoke.py::smoke_fe2_local
    PYTHONUTF8=1 uv run modal run --detach scripts/remote_code_smoke.py::smoke_fe1
    PYTHONUTF8=1 uv run modal run --detach scripts/remote_code_smoke.py::smoke_fe2
"""

from __future__ import annotations

import modal

app = modal.App("fusion-remote-code-smoke")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.24",
        "transformers>=4.46",
        "accelerate>=0.30",
        "soundfile>=0.12",
        "librosa>=0.10",
        "pillow>=10.0",
        "torchvision==0.21.0",
        "qwen-vl-utils>=0.0.14",
        "av>=12",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("fusion_embedding")
    .add_local_file("release/inference.py", "/root/fe1_inference.py")
    .add_local_file("fe2_release/inference.py", "/root/fe2_inference.py")
    .add_local_dir("release/remote_code", "/root/remote_code")
)

hf_secret = modal.Secret.from_name("huggingface")

FE1 = "EximiusLabs/fusion-embedding-1-2b-preview"
FE2 = "EximiusLabs/fusion-embedding-2-2b-preview"
TEXT = "a dog barks in the distance while rain falls on a metal roof"
N_FRAMES = 16


def _load_ref(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ref_inference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synth_frames():
    """Deterministic 16-frame clip: a bright square moving across a noisy field."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(7)
    base = (rng.random((240, 320, 3)) * 80).astype("uint8")
    frames = []
    for t in range(N_FRAMES):
        arr = base.copy()
        x = 10 + t * 17
        arr[80:160, x:x + 60, 0] = 230
        arr[80:160, x:x + 60, 1] = 120 + 6 * t
        frames.append(Image.fromarray(arr))
    return frames


def _synth_mp4(frames, path="/tmp/smoke_clip.mp4"):
    import os
    import subprocess
    d = "/tmp/smoke_frames"
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(f"{d}/{i:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4",
         "-i", f"{d}/%03d.png", "-pix_fmt", "yuv420p", path], check=True)
    return path


def _manual_base_video(m, frames):
    """The identical frames/template through the raw base model, no wrapper method:
    replicates embed_video's documented preprocessing inline and returns the
    embedding the BASE produces. Must be bitwise-equal to m.embed_video(frames)."""
    import torch
    from qwen_vl_utils import process_vision_info

    conversation = [{"role": "user", "content": [
        {"type": "video", "video": list(frames),
         "total_pixels": 10 * 768 * 32 * 32}]}]
    _, video_inputs, video_kwargs = process_vision_info(
        conversation, image_patch_size=16,
        return_video_metadata=True, return_video_kwargs=True)
    videos, metadata = zip(*video_inputs)
    text = ("<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|><|im_end|>\n"
            "<|im_start|>assistant\n")
    inputs = m._rt["proc"](text=[text], videos=list(videos),
                           video_metadata=list(metadata), do_resize=False,
                           return_tensors="pt", **video_kwargs).to("cuda")
    h = m._rt["full"](**inputs).last_hidden_state
    mask = inputs["attention_mask"]
    lengths = (mask.long().sum(dim=1) - 1).clamp(min=0)
    idx = lengths.view(-1, 1, 1).expand(-1, 1, h.size(-1))
    pooled = h.gather(1, idx).squeeze(1)
    dim = m.config.mrl_default
    return torch.nn.functional.normalize(
        pooled.float()[..., :dim], p=2, dim=-1).squeeze(0).cpu()


def _local_repo_dir(repo: str, gen: int) -> str:
    """Pre-push source of truth: working-tree remote-code files + published weights."""
    import os
    import shutil
    from huggingface_hub import hf_hub_download

    d = f"/tmp/local_fe{gen}"
    os.makedirs(d, exist_ok=True)
    shutil.copy("/root/remote_code/modeling_fusion_embedding.py", d)
    shutil.copy("/root/remote_code/configuration_fusion_embedding.py", d)
    shutil.copy(f"/root/remote_code/out_fe{gen}_config.json", f"{d}/config.json")
    st = hf_hub_download(repo, "model.safetensors")
    shutil.copy(st, f"{d}/model.safetensors")
    return d


def _smoke(repo: str, ref_path: str, check_hooks_removed: bool,
           local_gen: int | None = None) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel

    rng = np.random.default_rng(0)
    wav = (rng.standard_normal(16000 * 4)).astype("float32") * 0.05
    img = Image.fromarray((rng.random((224, 224, 3)) * 255).astype("uint8"))
    frames = _synth_frames()

    # --- AutoModel path (remote code: working tree in local mode, Hub otherwise) ---
    src = _local_repo_dir(repo, local_gen) if local_gen else repo
    m = AutoModel.from_pretrained(src, trust_remote_code=True)
    m = m.to("cuda").eval()
    t_auto = m.embed_text(TEXT)
    a_auto = m.embed_audio(wav, sr=16000)
    i_auto = m.embed_image(img)
    v_auto = m.embed_video(frames)

    out = {"repo": repo, "mode": "local" if local_gen else "hub",
           "dims": {"text": list(t_auto.shape), "audio": list(a_auto.shape),
                    "image": list(i_auto.shape), "video": list(v_auto.shape)},
           "video_norm": round(float(v_auto @ v_auto), 6),
           "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2)}

    # --- video: manual replication through the raw base, same process ---
    v_manual = _manual_base_video(m, frames)
    out["bitwise_video_vs_raw_base"] = bool(torch.equal(v_auto, v_manual))

    # --- video: file-path input exercises the decoder branch end-to-end ---
    v_path = m.embed_video(_synth_mp4(frames))
    out["video_path_input"] = {"dims": list(v_path.shape),
                               "norm": round(float(v_path @ v_path), 6),
                               "cos_vs_frames": round(float(v_path @ v_auto), 4)}

    # --- reference loader (the repo's inference.py), same process/device ---
    ref = _load_ref(ref_path)
    fe = ref.FusionEmbedder.from_pretrained(repo, device="cuda")
    t_ref = fe.embed_text(TEXT)
    a_ref = fe.embed_audio(wav, sr=16000)
    i_ref = fe.embed_image(img)
    v_ref = fe.embed_video(frames)

    out["bitwise_vs_reference"] = {
        "text": bool(torch.equal(t_auto, t_ref)),
        "audio": bool(torch.equal(a_auto, a_ref)),
        "image": bool(torch.equal(i_auto, i_ref)),
        "video": bool(torch.equal(v_auto, v_ref)),
    }

    # batched entry points: B=1 must be bitwise-identical to the single-item path;
    # B=2 (ragged lengths) is reported as max-abs-diff for information.
    tb1 = m.embed_text_batch([TEXT])[0]
    ab1 = m.embed_audio_batch([wav], sr=16000)[0]
    tb2 = m.embed_text_batch([TEXT, "water dripping in a cave"])[0]
    out["bitwise_batch1_vs_single"] = {
        "text": bool(torch.equal(tb1, t_auto)),
        "audio": bool(torch.equal(ab1, a_auto)),
    }
    out["batch2_text_maxdiff"] = float((tb2 - t_auto).abs().max())

    if check_hooks_removed:
        # gate guards must refuse non-audio embeds while the gate is open
        guards = {}
        for name, call in (("video_automodel", lambda: m.embed_video(frames)),
                           ("image_automodel", lambda: m.embed_image(img)),
                           ("video_reference", lambda: fe.embed_video(frames))):
            gate = m._rt["gate"] if "automodel" in name else fe.model._adapter_gate
            try:
                with gate:
                    call()
                guards[name] = False
            except RuntimeError:
                guards[name] = True
        out["gate_guard_raises"] = guards

        t_h, i_h, v_h = m.embed_text(TEXT), m.embed_image(img), m.embed_video(frames)
        for h in m._rt["adapter_handles"]:
            h.remove()
        out["bitwise_hooks_removed"] = {
            "text": bool(torch.equal(t_h, m.embed_text(TEXT))),
            "image": bool(torch.equal(i_h, m.embed_image(img))),
            "video": bool(torch.equal(v_h, m.embed_video(frames))),
        }

    ok = (all(out["bitwise_vs_reference"].values())
          and out["bitwise_video_vs_raw_base"]
          and all(out["bitwise_batch1_vs_single"].values())
          and abs(out["video_norm"] - 1.0) < 1e-4
          and abs(out["video_path_input"]["norm"] - 1.0) < 1e-4)
    if check_hooks_removed:
        ok = ok and all(out["bitwise_hooks_removed"].values()) \
                and all(out["gate_guard_raises"].values())
    out["ok"] = bool(ok and abs(float(t_auto @ t_auto) - 1.0) < 1e-4)
    print("REMOTE_CODE_SMOKE:", out)
    assert out["ok"], "remote-code smoke failed"
    return out


@app.function(gpu="L4", image=image, secrets=[hf_secret], timeout=2400)
def smoke_fe1() -> dict:
    return _smoke(FE1, "/root/fe1_inference.py", check_hooks_removed=False)


@app.function(gpu="L4", image=image, secrets=[hf_secret], timeout=2400)
def smoke_fe2() -> dict:
    return _smoke(FE2, "/root/fe2_inference.py", check_hooks_removed=True)


@app.function(gpu="L4", image=image, secrets=[hf_secret], timeout=2400)
def smoke_fe1_local() -> dict:
    return _smoke(FE1, "/root/fe1_inference.py", check_hooks_removed=False,
                  local_gen=1)


@app.function(gpu="L4", image=image, secrets=[hf_secret], timeout=2400)
def smoke_fe2_local() -> dict:
    return _smoke(FE2, "/root/fe2_inference.py", check_hooks_removed=True,
                  local_gen=2)
