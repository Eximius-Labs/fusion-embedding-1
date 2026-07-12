"""GPU smoke for the Hub remote-code (AutoModel + trust_remote_code) loading path.

Standalone Modal app (deliberately not part of modal_app.py): loads each released
repo through plain transformers AutoModel, embeds text/audio/image, and asserts the
TEXT embedding is bitwise-equal to the same input embedded through the repository's
reference inference.py loader in the same process. For fusion-embedding-2 it also
repeats the hooks-removed bitwise-equality check on text and image.

Run (after the remote-code files are uploaded to the repos):
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
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("fusion_embedding")
    .add_local_file("release/inference.py", "/root/fe1_inference.py")
    .add_local_file("fe2_release/inference.py", "/root/fe2_inference.py")
)

hf_secret = modal.Secret.from_name("huggingface")

FE1 = "EximiusLabs/fusion-embedding-1-2b-preview"
FE2 = "EximiusLabs/fusion-embedding-2-2b-preview"
TEXT = "a dog barks in the distance while rain falls on a metal roof"


def _load_ref(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ref_inference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _smoke(repo: str, ref_path: str, check_hooks_removed: bool) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel

    rng = np.random.default_rng(0)
    wav = (rng.standard_normal(16000 * 4)).astype("float32") * 0.05
    img = Image.fromarray((rng.random((224, 224, 3)) * 255).astype("uint8"))

    # --- AutoModel path (the Hub remote code) ---
    m = AutoModel.from_pretrained(repo, trust_remote_code=True)
    m = m.to("cuda").eval()
    t_auto = m.embed_text(TEXT)
    a_auto = m.embed_audio(wav, sr=16000)
    i_auto = m.embed_image(img)

    out = {"repo": repo,
           "dims": {"text": list(t_auto.shape), "audio": list(a_auto.shape),
                    "image": list(i_auto.shape)},
           "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2)}

    # --- reference loader (the repo's inference.py), same process/device ---
    ref = _load_ref(ref_path)
    fe = ref.FusionEmbedder.from_pretrained(repo, device="cuda")
    t_ref = fe.embed_text(TEXT)
    a_ref = fe.embed_audio(wav, sr=16000)
    i_ref = fe.embed_image(img)

    out["bitwise_vs_reference"] = {
        "text": bool(torch.equal(t_auto, t_ref)),
        "audio": bool(torch.equal(a_auto, a_ref)),
        "image": bool(torch.equal(i_auto, i_ref)),
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
        t_h, i_h = m.embed_text(TEXT), m.embed_image(img)
        for h in m._rt["adapter_handles"]:
            h.remove()
        out["bitwise_hooks_removed"] = {
            "text": bool(torch.equal(t_h, m.embed_text(TEXT))),
            "image": bool(torch.equal(i_h, m.embed_image(img))),
        }

    ok = out["bitwise_vs_reference"]["text"] and all(
        out["bitwise_batch1_vs_single"].values())
    if check_hooks_removed:
        ok = ok and all(out["bitwise_hooks_removed"].values())
    out["ok"] = bool(ok and abs(float(t_auto @ t_auto) - 1.0) < 1e-4)
    print("REMOTE_CODE_SMOKE:", out)
    assert out["ok"], "remote-code smoke failed"
    return out


@app.function(gpu="L4", image=image, secrets=[hf_secret], timeout=1800)
def smoke_fe1() -> dict:
    return _smoke(FE1, "/root/fe1_inference.py", check_hooks_removed=False)


@app.function(gpu="L4", image=image, secrets=[hf_secret], timeout=1800)
def smoke_fe2() -> dict:
    return _smoke(FE2, "/root/fe2_inference.py", check_hooks_removed=True)
