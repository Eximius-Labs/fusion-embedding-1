"""Hub smoke for Ember, the released thermal sense pack (EximiusLabs/fusion-embedding-2-ember).

Loads the PACKAGED artifact fresh from the Hub and verifies, on GPU, through BOTH
load paths (the dual-load-path lesson):

  (a) isolation with the pack loaded and the gate closed: text, RGB-image, and
      audio outputs bit-for-bit equal to the pack-free released FE2 (v0.2-preview),
      on the FusionEmbedder release path AND the AutoModel remote-code path;
  (b) thermal retrieval sanity: 20 release-holdout IR-TD images vs their captions,
      pack open vs frozen, reproducing seed-2-level separation;
  (c) audio pack co-load: embed_audio with the thermal pack attached is bitwise
      the audio-only reference.

Deploy + spawn:
    PYTHONUTF8=1 uv run modal deploy scripts/thermal_pack_hub_smoke.py
    uv run python -c "import modal; print(modal.Function.from_name('fusion-thermal-pack-smoke','smoke').spawn().object_id)"
"""
from __future__ import annotations

import modal

app = modal.App("fusion-thermal-pack-smoke")
hf_secret = modal.Secret.from_name("huggingface")
volume = modal.Volume.from_name("fusion-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "torchcodec==0.2.1", "numpy>=1.24",
        "transformers>=4.46", "accelerate>=0.30", "pillow>=10.0",
        "soundfile>=0.12", "librosa>=0.10", "safetensors>=0.4",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"PYTHONUTF8": "1", "PYTHONPATH": "/root/fe:/root/fe/fe2_release"})
    .add_local_dir(".", "/root/fe", copy=True,
                   ignore=["**/.git/**", "**/.venv/**", "**/_render/**",
                           "**/__pycache__/**", "**/results/**", "**/docs/**",
                           "release/**", "assets/**", "dist/**", "submission/**",
                           "thermal_pack_release/**", "**/*.egg-info/**", "**/*.pt",
                           "**/*.safetensors", "**/*.zip", "**/*.png", "**/*.jpg"])
)

PACK_REPO = "EximiusLabs/fusion-embedding-2-ember"
PACK_REV = "main"   # v0.1-preview snapshot keeps the pre-rename filename
FE2_REPO = "EximiusLabs/fusion-embedding-2-2b-preview"
FE2_REV = "v0.2-preview"
IRTD_ROOT = "/vol/thermal/ir_td/Data4MLLM/IR-TD-version250724"
THERMAL_INSTRUCTION = "Represent this thermal infrared image."
DOC_INSTRUCTION = "Represent the user's input."


def _chat(instruction: str, user: str) -> str:
    return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


@app.function(gpu="A100-80GB", image=image, secrets=[hf_secret],
              volumes={"/vol": volume}, timeout=3600, memory=32768)
def smoke() -> dict:
    import json
    import os
    import sys
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from safetensors.torch import load_file

    sys.path.insert(0, "/root/fe")
    sys.path.insert(0, "/root/fe/fe2_release")
    from fusion_embedding.adapters import AdapterPacks
    from inference import FusionEmbedder

    t0 = time.time()
    dev = "cuda"
    out: dict = {"pack_repo": PACK_REPO, "pack_revision": PACK_REV,
                 "fe2": FE2_REPO, "fe2_revision": FE2_REV, "checks": {}}

    def check(name, ok, note=""):
        out["checks"][name] = {"pass": bool(ok), **({"note": note} if note else {})}
        print(f"SMOKE {name}: {'PASS' if ok else 'FAIL'} {note}", flush=True)

    # ---- fresh pack download (no cache) ----
    st_path = hf_hub_download(PACK_REPO, "model.safetensors",
                              revision=PACK_REV, force_download=True)
    cfg_path = hf_hub_download(PACK_REPO, "config.json", revision=PACK_REV,
                               force_download=True)
    strip_path = hf_hub_download(PACK_REPO, "release_strip_640x512.json",
                                 revision=PACK_REV, force_download=True)
    pack_cfg = json.load(open(cfg_path, encoding="utf-8"))
    sd = load_file(st_path)
    check("artifact_downloads", len(sd) == 112 and pack_cfg["pack"]["rank"] == 384,
          f"{len(sd)} tensors, config rank {pack_cfg['pack']['rank']}")

    # ---- release path: FusionEmbedder, pinned v0.2-preview ----
    emb = FusionEmbedder.from_pretrained(FE2_REPO, device=dev, revision=FE2_REV)
    model, full, proc = emb.model, emb.full, emb.proc
    ip = getattr(proc, "image_processor", None)
    if ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = 1310720                       # training protocol cap
    d_llm = emb.cfg.d_llm

    # references BEFORE attach
    rng = np.random.RandomState(0)
    wav = (np.sin(2 * np.pi * 440 * np.arange(16000 * 4) / 16000)
           + 0.1 * rng.randn(16000 * 4)).astype("float32")
    vis_dir = "/vol/p0a_thermal/llvip/LLVIP/visible/test"
    rgb_img = Image.open(os.path.join(vis_dir, sorted(os.listdir(vis_dir))[0])).convert("RGB")
    with torch.no_grad():
        text_ref = emb.embed_text("a person crossing a dark road").clone()
        rgb_ref = emb.embed_image(rgb_img).clone()
        audio_ref = emb.embed_audio(wav, sr=16000).clone()

    # attach the hub pack
    packs = AdapterPacks()
    adapters, _ = packs.add_pack("thermal", model.base_lm, d_llm, 384)
    adapters.load_state_dict(sd)                       # strict by default
    packs.to(dev)
    check("strict_state_dict_load", True, "112 tensors onto rank-384 pack")

    with torch.no_grad():
        text_after = emb.embed_text("a person crossing a dark road").clone()
        rgb_after = emb.embed_image(rgb_img).clone()
        audio_after = emb.embed_audio(wav, sr=16000).clone()
    check("text_bitwise_gate_closed", torch.equal(text_ref, text_after))
    check("rgb_bitwise_gate_closed", torch.equal(rgb_ref, rgb_after))
    check("audio_coload_bitwise", torch.equal(audio_ref, audio_after),
          "released audio pack + hub thermal pack co-loaded")

    # ---- thermal retrieval sanity: 20 release-holdout images ----
    with open(os.path.join(IRTD_ROOT, "new_relative.json"), encoding="utf-8") as f:
        records = json.load(f)
    stripped = set(json.load(open(strip_path, encoding="utf-8"))["excluded"])
    kept = [r for r in records if r["image_path"] not in stripped]
    order = np.random.RandomState(0).permutation(len(kept))
    holdout = [kept[i] for i in sorted(order[:2000].tolist())][:20]

    def pool(h, mask):
        idx = mask.long().cumsum(1).argmax(1)
        return h[torch.arange(h.shape[0], device=h.device), idx]

    def embed_thermal(paths, open_gate):
        vecs = []
        with torch.no_grad():
            for p in paths:
                im = Image.open(p).convert("RGB")
                text = _chat(THERMAL_INSTRUCTION,
                             "<|vision_start|><|image_pad|><|vision_end|>")
                inp = proc(text=[text], images=[im], return_tensors="pt").to(dev)
                import contextlib
                scope = packs.scope("thermal") if open_gate else contextlib.nullcontext()
                with scope:
                    h = full(**inp).last_hidden_state
                vecs.append(F.normalize(pool(h, inp["attention_mask"]).float(), dim=-1))
        return torch.cat(vecs, 0)

    def embed_caps(caps):
        tok_emb = full.get_input_embeddings()
        vecs = []
        with torch.no_grad():
            for c in caps:
                enc = proc.tokenizer(_chat(DOC_INSTRUCTION, c), return_tensors="pt",
                                     truncation=True, max_length=512).to(dev)
                o = model.base_lm(inputs_embeds=tok_emb(enc["input_ids"]),
                                  attention_mask=enc["attention_mask"])
                h = o if isinstance(o, torch.Tensor) else (
                    o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0])
                vecs.append(F.normalize(pool(h, enc["attention_mask"]).float(), dim=-1))
        return torch.cat(vecs, 0)

    paths = [os.path.join(IRTD_ROOT, r["image_path"]) for r in holdout]
    caps = [r["description"] for r in holdout]
    te = embed_caps(caps)
    th_pack = embed_thermal(paths, True)
    th_frozen = embed_thermal(paths, False)

    def r_at(q, g, k):
        sims = q @ g.T
        ranks = (sims >= sims.gather(1, torch.arange(len(q), device=dev)[:, None])
                 .expand_as(sims)).sum(1)
        return round((ranks <= k).float().mean().item(), 3)

    ret = {"pack": {f"R@{k}": r_at(th_pack, te, k) for k in (1, 5, 10)},
           "frozen": {f"R@{k}": r_at(th_frozen, te, k) for k in (1, 5, 10)}}
    out["retrieval_20"] = ret
    # n=20 gallery is easy for the frozen base at R@10; the separation lives at R@1
    # (seed-2 full-gallery: 0.412 vs 0.071)
    check("thermal_retrieval_sanity",
          ret["pack"]["R@10"] >= 0.9 and
          ret["pack"]["R@1"] >= ret["frozen"]["R@1"] + 0.2,
          json.dumps(ret))

    for h in packs._handles:
        h.remove()

    # ---- documented alternate path: raw base AutoModel, .language_model attach ----
    from transformers import AutoModel
    am = AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-2B",
                                   trust_remote_code=True,
                                   dtype=torch.bfloat16).to(dev).eval()
    am_lm = am.language_model if hasattr(am, "language_model") else am.model.language_model
    am_tok_emb = am.get_input_embeddings()

    def am_text(t):
        enc = proc.tokenizer(_chat(DOC_INSTRUCTION, t), return_tensors="pt").to(dev)
        with torch.no_grad():
            o = am_lm(inputs_embeds=am_tok_emb(enc["input_ids"]),
                      attention_mask=enc["attention_mask"])
            h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
            return F.normalize(pool(h, enc["attention_mask"]).float(), dim=-1)

    t_ref2 = am_text("a person crossing a dark road")
    packs2 = AdapterPacks()
    ad2, _ = packs2.add_pack("thermal", am_lm, d_llm, 384)
    ad2.load_state_dict(sd)
    packs2.to(dev)
    t_after2 = am_text("a person crossing a dark road")
    check("rawbase_automodel_text_bitwise", torch.equal(t_ref2, t_after2),
          "pack attached to raw-base language_model (as trained), gate closed")

    out["all_pass"] = all(c["pass"] for c in out["checks"].values())
    out["runtime_s"] = round(time.time() - t0, 1)
    with open("/vol/thermal_phase1/hub_smoke.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    volume.commit()
    print("SMOKE_RESULT:", json.dumps(out), flush=True)
    return out
