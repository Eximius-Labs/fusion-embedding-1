"""Composability matrix — audio pack + thermal pack co-loaded, exact isolation verified.

The experiment behind Paper 2's composability claim (docs/research_composability_prior_art.md
section 6): with the released FE2 AUDIO adapter pack (legacy single-gate attach, exactly as
shipped) and the P0b THERMAL adapter pack (AdapterPacks registry) co-loaded on the same frozen
decoder layers, verify the exclusivity invariant bitwise:

  (a) audio inputs   -> bitwise-identical to the audio-pack-only model (FE2 as released);
  (b) thermal inputs -> bitwise-identical to the thermal-pack-only model;
  (c) text / RGB-image / video inputs -> bitwise-identical to the raw frozen base;
  (d) eval numbers reproduce EXACTLY under co-load: the AudioCaps-816 frames protocol
      (audio-only vs co-loaded) and thermal->RGB-twin retrieval (thermal-only vs co-loaded).

Also closes the registry-vs-legacy risk first: the released audio adapter weights driven
through the AdapterPacks code path must produce bitwise-identical decoder outputs vs the
shipped single-gate path.

P0b did not persist its per-seed packs, so this run retrains the seed-1 thermal pack with the
identical recipe (LLVIP thermal->RGB-twin, 800 steps, rank 384, cached RGB targets reused from
the volume) and SAVES it to /vol/p0b_thermal/thermal_pack_seed1.pt.

Mixed inputs that would open two gates in one forward are outside the guarantee and are not
tested (see the scoping caveat in the prior-art doc).

Run (deploy + spawn; client-independent):
    PYTHONUTF8=1 uv run modal deploy scripts/composability_matrix.py
    PYTHONUTF8=1 uv run python -c "import modal; modal.Function.from_name('fusion-composability','run').spawn(smoke=True)"
    PYTHONUTF8=1 uv run python -c "import modal; modal.Function.from_name('fusion-composability','run').spawn()"
"""

from __future__ import annotations

import modal

app = modal.App("fusion-composability")
hf_secret = modal.Secret.from_name("huggingface")
volume = modal.Volume.from_name("fusion-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "numpy>=1.24",
        "transformers>=4.46", "accelerate>=0.30", "pillow>=10.0",
        "librosa>=0.10", "soundfile>=0.12", "huggingface_hub",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"HF_HOME": "/vol/hf-cache", "PYTHONPATH": "/root/fe:/root/fe/fe2_release"})
    .add_local_dir(".", "/root/fe", copy=True,
                   ignore=["**/.git/**", "**/.venv/**", "**/_render/**",
                           "**/__pycache__/**", "**/results/**", "**/docs/**",
                           "release/**", "assets/**", "dist/**", "submission/**",
                           "**/*.egg-info/**", "**/*.pt", "**/*.safetensors",
                           "**/*.zip", "**/*.png", "**/*.jpg", "data/**"])
)

FE2_REPO = "EximiusLabs/fusion-embedding-2-2b-preview"
FE2_REV = "2d30494e4ef581ccad66e9d90b55dc4a84b96bc4"
THERMAL_INSTRUCTION = "Represent this thermal infrared image."
DOC_INSTRUCTION = "Represent the user's input."
RANK, STEPS, BS, SEED = 384, 800, 16, 1


def _chat(instruction: str, user: str) -> str:
    return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


@app.function(gpu="A100-80GB", image=image, secrets=[hf_secret],
              volumes={"/vol": volume}, timeout=5 * 3600, memory=32768)
def run(smoke: bool = False) -> dict:
    import contextlib
    import json
    import os
    import sys
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    sys.path.insert(0, "/root/fe")
    sys.path.insert(0, "/root/fe/fe2_release")
    sys.path.insert(0, "/root/fe/scripts")

    from fusion_embedding.adapters import AdapterPacks
    from inference import FusionEmbedder

    t0 = time.time()
    dev = "cuda"
    out_dir = "/vol/p0b_thermal"
    matrix: dict = {"smoke": smoke, "fe2_repo": FE2_REPO, "fe2_revision": FE2_REV,
                    "thermal_recipe": {"rank": RANK, "steps": STEPS, "batch": BS,
                                       "seed": SEED, "objective": "thermal->RGB-twin, LLVIP"},
                    "cells": {}, "eval_reproduction": {}}

    def cell(name, ok, note=""):
        matrix["cells"][name] = {"bitwise": bool(ok), **({"note": note} if note else {})}
        print(f"MATRIX {name}: {'BITWISE-EQUAL' if ok else 'MISMATCH'} {note}", flush=True)

    # ---------------- host: released FE2, pinned revision ----------------
    print("COMPOSE: loading FE2 (pinned) ...", flush=True)
    emb = FusionEmbedder.from_pretrained(FE2_REPO, device=dev, revision=FE2_REV)
    model, full, proc = emb.model, emb.full, emb.proc
    assert model.audio_adapters is not None and model._adapter_gate is not None
    d_llm = model.cfg.d_llm if hasattr(model, "cfg") else emb.cfg.d_llm

    # ---------------- A1: registry-vs-legacy on the released audio weights ----------
    g = torch.Generator().manual_seed(0)
    x_syn = torch.randn(2, 12, d_llm, generator=g).to(dev, torch.bfloat16)
    m_syn = torch.ones(2, 12, dtype=torch.long, device=dev)
    with torch.no_grad(), model.adapter_scope():
        y_legacy = model.base_lm(inputs_embeds=x_syn, attention_mask=m_syn).clone()
    packs_a = AdapterPacks()
    reg_audio, _ = packs_a.add_pack("audio", model.base_lm, d_llm, RANK)
    packs_a.to(dev)
    reg_audio.load_state_dict(model.audio_adapters.state_dict())
    with torch.no_grad(), packs_a.scope("audio"):        # legacy gate closed here
        y_registry = model.base_lm(inputs_embeds=x_syn, attention_mask=m_syn).clone()
    for h in packs_a._handles:
        h.remove()
    cell("audio_registry_vs_legacy", torch.equal(y_legacy, y_registry),
         "released audio weights, AdapterPacks path vs shipped single-gate path")

    # ---------------- audio-only references (config: FE2 as released) ----------
    rng = np.random.RandomState(0)
    wavs = [np.sin(2 * np.pi * f * np.arange(16000 * 4) / 16000).astype("float32")
            + 0.1 * rng.randn(16000 * 4).astype("float32") for f in (261.6, 440.0, 880.0)]
    audio_ref = [emb.embed_audio(w, sr=16000) for w in wavs]

    score_kw = dict(device=dev, dim=0, task="sound")
    score_audio_only = None
    if not smoke:
        from modal_app import _HFTok, _score_816_protocol
        from fusion_embedding.data import FrameCollator
        collator = FrameCollator(emb.cfg, _HFTok(emb.tok, emb.cfg))
        print("COMPOSE: SCORE816 audio-only ...", flush=True)
        score_audio_only = _score_816_protocol(model, emb.cfg, collator,
                                               "audiocaps_test816", **score_kw)
        print("SCORE816_AUDIO_ONLY:", json.dumps(score_audio_only), flush=True)

    # ---------------- thermal data + cached RGB targets (from P0b) ----------
    data_root = "/vol/p0a_thermal/llvip"

    def find_split(spectrum, split):
        for r, _d, files in os.walk(data_root):
            n = r.lower().replace("\\", "/")
            if spectrum in n and f"/{split}" in n and any(f.endswith(".jpg") for f in files):
                return r
        raise FileNotFoundError(f"{spectrum}/{split}")

    ir_tr, vis_tr = find_split("infrared", "train"), find_split("visible", "train")
    ir_te, vis_te = find_split("infrared", "test"), find_split("visible", "test")

    def load_cache(tag):
        blob = torch.load(os.path.join(out_dir, "rgb_cache", f"rgb_{tag}.pt"),
                          map_location="cpu")
        return blob["stems"], torch.stack(list(blob["embs"]), 0)

    tr_stems, rgb_tr = load_cache("smoke_train" if smoke else "train")
    te_stems, rgb_te = load_cache("smoke_test" if smoke else "test")
    rgb_te = rgb_te.to(dev)
    if smoke:
        te_stems, rgb_te = te_stems[:96], rgb_te[:96]
    print(f"COMPOSE: cached RGB targets train={len(tr_stems)} test={len(te_stems)}", flush=True)

    def pool(h, mask):
        idx = mask.long().cumsum(1).argmax(1)
        return h[torch.arange(h.shape[0], device=h.device), idx]

    def embed_thermal(stems, ir_dir, scope, grad=False, chunk=12):
        embs, ctx = [], (torch.enable_grad() if grad else torch.no_grad())
        cz = chunk if not grad else len(stems)
        with ctx:
            for i in range(0, len(stems), cz):
                pils = [Image.open(os.path.join(ir_dir, s)).convert("RGB")
                        for s in stems[i:i + cz]]
                text = _chat(THERMAL_INSTRUCTION,
                             "<|vision_start|><|image_pad|><|vision_end|>")
                inp = proc(text=[text] * len(pils), images=pils,
                           return_tensors="pt").to(dev)
                with scope():
                    h = full(**inp).last_hidden_state
                v = F.normalize(pool(h, inp["attention_mask"]).float(), dim=-1)
                embs.append(v if grad else v.detach().cpu())
                if not grad and (i // cz) % 25 == 0:
                    torch.cuda.empty_cache()
        out = torch.cat(embs, 0)
        return out if grad else out.to(dev)

    def r_at(therm, rgb):
        sims = therm @ rgb.T
        ranks = (sims >= sims.gather(
            1, torch.arange(sims.shape[0], device=sims.device)[:, None]).expand_as(sims)).sum(1)
        return {f"R@{k}": round((ranks <= k).float().mean().item(), 4) for k in (1, 5, 10)}

    # ---------------- A3: attach + train the thermal pack (P0b recipe, seed 1) -------
    steps, bs = (8, 6) if smoke else (STEPS, BS)
    for m in (model.base_lm, full):
        if hasattr(m, "gradient_checkpointing_enable"):
            with contextlib.suppress(Exception):
                m.gradient_checkpointing_enable()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    packs = AdapterPacks()
    thermal_ad, _ = packs.add_pack("thermal", model.base_lm, d_llm, RANK)
    packs.to(dev)
    logit_scale = torch.nn.Parameter(torch.tensor(float(np.log(1 / 0.07)), device=dev))
    opt = torch.optim.AdamW(list(thermal_ad.parameters()) + [logit_scale],
                            lr=1e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rgb_by_stem = {s: rgb_tr[i] for i, s in enumerate(tr_stems)}
    order, ptr = np.random.permutation(len(tr_stems)), 0
    print(f"COMPOSE: training thermal pack ({steps} steps, audio pack co-loaded+closed) ...",
          flush=True)
    for step in range(steps):
        if ptr + bs > len(order):
            order, ptr = np.random.permutation(len(tr_stems)), 0
        stems = [tr_stems[i] for i in order[ptr:ptr + bs]]; ptr += bs
        rgb = torch.stack([rgb_by_stem[s] for s in stems]).to(dev)
        with packs.scope("thermal"):
            th = embed_thermal(stems, ir_tr, contextlib.nullcontext, grad=True)
            scale = logit_scale.exp().clamp(max=100.0)
            logits = scale * th @ rgb.T
            labels = torch.arange(len(stems), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, labels)
                          + F.cross_entropy(logits.T, labels))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % (2 if smoke else 100) == 0:
            print(f"COMPOSE train step{step} loss {loss.item():.4f}", flush=True)

    torch.save({"thermal_adapters": thermal_ad.state_dict(),
                "recipe": matrix["thermal_recipe"]},
               os.path.join(out_dir, "thermal_pack_seed1.pt"))
    volume.commit()
    print("COMPOSE: thermal pack saved to /vol/p0b_thermal/thermal_pack_seed1.pt", flush=True)

    # ---------------- A4: co-loaded outputs (audio hooks + thermal pack) ----------
    audio_co = [emb.embed_audio(w, sr=16000) for w in wavs]
    cell("audio_coloaded_vs_audio_only",
         all(torch.equal(a, b) for a, b in zip(audio_co, audio_ref)),
         "3 wavs through tower+resampler+decoder, legacy gate; thermal attached")

    score_coloaded = None
    if not smoke:
        print("COMPOSE: SCORE816 co-loaded ...", flush=True)
        score_coloaded = _score_816_protocol(model, emb.cfg, collator,
                                             "audiocaps_test816", **score_kw)
        print("SCORE816_COLOADED:", json.dumps(score_coloaded), flush=True)
        same = {k: score_coloaded.get(k) == score_audio_only.get(k)
                for k in score_audio_only if isinstance(score_audio_only[k], (int, float))}
        matrix["eval_reproduction"]["audiocaps816"] = {
            "audio_only": score_audio_only, "coloaded": score_coloaded,
            "identical": all(same.values()), "per_metric": same}
        print("EVAL_REPRO audio816 identical:", all(same.values()), flush=True)

    texts = ["a dog barks twice", "rain on a tin roof",
             "an empty street at night", "orchestral strings swell",
             "two people argue in a kitchen", "a train passes a crossing"]
    text_co = [emb.embed_text(t) for t in texts]
    rgb_imgs = [Image.open(os.path.join(vis_te, s)).convert("RGB") for s in te_stems[:6]]
    img_co = [emb.embed_image(im) for im in rgb_imgs]
    gv = torch.Generator().manual_seed(7)
    vids = [torch.randint(0, 255, (6, 3, 128, 160), generator=gv, dtype=torch.uint8)
            for _ in range(2)]
    vid_co = [emb.embed_video(v) for v in vids]

    therm_co = embed_thermal(te_stems, ir_te, lambda: packs.scope("thermal"))
    r_co = r_at(therm_co, rgb_te)
    print("THERMAL coloaded retrieval:", json.dumps(r_co), flush=True)

    # ---------------- A5: thermal-only (remove the audio hooks) ----------
    for h in model._adapter_handles:
        h.remove()
    therm_solo = embed_thermal(te_stems, ir_te, lambda: packs.scope("thermal"))
    r_solo = r_at(therm_solo, rgb_te)
    cell("thermal_coloaded_vs_thermal_only", torch.equal(therm_co, therm_solo),
         f"{len(te_stems)} LLVIP test thermal images")
    matrix["eval_reproduction"]["thermal_rgb_twin"] = {
        "thermal_only": r_solo, "coloaded": r_co, "identical": r_co == r_solo,
        "p0b_seed1_reference_R@10": 0.3898}

    # ---------------- A6: raw base (remove the thermal hooks too) ----------
    for h in packs._handles:
        h.remove()
    text_base = [emb.embed_text(t) for t in texts]
    img_base = [emb.embed_image(im) for im in rgb_imgs]
    vid_base = [emb.embed_video(v) for v in vids]
    cell("text_coloaded_vs_base",
         all(torch.equal(a, b) for a, b in zip(text_co, text_base)), f"{len(texts)} texts")
    cell("image_coloaded_vs_base",
         all(torch.equal(a, b) for a, b in zip(img_co, img_base)), "6 LLVIP visible images")
    cell("video_coloaded_vs_base",
         all(torch.equal(a, b) for a, b in zip(vid_co, vid_base)), "2 synthetic frame tensors")

    matrix["all_bitwise"] = all(c["bitwise"] for c in matrix["cells"].values())
    matrix["eval_identical"] = all(v["identical"]
                                   for v in matrix["eval_reproduction"].values()) \
        if matrix["eval_reproduction"] else None
    matrix["runtime_s"] = round(time.time() - t0, 1)
    tag = "smoke" if smoke else "full"
    with open(os.path.join(out_dir, f"composability_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=1)
    volume.commit()
    print("COMPOSABILITY_RESULT:", json.dumps(matrix), flush=True)
    assert matrix["all_bitwise"], "bitwise cell failed — see matrix"
    if matrix["eval_identical"] is not None:
        assert matrix["eval_identical"], "eval reproduction differed — see matrix"
    return matrix
