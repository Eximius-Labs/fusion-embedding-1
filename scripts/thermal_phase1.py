"""Thermal Phase 1 — shippable thermal pack, thermal->TEXT (docs/sensor_extension_plan.md §4).

User-approved Phase 1 on the P0b GO (+22 R@10 at every seed). Trains the thermal-gated
adapter pack (rank 384, zero-init, AdapterPacks multi-gate registry) on IR-TD's 84K real
thermal images + descriptive captions, contrastively thermal->text, frozen base.

Two caption-style arms (the v0.4 style-region lesson):
  * Arm "full":  train on the full IR-TD captions (mean ~131 words).
  * Arm "short": same data, captions truncated to their FIRST SENTENCE.

Each arm is ONE self-contained Modal job (no phase handoffs): data prep -> phash dedup
vs LLVIP test -> frozen text precompute (cached, resumable) -> train -> eval gates ->
durable ckpt save + volume.commit. Ckpt save+commit is PRE-FLIGHT verified before any
training spend.

Pre-registered gates (report verbatim):
  1. IR-TD-holdout thermal->text R@10: winner >= frozen-baseline +3 points.
  2. LLVIP-ZS (P0a calibrated ensemble harness, thermal crops encoded WITH the pack)
     >= 87.0 (not degraded >2.5 from 89.5).
  3. RGB-image + text bitwise isolation with thermal gate closed: TRUE (real weights;
     video shares the vision path + decoder hooks, covered by the same mechanism +
     unit suite).
  4. Non-gating: LLVIP thermal->RGB-twin R@10 vs P0b's 0.39 (generalization), audio-pack
     coexistence spot-check (co-load, bitwise with gates closed).

Run (deploy + spawn; client-independent):
    PYTHONUTF8=1 uv run modal deploy scripts/thermal_phase1.py
    ... spawn(arm="full", smoke=True)   # smoke first (standing rule)
    ... spawn(arm="full") / spawn(arm="short")
Ckpts land at /vol/checkpoints/thermal_phase1_<arm>.pt; results print as TP1_* lines.
"""

from __future__ import annotations

import modal

app = modal.App("fusion-thermal-phase1")
hf_secret = modal.Secret.from_name("huggingface")
volume = modal.Volume.from_name("fusion-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "numpy>=1.24",
        "transformers>=4.46", "accelerate>=0.30", "pillow>=10.0",
        "imagehash>=4.3",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"PYTHONUTF8": "1"})
    .add_local_dir(".", "/root/fe", copy=True,
                   ignore=["**/.git/**", "**/.venv/**", "**/_render/**",
                           "**/__pycache__/**", "**/results/**", "**/docs/**",
                           "release/**", "fe2_release/**", "assets/**", "dist/**",
                           "submission/**", "**/*.egg-info/**", "**/*.pt",
                           "**/*.safetensors", "**/*.zip", "**/*.png", "**/*.jpg"])
)

BASE_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DOC_INSTRUCTION = "Represent the user's input."
THERMAL_INSTRUCTION = "Represent this thermal infrared image."
RANK = 384
IRTD_ROOT = "/vol/thermal/ir_td/Data4MLLM/IR-TD-version250724"
LLVIP_ROOT = "/vol/p0a_thermal/llvip"
P0B_RGB_CACHE = "/vol/p0b_thermal/rgb_cache/rgb_test.pt"
OUT_ROOT = "/vol/thermal_phase1"
CKPT_DIR = "/vol/checkpoints"
HOLDOUT_N = 2000
PHASH_MAX_DIST = 4          # <= this hamming distance vs any LLVIP TEST phash = dedup hit


def _chat(instruction: str, user: str) -> str:
    return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


def _first_sentence(desc: str) -> str:
    import re
    parts = re.split(r"(?<=[.!?])\s+", desc.strip(), maxsplit=1)
    s = parts[0].strip()
    if not s:                                # caption without terminal punctuation
        s = " ".join(desc.strip().split()[:40])
    return s


@app.function(gpu="A100-80GB", image=image, secrets=[hf_secret],
              volumes={"/vol": volume}, timeout=10 * 3600, memory=32768)
def run(arm: str = "full", smoke: bool = False, seed: int = 1) -> dict:
    import gc
    import json
    import os
    import sys
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    assert arm in ("full", "short"), f"unknown arm {arm!r}"
    sys.path.insert(0, "/root/fe")
    sys.path.insert(0, "/root/fe/scripts")
    from fusion_embedding.adapters import AdapterPacks

    t0 = time.time()
    dev = "cuda"
    tag = f"{arm}{'_smoke' if smoke else ''}"
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"thermal_phase1_{arm}{'_smoke' if smoke else ''}.pt")

    # ---- PRE-FLIGHT: durable ckpt save + commit + readback BEFORE any spend ----
    torch.save({"preflight": torch.zeros(4)}, ckpt_path)
    volume.commit()
    assert torch.load(ckpt_path)["preflight"].shape == (4,), "ckpt preflight readback failed"
    print(f"TP1_PREFLIGHT ckpt save+commit+readback OK at {ckpt_path}", flush=True)

    # ---- data: IR-TD captions ----
    cap_path = os.path.join(IRTD_ROOT, "new_relative.json")
    with open(cap_path, encoding="utf-8") as f:
        records = json.load(f)
    assert isinstance(records, list) and "image_path" in records[0] and \
        "description" in records[0], f"unexpected caption schema: {list(records[0])}"
    print(f"TP1_DATA raw records: {len(records)}; sample: "
          f"{json.dumps({k: str(records[0][k])[:120] for k in records[0]})}", flush=True)

    def img_abspath(rel):
        return os.path.join(IRTD_ROOT, rel)

    # drop records whose image file is missing (report count)
    kept = [r for r in records if os.path.exists(img_abspath(r["image_path"]))]
    n_missing = len(records) - len(kept)
    print(f"TP1_DATA missing image files dropped: {n_missing}", flush=True)

    # deterministic holdout (disjoint by image)
    rng = np.random.RandomState(0)
    order = rng.permutation(len(kept))
    holdout_idx = set(order[:HOLDOUT_N].tolist())
    holdout = [kept[i] for i in sorted(holdout_idx)]
    train = [kept[i] for i in range(len(kept)) if i not in holdout_idx]
    if smoke:
        train, holdout = train[:400], holdout[:60]

    # ---- phash dedup: IR-TD TRAIN vs LLVIP TEST (protect the LLVIP-ZS gate) ----
    import imagehash

    def _llvip_test_dir():
        for r, _d, files in os.walk(LLVIP_ROOT):
            n = r.lower().replace("\\", "/")
            if "infrared" in n and "/test" in n and any(f.endswith(".jpg") for f in files):
                return r
        raise RuntimeError("LLVIP infrared test dir not found")

    phash_cache = os.path.join(OUT_ROOT, "phash_llvip_test.json")
    if os.path.exists(phash_cache):
        llvip_hashes = [imagehash.hex_to_hash(h) for h in json.load(open(phash_cache, encoding="utf-8"))]
        print(f"TP1_DEDUP llvip test hashes cached: {len(llvip_hashes)}", flush=True)
    else:
        d = _llvip_test_dir()
        files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
        if smoke:
            files = files[:80]
        llvip_hashes = []
        for i, fn in enumerate(files):
            llvip_hashes.append(imagehash.phash(Image.open(os.path.join(d, fn))))
            if i % 500 == 0:
                print(f"TP1_DEDUP llvip phash {i}/{len(files)}", flush=True)
        if not smoke:
            json.dump([str(h) for h in llvip_hashes], open(phash_cache, "w", encoding="utf-8"))
            volume.commit()

    llvip_arr = np.stack([h.hash.flatten() for h in llvip_hashes])  # [M,64] bool

    def _min_dist(h):
        return int(np.bitwise_xor(llvip_arr, h.hash.flatten()).sum(1).min())

    irtd_hash_cache = os.path.join(OUT_ROOT, "phash_irtd.json")
    irtd_hashes = json.load(open(irtd_hash_cache, encoding="utf-8")) if os.path.exists(irtd_hash_cache) else {}
    dedup_hits, clean_train, t_dd = [], [], time.time()
    for i, r in enumerate(train):
        rel = r["image_path"]
        if rel in irtd_hashes:
            h = imagehash.hex_to_hash(irtd_hashes[rel])
        else:
            try:
                h = imagehash.phash(Image.open(img_abspath(rel)))
            except Exception as e:
                print(f"TP1_DEDUP unreadable {rel}: {type(e).__name__}", flush=True)
                continue
            irtd_hashes[rel] = str(h)
        (dedup_hits if _min_dist(h) <= PHASH_MAX_DIST else clean_train).append(r)
        if i % 2000 == 0:
            rate = (i + 1) / max(time.time() - t_dd, 1e-6)
            print(f"TP1_DEDUP irtd {i}/{len(train)} ({rate:.0f} img/s, "
                  f"hits {len(dedup_hits)})", flush=True)
            if i % 10000 == 0 and i > 0 and not smoke:
                json.dump(irtd_hashes, open(irtd_hash_cache, "w", encoding="utf-8")); volume.commit()
    if not smoke:
        json.dump(irtd_hashes, open(irtd_hash_cache, "w", encoding="utf-8")); volume.commit()
    print(f"TP1_DEDUP done: {len(dedup_hits)} hits removed, {len(clean_train)} train left "
          f"({time.time() - t_dd:.0f}s)", flush=True)
    train = clean_train

    def cap_of(r):
        return r["description"] if arm == "full" else _first_sentence(r["description"])

    # ---- base model (frozen) ----
    from transformers import AutoModel, AutoProcessor
    base = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True,
                                     dtype=torch.bfloat16).to(dev).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    lm = base.language_model if hasattr(base, "language_model") else base.model.language_model
    d_llm = lm.config.hidden_size if hasattr(lm, "config") else base.config.text_config.hidden_size
    for m in (lm, base):
        if hasattr(m, "gradient_checkpointing_enable"):
            try:
                m.gradient_checkpointing_enable()
            except Exception:
                pass

    def _pool(h, mask):
        idx = mask.long().cumsum(1).argmax(1)     # last real token; padding-side robust
        return h[torch.arange(h.shape[0], device=h.device), idx]

    # ---- frozen TEXT embeddings (LM path; the full-model forward rejects text-only) ----
    def embed_texts(caps, bs=32):
        tok_emb = base.get_input_embeddings()
        out = []
        with torch.no_grad():
            for i in range(0, len(caps), bs):
                txts = [_chat(DOC_INSTRUCTION, c) for c in caps[i:i + bs]]
                enc = proc.tokenizer(txts, return_tensors="pt", padding=True,
                                     truncation=True, max_length=512).to(dev)
                emb = tok_emb(enc["input_ids"])
                o = lm(inputs_embeds=emb, attention_mask=enc["attention_mask"])
                h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
                out.append(F.normalize(_pool(h, enc["attention_mask"]).float(), dim=-1).cpu())
        return torch.cat(out, 0)

    text_cache = os.path.join(OUT_ROOT, f"text_{tag}.pt")
    if os.path.exists(text_cache):
        blob = torch.load(text_cache, map_location="cpu")
        text_tr = blob["embs"]
        assert blob["n"] == len(train), "text cache size mismatch vs train after dedup"
        print(f"TP1_TEXT cached: {tuple(text_tr.shape)}", flush=True)
    else:
        print(f"TP1_TEXT precomputing {len(train)} {arm} captions ...", flush=True)
        t_tx = time.time()
        chunks, CH = [], 2048
        for i in range(0, len(train), CH):
            chunks.append(embed_texts([cap_of(r) for r in train[i:i + CH]]))
            print(f"TP1_TEXT {min(i + CH, len(train))}/{len(train)} "
                  f"({(time.time() - t_tx):.0f}s)", flush=True)
            torch.cuda.empty_cache()
        text_tr = torch.cat(chunks, 0)
        torch.save({"embs": text_tr, "n": len(train)}, text_cache)
        volume.commit()
        print(f"TP1_TEXT done {tuple(text_tr.shape)} in {time.time() - t_tx:.0f}s", flush=True)

    # ---- thermal image embedding (vision path; gate open when packs given) ----
    import contextlib

    def _proc_batch(pils, instruction):
        text = _chat(instruction, "<|vision_start|><|image_pad|><|vision_end|>")
        return proc(text=[text] * len(pils), images=list(pils), return_tensors="pt").to(dev)

    def embed_images(paths, packs=None, gate="thermal", grad=False, chunk=None, hb=None):
        cz = chunk or len(paths)
        embs, t_e = [], time.time()
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for k, i in enumerate(range(0, len(paths), cz)):
                pils = [Image.open(p).convert("RGB") for p in paths[i:i + cz]]
                inp = _proc_batch(pils, THERMAL_INSTRUCTION)
                scope = packs.scope(gate) if packs else contextlib.nullcontext()
                with scope:
                    h = base(**inp).last_hidden_state
                vecs = F.normalize(_pool(h, inp["attention_mask"]).float(), dim=-1)
                if grad:
                    embs.append(vecs)
                else:
                    embs.append(vecs.detach().cpu())
                    del pils, inp, h, vecs
                    if k % 25 == 0:
                        torch.cuda.empty_cache()
                        if hb:
                            print(f"TP1_EMB {hb} {i + cz}/{len(paths)} "
                                  f"({(i + cz) / max(time.time() - t_e, 1e-6):.1f}/s)",
                                  flush=True)
        out = torch.cat(embs, 0)
        return out if grad else out.to(dev)

    def retrieval_r(q, g):
        sims = q @ g.T
        ranks = (sims >= sims.gather(1, torch.arange(sims.shape[0], device=sims.device)
                                     [:, None]).expand_as(sims)).sum(1)
        return {f"R@{k}": round((ranks <= k).float().mean().item(), 4) for k in (1, 5, 10)}

    # ---- frozen holdout baseline (gate 1 reference) ----
    ho_paths = [img_abspath(r["image_path"]) for r in holdout]
    ho_text = embed_texts([r["description"] for r in holdout]).to(dev)       # eval on FULL
    ho_text_short = embed_texts([_first_sentence(r["description"]) for r in holdout]).to(dev)
    print("TP1: frozen holdout baseline ...", flush=True)
    ho_frozen = embed_images(ho_paths, chunk=8, hb="ho_frozen")
    base_full = retrieval_r(ho_frozen, ho_text)
    base_short = retrieval_r(ho_frozen, ho_text_short)
    print(f"TP1_BASELINE frozen holdout t2t: full {json.dumps(base_full)} "
          f"short {json.dumps(base_short)}", flush=True)

    # ---- train the thermal pack ----
    steps = 25 if smoke else 5000
    bs = 8 if smoke else 16
    bank_k = 128 if smoke else 1024
    torch.manual_seed(seed)
    np.random.seed(seed)
    packs = AdapterPacks()
    adapters, _gate = packs.add_pack("thermal", lm, d_llm, RANK)
    packs.to(dev)
    n_params = sum(p.numel() for p in packs.parameters_of("thermal"))
    print(f"TP1_TRAIN arm={arm} steps={steps} bs={bs} bank={bank_k} seed={seed} "
          f"adapter_params={n_params/1e6:.1f}M", flush=True)

    logit_scale = torch.nn.Parameter(torch.tensor(float(np.log(1 / 0.07)), device=dev))
    opt = torch.optim.AdamW(list(adapters.parameters()) + [logit_scale],
                            lr=1e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    tr_paths = [img_abspath(r["image_path"]) for r in train]

    order = np.random.permutation(len(train))
    ptr, t_tr = 0, time.time()
    for step in range(steps):
        if ptr + bs > len(order):
            order = np.random.permutation(len(train)); ptr = 0
        idx = order[ptr:ptr + bs]; ptr += bs
        txt = text_tr[idx].to(dev)                                  # [B,D] frozen
        neg_pool = np.setdiff1d(np.arange(len(train)), idx)
        neg = text_tr[np.random.choice(neg_pool, bank_k, replace=False)].to(dev)
        with packs.scope("thermal"):
            th = embed_images([tr_paths[i] for i in idx], packs, grad=True)
            scale = logit_scale.exp().clamp(max=100.0)
            logits = scale * th @ torch.cat([txt, neg], 0).T        # [B, B+K]
            labels = torch.arange(bs, device=dev)
            loss = F.cross_entropy(logits, labels) \
                + F.cross_entropy(logits[:, :bs].T, labels)         # t2i on in-batch square
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % (5 if smoke else 100) == 0:
            gn = sum(p.grad.norm().item() for p in adapters.parameters()
                     if p.grad is not None)
            sps = (step + 1) / max(time.time() - t_tr, 1e-6)
            print(f"TP1_TRAIN step {step}/{steps} loss {loss.item():.4f} "
                  f"gn {gn:.2f} {sps:.2f} step/s eta {((steps - step) / max(sps, 1e-6)) / 60:.0f}m",
                  flush=True)
        if step % 1000 == 0 and step > 0:
            torch.save({"thermal_adapters": adapters.state_dict(),
                        "step": step, "arm": arm, "seed": seed, "rank": RANK},
                       ckpt_path)
            volume.commit()
            print(f"TP1_CKPT interim saved at step {step}", flush=True)

    # ---- durable final ckpt ----
    torch.save({"thermal_adapters": adapters.state_dict(),
                "logit_scale": logit_scale.detach().cpu(),
                "step": steps, "arm": arm, "seed": seed, "rank": RANK,
                "base": BASE_MODEL, "instruction": THERMAL_INSTRUCTION,
                "train_n": len(train), "dedup_hits": len(dedup_hits)},
               ckpt_path)
    volume.commit()
    print(f"TP1_CKPT final saved: {ckpt_path}", flush=True)

    # ---- GATE 1: holdout thermal->text with the pack ----
    ho_adapted = embed_images(ho_paths, packs, chunk=8, hb="ho_adapted")
    adapted_full = retrieval_r(ho_adapted, ho_text)
    adapted_short = retrieval_r(ho_adapted, ho_text_short)
    print(f"TP1_GATE1 adapted holdout: full {json.dumps(adapted_full)} "
          f"short {json.dumps(adapted_short)}", flush=True)

    # ---- GATE 2: LLVIP-ZS calibrated, thermal crops encoded WITH the pack ----
    from p0a_thermal_probe import ENSEMBLE, _build_crop_sets, _find_layout
    ann_dir, ir_test, _v, _ = _find_layout(LLVIP_ROOT)
    n_crops = 40 if smoke else 500
    persons, bgs, _ = _build_crop_sets(ann_dir, ir_test, n_crops, seed=0)

    def emb_crops(crops, use_pack):
        out = []
        with torch.no_grad():
            for im in crops:
                inp = _proc_batch([im.convert("RGB")], THERMAL_INSTRUCTION)
                scope = packs.scope("thermal") if use_pack else contextlib.nullcontext()
                with scope:
                    h = base(**inp).last_hidden_state
                out.append(F.normalize(_pool(h, inp["attention_mask"]).float(), dim=-1))
        return torch.cat(out, 0)

    def ens(prompts):
        e = embed_texts(prompts).mean(0, keepdim=True).to(dev)
        return e / e.norm(dim=1, keepdim=True)

    def llvip_zs(use_pack):
        pe = emb_crops(persons, use_pack); be = emb_crops(bgs, use_pack)
        cls = torch.cat([ens(ENSEMBLE["person"]), ens(ENSEMBLE["background"])], 0)
        sp, sn = pe @ cls.T, be @ cls.T
        bias = torch.cat([sp, sn], 0).mean(0, keepdim=True)
        pc = ((sp - bias).argmax(1) == 0).float().mean().item()
        bc = ((sn - bias).argmax(1) == 1).float().mean().item()
        return round((pc * len(persons) + bc * len(bgs)) / (len(persons) + len(bgs)) * 100, 2)

    zs_pack = llvip_zs(True)
    zs_frozen = llvip_zs(False)
    print(f"TP1_GATE2 LLVIP_ZS with pack {zs_pack} (frozen ref {zs_frozen})", flush=True)

    # ---- GATE 3: isolation, real trained weights (pack attached + gate closed must be
    # bitwise identical to the hook-free base on RGB-image AND text forwards) ----
    def _txt_once():
        with torch.no_grad():
            return embed_texts(["a night road with two pedestrians"])

    rgb_probe = os.path.join(_find_layout(LLVIP_ROOT)[2],
                             sorted(os.listdir(_find_layout(LLVIP_ROOT)[2]))[0])
    with torch.no_grad():
        inp = _proc_batch([Image.open(rgb_probe).convert("RGB")], DOC_INSTRUCTION)
        out_pack = base(**inp).last_hidden_state.clone()       # pack attached, gate CLOSED
    txt_pack = _txt_once()
    for h in packs._handles:                                   # strip ALL hooks -> true base
        h.remove()
    with torch.no_grad():
        out_base = base(**_proc_batch([Image.open(rgb_probe).convert("RGB")],
                                      DOC_INSTRUCTION)).last_hidden_state.clone()
    txt_base = _txt_once()
    iso_rgb = bool(torch.equal(out_pack, out_base))
    iso_text = bool(torch.equal(txt_pack, txt_base))
    print(f"TP1_GATE3 isolation rgb_bitwise {iso_rgb} text_bitwise {iso_text}", flush=True)
    # re-attach the trained pack for the remaining evals
    packs2 = AdapterPacks()
    ad2, _ = packs2.add_pack("thermal", lm, d_llm, RANK)
    ad2.load_state_dict(adapters.state_dict())
    packs2.to(dev)
    packs = packs2
    txt_a = _txt_once()                                        # coexistence reference

    # ---- non-gating: LLVIP thermal->RGB-twin generalization (vs P0b 0.39) ----
    twin = {}
    try:
        blob = torch.load(P0B_RGB_CACHE, map_location="cpu")
        stems = blob["stems"] if not smoke else blob["stems"][:128]
        rgb_te = torch.stack([e for s, e in zip(blob["stems"], blob["embs"])
                              if s in set(stems)], 0).to(dev)
        ir_dir = ir_test
        th_te = embed_images([os.path.join(ir_dir, s) for s in stems], packs,
                             chunk=8, hb="llvip_twin")
        twin = retrieval_r(th_te, rgb_te)
        print(f"TP1_TWIN llvip thermal->rgb {json.dumps(twin)} (P0b frozen 0.165, "
              f"P0b trained 0.39)", flush=True)
    except Exception as e:
        twin = {"error": f"{type(e).__name__}: {e}"}
        print(f"TP1_TWIN skipped: {twin['error']}", flush=True)

    # ---- non-gating: audio-pack coexistence spot-check ----
    coexist = {}
    try:
        audio_ck = None
        for fn in sorted(os.listdir(CKPT_DIR)):
            if "adprobe_r384" in fn and fn.endswith(".pt"):
                audio_ck = os.path.join(CKPT_DIR, fn)
                break
        packs.add_pack("audio", lm, d_llm, RANK)
        loaded = "random-init"
        if audio_ck:
            sd = torch.load(audio_ck, map_location="cpu", weights_only=False)
            for key in ("audio_adapters", "adapters", "state_dict"):
                if isinstance(sd, dict) and key in sd:
                    sd = sd[key]
                    break
            try:
                packs.audio_adapters.load_state_dict(sd)
                loaded = os.path.basename(audio_ck)
            except Exception:
                loaded = f"random-init (key mismatch in {os.path.basename(audio_ck)})"
        packs.to(dev)
        txt_b = _txt_once()                                    # both packs, gates closed
        co_text = bool(torch.equal(txt_a, txt_b))
        th_probe = [img_abspath(holdout[0]["image_path"])]
        th_only = embed_images(th_probe, packs)                # thermal open, audio closed
        coexist = {"audio_weights": loaded, "text_bitwise_both_packs": co_text,
                   "thermal_encode_ok_with_audio_loaded": bool(th_only.isfinite().all())}
        print(f"TP1_COEXIST {json.dumps(coexist)}", flush=True)
    except Exception as e:
        coexist = {"error": f"{type(e).__name__}: {e}"}
        print(f"TP1_COEXIST failed: {coexist['error']}", flush=True)

    # ---- verdicts ----
    d_full = adapted_full["R@10"] - base_full["R@10"]
    gate1 = d_full >= 0.03
    gate2 = zs_pack >= 87.0
    gate3 = iso_rgb and iso_text
    result = {
        "phase": "thermal_phase1", "arm": arm, "smoke": smoke, "seed": seed,
        "base": BASE_MODEL, "rank": RANK, "steps": steps, "batch": bs, "bank": bank_k,
        "data": {"irtd_records": len(records), "missing_files": n_missing,
                 "train_after_dedup": len(train), "holdout": len(holdout),
                 "dedup_hits_vs_llvip_test": len(dedup_hits),
                 "phash_max_dist": PHASH_MAX_DIST},
        "baseline_frozen_holdout": {"full": base_full, "short": base_short},
        "adapted_holdout": {"full": adapted_full, "short": adapted_short},
        "gate1_delta_R10_full": round(d_full, 4), "gate1_pass": bool(gate1),
        "llvip_zs_with_pack": zs_pack, "llvip_zs_frozen_ref": zs_frozen,
        "gate2_pass": bool(gate2),
        "isolation": {"rgb_bitwise": iso_rgb, "text_bitwise": iso_text},
        "gate3_pass": bool(gate3),
        "llvip_twin_generalization": twin,
        "audio_coexistence": coexist,
        "ckpt": ckpt_path,
        "provenance": {"corpus": "IR-TD (84K real thermal + LLM descriptive captions)",
                       "license_posture": "IR-TD academic/research release; treat as "
                                          "non-commercial until release-hygiene review",
                       "llvip_use": "eval only (ZS gate + twin generalization)"},
        "runtime_s": round(time.time() - t0, 1),
    }
    out_json = os.path.join(OUT_ROOT, f"result_{tag}.json")
    json.dump(result, open(out_json, "w", encoding="utf-8"), indent=2)
    volume.commit()
    print("TP1_RESULT:", json.dumps(result), flush=True)
    return result
