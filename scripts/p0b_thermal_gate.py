"""P0b — thermal adapter-gate probe (docs/sensor_extension_plan.md §3, caption-free).

Tests whether a THERMAL-gated deep-adapter pack adds in-pathway capacity for thermal,
using the caption-free objective the user approved: align each thermal image to its
registered RGB visible twin, contrastively, on LLVIP's aligned pairs. No captions, no
captioner, no synthetic data.

Design (mirrors the audio Stage-2 gate probe):
  * Base = Qwen3-VL-Embedding-2B, frozen; the released image path (a
    <|vision_start|><|image_pad|><|vision_end|> doc prompt through the full model,
    last-token pooling, L2-norm) reproduced here so the gate can be opened during
    training. Thermal images are 3x-channel-replicated (as P0a).
  * Thermal pack = one GatedAdapter per decoder layer, rank 384, zero-init, attached
    via AdapterPacks. A thermal encode opens the thermal scope (fwd AND bwd); an RGB /
    text encode leaves it closed -> bitwise-identical to the frozen base.
  * Arm A: thermal adapters r384 trained; Arm B: control (no adapters). Retrieval R@10
    is rank-invariant to the logit temperature, so the control is the frozen-base
    thermal->RGB retrieval computed in one pass, identical across seeds -- the real
    spend is Arm A x 3 seeds.
  * Data: train on LLVIP TRAIN pairs, eval on LLVIP TEST (held out, disjoint -> the
    dedup requirement is satisfied by construction; LLVIP test is also the P0a LLVIP-ZS
    set). RGB-twin embeddings are frozen -> precomputed once and cached.

Eval per arm: thermal->RGB-twin retrieval R@10 (primary gate) + P0a calibrated LLVIP-ZS
(must-not-degrade). Isolation test from run 1: an RGB-image forward with the thermal
gate closed is bit-for-bit the no-adapter base.

Gate (pre-registered): Arm A >= +3 R@10 over Arm B at EVERY seed AND LLVIP-ZS not
degraded -> thermal pack GO for Phase 1; +1..+3 -> rank sweep; <+1 -> STOP.

Run (deploy + spawn; client-independent):
    PYTHONUTF8=1 uv run modal deploy scripts/p0b_thermal_gate.py
    PYTHONUTF8=1 uv run python -c "import modal; modal.Function.from_name('fusion-p0b-thermal','run').spawn(smoke=True)"
    PYTHONUTF8=1 uv run python -c "import modal; modal.Function.from_name('fusion-p0b-thermal','run').spawn()"
Results land on the fusion-data Volume under p0b_thermal/ and print as P0B_* lines.
"""

from __future__ import annotations

import modal

app = modal.App("fusion-p0b-thermal")
hf_secret = modal.Secret.from_name("huggingface")
volume = modal.Volume.from_name("fusion-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "numpy>=1.24",
        "transformers>=4.46", "accelerate>=0.30", "pillow>=10.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
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


def _chat(instruction: str, user: str) -> str:
    return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


@app.function(gpu="A100-80GB", image=image, secrets=[hf_secret],
              volumes={"/vol": volume}, timeout=6 * 3600, memory=32768)
def run(smoke: bool = False) -> dict:
    import gc
    import json
    import os
    import sys
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    sys.path.insert(0, "/root/fe")
    from fusion_embedding.adapters import AdapterPacks

    t0 = time.time()
    out_dir = "/vol/p0b_thermal"
    os.makedirs(out_dir, exist_ok=True)
    data_root = "/vol/p0a_thermal/llvip"          # reuse the P0a LLVIP extraction
    dev = "cuda"

    # ---- locate LLVIP infrared/visible train + test dirs (registered by stem) ----
    def find_split(root, spectrum, split):
        for r, _d, files in os.walk(root):
            n = r.lower().replace("\\", "/")
            if spectrum in n and f"/{split}" in n and any(f.lower().endswith(".jpg") for f in files):
                return r
        return None

    ir_tr = find_split(data_root, "infrared", "train")
    vis_tr = find_split(data_root, "visible", "train")
    ir_te = find_split(data_root, "infrared", "test")
    vis_te = find_split(data_root, "visible", "test")
    assert ir_tr and vis_tr and ir_te and vis_te, \
        f"LLVIP dirs: ir_tr={ir_tr} vis_tr={vis_tr} ir_te={ir_te} vis_te={vis_te}"

    def pairs(ir_dir, vis_dir):
        vis = {f for f in os.listdir(vis_dir) if f.lower().endswith(".jpg")}
        return sorted(f for f in os.listdir(ir_dir)
                      if f.lower().endswith(".jpg") and f in vis)

    train_stems = pairs(ir_tr, vis_tr)
    test_stems = pairs(ir_te, vis_te)
    if smoke:
        train_stems, test_stems = train_stems[:256], test_stems[:128]
    counts = {"train_pairs": len(train_stems), "test_pairs": len(test_stems),
              "ir_train_dir": ir_tr, "vis_train_dir": vis_tr,
              "ir_test_dir": ir_te, "vis_test_dir": vis_te}
    print("P0B_DATA:", json.dumps(counts))
    assert len(train_stems) >= (100 if smoke else 8000), "too few LLVIP train pairs"

    # ---- base model (frozen) + processor, released image path reproduced ----
    from transformers import AutoModel, AutoProcessor
    dtype = torch.bfloat16
    base = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True, dtype=dtype).to(dev).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    lm = base.language_model if hasattr(base, "language_model") else base.model.language_model
    d_llm = lm.config.hidden_size if hasattr(lm, "config") else base.config.text_config.hidden_size
    # gradient checkpointing: batch of full-base forwards must fit memory. The thermal
    # scope is held across fwd AND bwd (below), so recompute sees the gate open — the
    # hazard the unit tests lock in.
    for m in (lm, base):
        if hasattr(m, "gradient_checkpointing_enable"):
            try:
                m.gradient_checkpointing_enable()
            except Exception:
                pass

    def load_img(d, stem, thermal):
        im = Image.open(os.path.join(d, stem)).convert("RGB")   # 3x replicate for thermal
        return im

    def _proc(pil, instruction):
        # single image (kept for the isolation test); no_grad-safe
        text = _chat(instruction, "<|vision_start|><|image_pad|><|vision_end|>")
        return proc(text=[text], images=[pil], return_tensors="pt").to(dev)

    def _proc_batch(pils, instruction):
        text = _chat(instruction, "<|vision_start|><|image_pad|><|vision_end|>")
        return proc(text=[text] * len(pils), images=list(pils), return_tensors="pt").to(dev)

    def _pool(h, mask):
        # last-token (EOS) pooling. Padding-side robust: the index of the LAST 1 in the
        # mask is argmax of the running cumsum (argmax returns first-occurrence of the
        # max), correct for both right- and left-padded batches.
        idx = mask.long().cumsum(1).argmax(1)
        return h[torch.arange(h.shape[0], device=h.device), idx]

    def embed_batch(dirs_stems, instruction, packs=None, gate_name=None, grad=False,
                    chunk=None):
        """Embed a list of (dir, stem) through the full frozen base; if packs+gate_name
        given, the pack fires (thermal path). Returns [N,d] unit-norm (float32).

        Images are processed in BATCHED chunks (one processor + base() call per chunk)
        for throughput. For a grad step pass the whole batch as one chunk (default) so
        the InfoNCE forward is a single graph; for precompute pass a chunk size."""
        thermal = instruction == THERMAL_INSTRUCTION
        cz = chunk or len(dirs_stems)                      # images loaded lazily per chunk
        embs = []
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for k, i in enumerate(range(0, len(dirs_stems), cz)):
                pils = [load_img(d, stem, thermal) for d, stem in dirs_stems[i:i + cz]]
                inp = _proc_batch(pils, instruction)
                scope = packs.scope(gate_name) if (packs and gate_name) else _null()
                with scope:
                    h = base(**inp).last_hidden_state
                pooled = _pool(h, inp["attention_mask"]).float()
                vecs = F.normalize(pooled, dim=-1)
                if grad:
                    embs.append(vecs)                      # keep on GPU (autograd)
                else:
                    embs.append(vecs.detach().cpu())       # host-accumulate (bounded)
                    del pils, inp, h, pooled, vecs
                    if k % 25 == 0:
                        torch.cuda.empty_cache()
        out = torch.cat(embs, 0)
        return out if grad else out.to(dev)

    import contextlib
    def _null():
        return contextlib.nullcontext()

    # ---- precompute frozen RGB-twin embeddings (train + test), no adapters ----
    def rgb_items(stems, d):
        return [(d, s) for s in stems]

    cache_dir = os.path.join(out_dir, "rgb_cache")
    os.makedirs(cache_dir, exist_ok=True)

    def precompute_cached(stems, vis_dir, tag):
        """Frozen RGB-twin embeddings, computed on the FULL frozen base and CACHED to the
        Volume incrementally so a crash RESUMES instead of restarting. A prior zombie died
        silently mid-precompute (SIGKILL, no traceback) over 15K one-image reads, so this
        path is memory-disciplined (small chunk, CPU accumulation, periodic empty_cache +
        gc) and heartbeats with flush so log-silence is never mistaken for progress."""
        smoke_tag = "smoke_" if smoke else ""
        path = os.path.join(cache_dir, f"rgb_{smoke_tag}{tag}.pt")
        want = list(stems)
        done, embs_by_stem = {}, {}
        if os.path.exists(path):
            blob = torch.load(path, map_location="cpu")
            embs_by_stem = {s: e for s, e in zip(blob["stems"], blob["embs"])}
            done = {s for s in want if s in embs_by_stem}
            print(f"P0B_PRECACHE {tag}: resume, {len(done)}/{len(want)} cached",
                  flush=True)
        todo = [s for s in want if s not in done]
        PC = 4 if smoke else 8                              # small chunk -> bounded peak
        t_pc = time.time()
        for i in range(0, len(todo), PC):
            batch_stems = todo[i:i + PC]
            try:
                with torch.no_grad():
                    pils = [load_img(vis_dir, s, False) for s in batch_stems]
                    inp = _proc_batch(pils, DOC_INSTRUCTION)
                    h = base(**inp).last_hidden_state
                    pooled = _pool(h, inp["attention_mask"]).float()
                    vecs = F.normalize(pooled, dim=-1).cpu()   # accumulate on HOST (small)
                for s, v in zip(batch_stems, vecs):
                    embs_by_stem[s] = v
                del pils, inp, h, pooled, vecs
            except Exception as e:                          # a bad image becomes VISIBLE
                print(f"P0B_PRECACHE {tag}: FAIL at {batch_stems}: "
                      f"{type(e).__name__} {e}", flush=True)
                raise
            n_done = len(done) + i + len(batch_stems)
            if (i // PC) % 25 == 0:                          # heartbeat + reclaim
                torch.cuda.empty_cache(); gc.collect()
                rate = (i + len(batch_stems)) / max(time.time() - t_pc, 1e-6)
                print(f"P0B_PRECACHE {tag}: {n_done}/{len(want)} "
                      f"({rate:.1f} img/s)", flush=True)
            if (i // PC) % 200 == 0 and i > 0:               # checkpoint to Volume
                _save_cache(path, want, embs_by_stem)
        _save_cache(path, want, embs_by_stem)
        print(f"P0B_PRECACHE {tag}: DONE {len(want)} in "
              f"{time.time() - t_pc:.0f}s", flush=True)
        return torch.stack([embs_by_stem[s] for s in want], 0)   # [N,D] on CPU

    def _save_cache(path, stems, embs_by_stem):
        keep = [s for s in stems if s in embs_by_stem]
        torch.save({"stems": keep, "embs": [embs_by_stem[s] for s in keep]}, path)
        volume.commit()

    print("P0B: precomputing frozen RGB embeddings (cached, resumable) ...", flush=True)
    rgb_tr = precompute_cached(train_stems, vis_tr, "train").detach()
    rgb_te = precompute_cached(test_stems, vis_te, "test").detach().to(dev)

    def retrieval_r10(therm_emb, rgb_emb):
        """thermal->RGB-twin R@10 (matched diagonal targets). Rank-invariant to temp."""
        sims = therm_emb @ rgb_emb.T                       # [N,N]
        ranks = (sims >= sims.gather(1, torch.arange(sims.shape[0], device=sims.device)[:, None]
                                     ).expand_as(sims)).sum(1)   # 1 = top
        return ((ranks <= 10).float().mean().item(),
                (ranks <= 5).float().mean().item(),
                (ranks <= 1).float().mean().item())

    # ---- LLVIP-ZS harness (reuse P0a ensemble+calibrated) ----
    sys.path.insert(0, "/root/fe/scripts")
    from p0a_thermal_probe import ENSEMBLE  # noqa

    def llvip_zs():
        # thermal person/background crops via the P0a VOC protocol, ensembled+calibrated
        from p0a_thermal_probe import _build_crop_sets, _find_layout
        ann_dir, ir_test, _vis, _ = _find_layout(data_root)
        n = 60 if smoke else 500
        persons, bgs, _ = _build_crop_sets(ann_dir, ir_test, n, seed=0)
        pe = embed_batch([("__mem__", p) for p in []], DOC_INSTRUCTION) if False else None
        # embed crops directly (they are PIL already)
        def emb_crops(crops):
            out = []
            with torch.no_grad():
                for im in crops:
                    inp = _proc(im.convert("RGB"), THERMAL_INSTRUCTION)
                    h = base(**inp).last_hidden_state
                    out.append(F.normalize(_pool(h, inp["attention_mask"]).float(), dim=-1))
            return torch.cat(out, 0)

        def ens(prompts):
            # Text goes through the LANGUAGE-MODEL path (embed_tokens -> lm -> pool),
            # not the full vision model (a text-only forward through the vision stack
            # errors). No adapters fire (gate closed), so this is the frozen text path.
            txts = [_chat(DOC_INSTRUCTION, p) for p in prompts]
            tok_ids = base.get_input_embeddings()
            te = []
            with torch.no_grad():
                for t in txts:
                    enc = proc.tokenizer(t, return_tensors="pt").to(dev)
                    emb = tok_ids(enc["input_ids"])
                    out = lm(inputs_embeds=emb, attention_mask=enc["attention_mask"])
                    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                    te.append(F.normalize(_pool(h, enc["attention_mask"]).float(), dim=-1))
            e = torch.stack(te).mean(0)          # te elems are [1, D] -> stack [K,1,D] -> [1,D]
            return e / e.norm(dim=1, keepdim=True)

        pe_, be_ = emb_crops(persons), emb_crops(bgs)
        cls_te = torch.cat([ens(ENSEMBLE["person"]), ens(ENSEMBLE["background"])], 0)
        sp = pe_ @ cls_te.T
        sn = be_ @ cls_te.T
        bias = torch.cat([sp, sn], 0).mean(0, keepdim=True)
        pc = ((sp - bias).argmax(1) == 0).float().mean().item()
        bc = ((sn - bias).argmax(1) == 1).float().mean().item()
        return round((pc * len(persons) + bc * len(bgs)) / (len(persons) + len(bgs)) * 100, 2)

    # ---- Arm B (control): frozen thermal -> RGB retrieval, temp-invariant ----
    print("P0B: Arm B control (frozen thermal->RGB) ...")
    therm_te_frozen = embed_batch(rgb_items(test_stems, ir_te), THERMAL_INSTRUCTION,
                                  chunk=12).detach()
    b_r10, b_r5, b_r1 = retrieval_r10(therm_te_frozen, rgb_te)
    control = {"R@10": round(b_r10, 4), "R@5": round(b_r5, 4), "R@1": round(b_r1, 4),
               "LLVIP_ZS": llvip_zs()}
    print("P0B_CONTROL:", json.dumps(control))

    # ---- isolation test (run once): an RGB-image forward is bit-for-bit the frozen
    # base once a (nonzero) thermal pack is attached, as long as the gate stays closed.
    d0, s0 = vis_te, test_stems[0]
    with torch.no_grad():                                   # (a) TRUE base, no pack
        inp = _proc(load_img(d0, s0, False), DOC_INSTRUCTION)
        ref_out = base(**inp).last_hidden_state.clone()
    packs_iso = AdapterPacks()                              # (b) attach + randomize up
    packs_iso.add_pack("thermal", lm, d_llm, RANK)
    packs_iso.to(dev)
    with torch.no_grad():
        for lin in packs_iso.parameters_of("thermal"):
            if lin.dim() == 2 and lin.shape[0] == d_llm and torch.count_nonzero(lin) == 0:
                lin.copy_(torch.randn_like(lin) * 0.02)     # nonzero 'up' -> real no-op test
        inp = _proc(load_img(d0, s0, False), DOC_INSTRUCTION)  # (c) gate CLOSED
        closed_out = base(**inp).last_hidden_state.clone()
    for h in packs_iso._handles:                            # (e) detach
        h.remove()
    isolation_bitwise = bool(torch.equal(ref_out, closed_out))
    print("P0B_ISOLATION bitwise_rgb_gate_closed:", isolation_bitwise)

    # ---- Arm A: train thermal adapters, 3 seeds ----
    steps = 40 if smoke else 800
    bs = 6 if smoke else 16
    seeds = [1] if smoke else [1, 2, 3]
    arm_a = {}
    train_items = rgb_items(train_stems, ir_tr)           # thermal train items
    rgb_tr_by_stem = {s: rgb_tr[i] for i, s in enumerate(train_stems)}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        packs = AdapterPacks()
        adapters, gate = packs.add_pack("thermal", lm, d_llm, RANK)
        packs.to(dev)
        logit_scale = torch.nn.Parameter(torch.tensor(float(np.log(1 / 0.07)), device=dev))
        params = list(adapters.parameters()) + [logit_scale]
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        order = np.random.permutation(len(train_items))
        ptr = 0
        for step in range(steps):
            if ptr + bs > len(order):
                order = np.random.permutation(len(train_items)); ptr = 0
            idx = order[ptr:ptr + bs]; ptr += bs
            items = [train_items[i] for i in idx]
            stems = [train_stems[i] for i in idx]
            rgb = torch.stack([rgb_tr_by_stem[s] for s in stems]).to(dev)   # frozen targets
            # thermal forward WITH gate open across fwd+bwd
            with packs.scope("thermal"):
                th = embed_batch(items, THERMAL_INSTRUCTION, packs, "thermal", grad=True)
                scale = logit_scale.exp().clamp(max=100.0)
                logits = scale * th @ rgb.T
                labels = torch.arange(len(items), device=dev)
                loss = 0.5 * (F.cross_entropy(logits, labels)
                              + F.cross_entropy(logits.T, labels))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % (10 if smoke else 100) == 0:
                gn = sum(p.grad.norm().item() for p in adapters.parameters() if p.grad is not None)
                print(f"P0B seed{seed} step{step} loss {loss.item():.4f} adapter_gradnorm {gn:.3f}")

        # eval this seed
        with torch.no_grad():
            th_eval = embed_batch(rgb_items(test_stems, ir_te), THERMAL_INSTRUCTION,
                                  packs, "thermal", chunk=12).detach()
        a_r10, a_r5, a_r1 = retrieval_r10(th_eval, rgb_te)
        zs = llvip_zs()
        arm_a[f"seed{seed}"] = {
            "R@10": round(a_r10, 4), "R@5": round(a_r5, 4), "R@1": round(a_r1, 4),
            "LLVIP_ZS": zs,
            "delta_R@10": round(a_r10 - b_r10, 4),
            "zs_delta": round(zs - control["LLVIP_ZS"], 2),
        }
        print(f"P0B_ARMA seed{seed}:", json.dumps(arm_a[f"seed{seed}"]))
        for h in packs._handles:
            h.remove()

    deltas = [arm_a[k]["delta_R@10"] for k in arm_a]
    zs_deltas = [arm_a[k]["zs_delta"] for k in arm_a]
    min_delta = min(deltas)
    zs_ok = min(zs_deltas) >= -2.0                          # "not degraded"
    if min_delta >= 0.03 and zs_ok:
        verdict = "GO — thermal pack adds capacity at every seed; proceed to Phase 1"
    elif min_delta >= 0.01:
        verdict = "MARGINAL — +1..+3 band; rank sweep (128/384/768) is the next step"
    else:
        verdict = "STOP — thermal adapters do not add capacity; analyze token statistics"

    result = {
        "probe": "P0b thermal adapter-gate (caption-free, thermal->RGB-twin, LLVIP)",
        "base": BASE_MODEL, "rank": RANK, "steps": steps, "batch": bs, "seeds": seeds,
        "smoke": smoke, "data": counts,
        "isolation_bitwise_rgb_gate_closed": isolation_bitwise,
        "control_ArmB": control, "ArmA": arm_a,
        "paired_deltas_R@10": deltas, "min_delta_R@10": round(min_delta, 4),
        "zs_deltas": zs_deltas, "llvip_zs_not_degraded": zs_ok,
        "gate_verdict": verdict, "runtime_s": round(time.time() - t0, 1),
    }
    tag = "smoke" if smoke else "full"
    with open(f"{out_dir}/p0b_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    volume.commit()
    print("P0B_RESULT:", json.dumps(result))
    return result
