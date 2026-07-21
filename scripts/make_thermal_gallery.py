"""Text->thermal retrieval gallery data for the pack card (fig 2).

Runs the released seed-2 thermal pack over the FULL release holdout (2K gallery),
picks diverse queries by caption keyword buckets, computes each query's top-5
thermal images and the true image's rank, and exports 320px thumbnails plus a
metadata JSON to /vol/thermal_phase1/gallery/. The figure itself is composed
locally by assets/make_thermal_gallery_figure.py from these artifacts.

Deploy + spawn:
    PYTHONUTF8=1 uv run modal deploy scripts/make_thermal_gallery.py
    uv run python -c "import modal; print(modal.Function.from_name('fusion-thermal-gallery','build').spawn().object_id)"
"""
from __future__ import annotations

import modal

app = modal.App("fusion-thermal-gallery")
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
    .add_local_dir("fusion_embedding", "/root/fe/fusion_embedding")
)

IRTD_ROOT = "/vol/thermal/ir_td/Data4MLLM/IR-TD-version250724"
STRIP = "/vol/thermal_phase1/release_strip_640x512.json"
CKPT = "/vol/checkpoints/thermal_release_seed2.pt"
OUT = "/vol/thermal_phase1/gallery"
THERMAL_INSTRUCTION = "Represent this thermal infrared image."
DOC_INSTRUCTION = "Represent the user's input."

BUCKETS = [
    ("night pedestrians", ["pedestrian", "person walking", "people walking"]),
    ("aerial view", ["aerial", "bird's-eye", "top-down view", "drone"]),
    ("industrial / buildings", ["industrial", "factory", "building"]),
    ("cyclists", ["bicycle", "cyclist", "motorcycl"]),
    ("road traffic", ["traffic", "vehicles", "cars on"]),
    ("winter / snow", ["snow", "winter"]),
]


def _chat(instruction: str, user: str) -> str:
    return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


@app.function(gpu="A100-80GB", image=image, secrets=[hf_secret],
              volumes={"/vol": volume}, timeout=3600, memory=32768)
def build() -> dict:
    import json
    import os
    import re
    import sys
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    sys.path.insert(0, "/root/fe")
    from fusion_embedding.adapters import AdapterPacks

    t0 = time.time()
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)

    # release holdout, exactly as the release runs built it
    with open(os.path.join(IRTD_ROOT, "new_relative.json"), encoding="utf-8") as f:
        records = json.load(f)
    stripped = set(json.load(open(STRIP, encoding="utf-8"))["excluded"])
    kept = [r for r in records if r["image_path"] not in stripped]
    order = np.random.RandomState(0).permutation(len(kept))
    holdout = [kept[i] for i in sorted(order[:2000].tolist())]

    from transformers import AutoModel, AutoProcessor
    base = AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-2B",
                                     trust_remote_code=True,
                                     dtype=torch.bfloat16).to(dev).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-Embedding-2B",
                                         trust_remote_code=True)
    ip = getattr(proc, "image_processor", None)
    if ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = 1310720
    lm = base.language_model if hasattr(base, "language_model") else base.model.language_model

    packs = AdapterPacks()
    adapters, _ = packs.add_pack("thermal", lm, 2048, 384)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    adapters.load_state_dict(ck["thermal_adapters"])
    packs.to(dev)

    def pool(h, mask):
        idx = mask.long().cumsum(1).argmax(1)
        return h[torch.arange(h.shape[0], device=h.device), idx]

    # thermal embeddings for the full 2K gallery (gate open), chunked + cached
    paths = [os.path.join(IRTD_ROOT, r["image_path"]) for r in holdout]
    th_cache = os.path.join(OUT, "th_embs.pt")
    if os.path.exists(th_cache):
        th = torch.load(th_cache, map_location="cpu").to(dev)
        print(f"GALLERY: thermal embeds cached {tuple(th.shape)}", flush=True)
    else:
        print("GALLERY: embedding 2000 thermal images ...", flush=True)
        embs, CH = [], 8
        with torch.no_grad():
            for i in range(0, len(paths), CH):
                pils = [Image.open(p).convert("RGB") for p in paths[i:i + CH]]
                text = _chat(THERMAL_INSTRUCTION,
                             "<|vision_start|><|image_pad|><|vision_end|>")
                inp = proc(text=[text] * len(pils), images=pils, padding=True,
                           return_tensors="pt").to(dev)
                with packs.scope("thermal"):
                    h = base(**inp).last_hidden_state
                embs.append(F.normalize(pool(h, inp["attention_mask"]).float(),
                                        dim=-1).cpu())
                if i % 400 == 0:
                    print(f"GALLERY thermal {i}/2000", flush=True)
                    torch.cuda.empty_cache()
        th = torch.cat(embs, 0)
        torch.save(th, th_cache)
        volume.commit()
        th = th.to(dev)

    # query captions (full text, as retrieval used them)
    tok_emb = base.get_input_embeddings()

    def embed_text(c):
        enc = proc.tokenizer(_chat(DOC_INSTRUCTION, c), return_tensors="pt",
                             truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            o = lm(inputs_embeds=tok_emb(enc["input_ids"]),
                   attention_mask=enc["attention_mask"])
            h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
            return F.normalize(pool(h, enc["attention_mask"]).float(), dim=-1)

    def first_sentence(desc):
        parts = re.split(r"(?<=[.!?])\s+", desc.strip(), maxsplit=1)
        return parts[0].strip() or " ".join(desc.split()[:30])

    def rank_of(j):
        q = embed_text(holdout[j]["description"]).to(dev)
        sims = (q @ th.T).squeeze(0)
        return int((sims >= sims[j]).sum().item()), sims

    # ---- diversity-aware row selection with SIBLING COLLAPSE in display ranking ----
    # IR-TD's video-derived sources contribute sequence siblings: frames that pass a
    # loose phash test yet read as the same shot. Display rule (disclosed in the
    # figure footer): each row shows the exact match (rank 1) plus the next ranked
    # results AFTER collapsing near-duplicates, where near-duplicate means
    #   phash distance <= 20  OR  (same size-signature AND embedding cosine >= 0.97).
    # Walking down the ranking skips siblings of anything already shown. The same
    # test enforces cross-row uniqueness. Rows that cannot fill 5 diverse thumbnails
    # switch to a different query (candidates with naturally diverse neighborhoods).
    import imagehash

    PH_T = 20                    # display-dedup phash threshold (strict)
    COS_T = 0.97                 # semantic-sibling cosine threshold (same size class)
    SCAN_CAP = 300               # how deep the ranking walk may go
    ph_cache: dict = {}
    size_cache: dict = {}

    def ph(j):
        if j not in ph_cache:
            ph_cache[j] = imagehash.phash(Image.open(paths[j]))
        return ph_cache[j]

    def phd(a, b):
        return int((ph(a).hash != ph(b).hash).sum())

    def size_class(j):
        if j not in size_cache:
            with Image.open(paths[j]) as im:
                size_cache[j] = im.size
        return size_cache[j]

    cosm = th @ th.T                                       # [2000,2000], unit-norm embeds

    def near_disp(a, b):
        if phd(a, b) <= PH_T:
            return True
        return size_class(a) == size_class(b) and float(cosm[a, b]) >= COS_T

    def build_row(j, sims):
        """Exact match first, then ranked results with siblings collapsed."""
        ranked = torch.argsort(sims, descending=True).tolist()[:SCAN_CAP]
        disp, skipped = [j], 0
        for i in ranked:
            if i == j:
                continue
            if len(disp) == 5:
                break
            if any(near_disp(i, d) for d in disp):
                skipped += 1
                continue
            disp.append(i)
        return (disp if len(disp) == 5 else None), skipped

    def legible(j, sims, row):
        """A demo row must convince a human in one second: the exact match needs a
        clearly visible bright subject (thermal contrast) and must stand out from
        its collapsed neighborhood (similarity margin), or the first sentence will
        appear to fit a neighbor as well as the match."""
        g = np.asarray(Image.open(paths[j]).convert("L"), dtype=np.float32)
        contrast = float(g.std())
        hot_frac = float((g >= 200).mean())
        margin = float(sims[j] - sims[row[1]])         # vs first displayed non-sibling
        return (contrast >= 45.0 and hot_frac >= 0.005 and margin >= 0.02,
                {"contrast": round(contrast, 1), "hot_frac": round(hot_frac, 4),
                 "margin": round(margin, 4)})

    chosen, used_imgs, used_classes = [], [], set()
    for bname, kws in BUCKETS:
        cands = [j for j, r in enumerate(holdout)
                 if any(k in first_sentence(r["description"]).lower() for k in kws)][:60]
        cands.sort(key=lambda j: size_class(j) in used_classes)
        picked = None
        for j in cands:
            rk, sims = rank_of(j)
            if rk != 1:
                continue
            row, skipped = build_row(j, sims)
            if row is None:
                continue
            if any(near_disp(a, u) for a in row for u in used_imgs):
                continue
            ok, stats = legible(j, sims, row)
            if not ok:
                continue
            picked = (j, row, skipped, stats)
            break
        if picked is None:
            print(f"GALLERY bucket {bname!r}: no rank-1 row passing display dedup, "
                  "skipped", flush=True)
            continue
        j, row, skipped, leg_stats = picked
        used_imgs.extend(row)
        used_classes.add(size_class(j))
        chosen.append({"bucket": bname, "query_idx": j,
                       "caption_first_sentence": first_sentence(holdout[j]["description"]),
                       "true_rank": 1, "top5_idx": row,
                       "top5_is_exact": [i == j for i in row],
                       "size_class": list(size_class(j)),
                       "display_skipped_siblings": skipped,
                       "legibility": leg_stats})
        print(f"GALLERY {bname}: rank-1 row, {skipped} siblings collapsed, "
              f"class {size_class(j)}", flush=True)

    # hard asserts at the strict thresholds (this is the contract; regeneration
    # cannot regress it), plus the verification numbers the composer re-checks
    assert all(c["true_rank"] == 1 for c in chosen)
    flat = [i for c in chosen for i in c["top5_idx"]]
    assert len(set(flat)) == len(flat), "image reused across rows"
    min_phd, max_cos_same = 64, 0.0
    for x_i in range(len(flat)):
        for y_i in range(x_i + 1, len(flat)):
            a, b = flat[x_i], flat[y_i]
            assert not near_disp(a, b), f"display near-duplicate survives: {a} vs {b}"
            min_phd = min(min_phd, phd(a, b))
            if size_class(a) == size_class(b):
                max_cos_same = max(max_cos_same, float(cosm[a, b]))
    display_contract = {"phash_threshold": PH_T, "cos_threshold": COS_T,
                        "min_pairwise_phash": min_phd,
                        "max_pairwise_cos_same_size": round(max_cos_same, 4)}
    print(f"GALLERY display contract: {json.dumps(display_contract)}", flush=True)

    # export 320px thumbnails for every image referenced
    need = sorted({i for c in chosen for i in c["top5_idx"]} |
                  {c["query_idx"] for c in chosen})
    for i in need:
        im = Image.open(paths[i]).convert("RGB")
        im.thumbnail((320, 320))
        im.save(os.path.join(OUT, f"thumb_{i}.png"))
    meta = {"queries": chosen, "gallery_size": 2000, "ckpt": CKPT,
            "display_contract": display_contract,
            "note": "rank-1 retrievals only; retrieval used full captions; displayed captions are first sentences",
            "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "gallery.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    volume.commit()
    print("GALLERY_RESULT:", json.dumps(meta), flush=True)
    return meta


@app.function(gpu="A100-80GB", image=image, secrets=[hf_secret],
              volumes={"/vol": volume}, timeout=1800, memory=32768)
def audit() -> dict:
    """Alignment audit: prove the caption<->image pairing survives the pipeline.

    (a) exports 10 seeded-random holdout pairs (thumbnail + first sentence) for a
        human contact-sheet check;
    (b) fingerprints the manifest (ordered image_path list) so any consumer can
        assert it is reading the same ordering;
    (c) provenance of the cached gallery embeddings: re-embeds 5 seeded-random
        holdout images fresh and compares to the cached rows at the same indices
        (cosine ~1.0 proves the cache rows follow the manifest order).
    """
    import hashlib
    import json
    import os
    import re
    import sys
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    sys.path.insert(0, "/root/fe")
    from fusion_embedding.adapters import AdapterPacks

    t0 = time.time()
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)

    # EXACT holdout construction of build() (single source of truth)
    with open(os.path.join(IRTD_ROOT, "new_relative.json"), encoding="utf-8") as f:
        records = json.load(f)
    stripped = set(json.load(open(STRIP, encoding="utf-8"))["excluded"])
    kept = [r for r in records if r["image_path"] not in stripped]
    order = np.random.RandomState(0).permutation(len(kept))
    holdout = [kept[i] for i in sorted(order[:2000].tolist())]
    paths = [os.path.join(IRTD_ROOT, r["image_path"]) for r in holdout]

    manifest = [r["image_path"] for r in holdout]
    fingerprint = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()

    def first_sentence(desc):
        parts = re.split(r"(?<=[.!?])\s+", desc.strip(), maxsplit=1)
        return parts[0].strip() or " ".join(desc.split()[:30])

    # (a) 10 seeded-random pairs for the human check
    rng = np.random.RandomState(7)
    sample = sorted(rng.choice(len(holdout), 10, replace=False).tolist())
    pairs = []
    for j in sample:
        im = Image.open(paths[j]).convert("RGB")
        im.thumbnail((360, 360))
        im.save(os.path.join(OUT, f"audit_{j}.png"))
        pairs.append({"idx": j, "image_path": holdout[j]["image_path"],
                      "first_sentence": first_sentence(holdout[j]["description"])})

    # (c) cache provenance: fresh re-embed of 5 indices vs cached rows
    th = torch.load(os.path.join(OUT, "th_embs.pt"), map_location="cpu")
    assert th.shape[0] == len(holdout), f"cache rows {th.shape[0]} != {len(holdout)}"
    from transformers import AutoModel, AutoProcessor
    base = AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-2B",
                                     trust_remote_code=True,
                                     dtype=torch.bfloat16).to(dev).eval()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-Embedding-2B",
                                         trust_remote_code=True)
    ip = getattr(proc, "image_processor", None)
    if ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = 1310720
    lm = base.language_model if hasattr(base, "language_model") else base.model.language_model
    packs = AdapterPacks()
    adapters, _ = packs.add_pack("thermal", lm, 2048, 384)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    adapters.load_state_dict(ck["thermal_adapters"])
    packs.to(dev)

    def pool(h, mask):
        idx = mask.long().cumsum(1).argmax(1)
        return h[torch.arange(h.shape[0], device=h.device), idx]

    rng2 = np.random.RandomState(11)
    probe = sorted(rng2.choice(len(holdout), 5, replace=False).tolist())
    cos = {}
    with torch.no_grad():
        for j in probe:
            im = Image.open(paths[j]).convert("RGB")
            text = _chat(THERMAL_INSTRUCTION,
                         "<|vision_start|><|image_pad|><|vision_end|>")
            inp = proc(text=[text], images=[im], return_tensors="pt").to(dev)
            with packs.scope("thermal"):
                h = base(**inp).last_hidden_state
            v = F.normalize(pool(h, inp["attention_mask"]).float(), dim=-1).cpu()
            cos[j] = round(float((v @ th[j][:, None]).item()), 6)

    out = {"manifest_fingerprint": fingerprint, "manifest_n": len(manifest),
           "pairs": pairs, "cache_provenance_cos": cos,
           "cache_rows_match_manifest": all(c > 0.999 for c in cos.values()),
           "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    volume.commit()
    print("AUDIT_RESULT:", json.dumps(out), flush=True)
    return out
