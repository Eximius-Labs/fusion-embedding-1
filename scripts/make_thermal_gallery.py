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

    # ---- diversity-aware row selection (rank-1 only) ----
    # IR-TD holdout carries near-duplicate consecutive frames from video-derived
    # sources; naive picks collapse rows onto the same clusters. Constraints:
    #   * rank-1 retrieval only (the gallery shows what correct retrieval looks
    #     like; honesty lives in the card's reported R@1)
    #   * global uniqueness: no image, nor any phash-near-duplicate (dist<=8) of
    #     one, appears in more than one row (across ALL thumbnails)
    #   * within-row diversity: min pairwise phash distance among the top-5 > 8
    #   * scene diversity: prefer an unused size-signature class per row (proxy
    #     for the source collection)
    import imagehash

    PH_T = 8
    ph_cache: dict = {}

    def ph(j):
        if j not in ph_cache:
            ph_cache[j] = imagehash.phash(Image.open(paths[j]))
        return ph_cache[j]

    def near(a, b):
        return (ph(a).hash != ph(b).hash).sum() <= PH_T

    chosen, used_imgs, used_classes = [], [], set()

    def row_ok(j, top5):
        if any(near(t, u) for t in top5 for u in used_imgs):
            return False                                   # collides with a used image
        for a_i in range(5):
            for b_i in range(a_i + 1, 5):
                if near(top5[a_i], top5[b_i]):
                    return False                           # row of near-duplicates
        return True

    for bname, kws in BUCKETS:
        cands = [j for j, r in enumerate(holdout)
                 if any(k in first_sentence(r["description"]).lower() for k in kws)][:60]
        # prefer candidates whose size class is new (source diversity proxy)
        def size_class(j):
            with Image.open(paths[j]) as im:
                return im.size
        cands.sort(key=lambda j: size_class(j) in used_classes)
        picked = None
        for j in cands:
            rk, sims = rank_of(j)
            if rk != 1:
                continue
            top5 = torch.topk(sims, 5).indices.tolist()
            if not row_ok(j, top5):
                continue
            picked = (j, top5)
            break
        if picked is None:
            print(f"GALLERY bucket {bname!r}: no rank-1 row passing diversity, skipped",
                  flush=True)
            continue
        j, top5 = picked
        used_imgs.extend(top5)
        used_classes.add(size_class(j))
        chosen.append({"bucket": bname, "query_idx": j,
                       "caption_first_sentence": first_sentence(holdout[j]["description"]),
                       "true_rank": 1, "top5_idx": top5,
                       "top5_is_exact": [i == j for i in top5],
                       "size_class": list(size_class(j)),
                       "top5_phash": [str(ph(i)) for i in top5]})
        print(f"GALLERY {bname}: rank-1 row accepted, class {size_class(j)}", flush=True)

    # hard asserts: regeneration cannot regress the diversity contract
    assert all(c["true_rank"] == 1 for c in chosen)
    flat = [i for c in chosen for i in c["top5_idx"]]
    for x_i in range(len(flat)):
        for y_i in range(x_i + 1, len(flat)):
            if flat[x_i] != flat[y_i]:
                assert not near(flat[x_i], flat[y_i]), \
                    f"near-duplicate across rows: {flat[x_i]} vs {flat[y_i]}"
    assert len(set(flat)) == len(flat), "image reused across rows"

    # export 320px thumbnails for every image referenced
    need = sorted({i for c in chosen for i in c["top5_idx"]} |
                  {c["query_idx"] for c in chosen})
    for i in need:
        im = Image.open(paths[i]).convert("RGB")
        im.thumbnail((320, 320))
        im.save(os.path.join(OUT, f"thumb_{i}.png"))
    meta = {"queries": chosen, "gallery_size": 2000, "ckpt": CKPT,
            "note": "rank-1 retrievals only; retrieval used full captions; displayed captions are first sentences",
            "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "gallery.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    volume.commit()
    print("GALLERY_RESULT:", json.dumps(meta), flush=True)
    return meta
