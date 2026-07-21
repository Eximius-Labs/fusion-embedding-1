"""Build the demo-Space galleries: license-clean audio + images, embedded with
the released FE2 checkpoint at the pinned revision.

Two-phase per the multi-phase lesson: a CPU function does all downloads and
curation (no GPU idling), then a GPU function embeds the curated assets.
Everything lands on the fusion-data Volume under demo_space/ and is committed
after each phase; phase markers make the driver polling-safe.

Audio: FSD50K *eval* clips (Zenodo), filtered to CC0 / CC-BY per-clip licenses
from the official metadata; attribution manifest kept. Eval clips are outside
our training corpus (we train on FSD50K dev only), so demo assets are also
clean of training data.
Images: Openverse API, license=cc0,by only, attribution manifest kept.

Run:
    PYTHONUTF8=1 uv run modal deploy scripts/demo_space/build_galleries.py
    uv run python -c "import modal; print(modal.Function.from_name('fusion-demo-galleries','build_assets').spawn().object_id)"
    # poll /vol/demo_space/_ASSETS_OK, then:
    uv run python -c "import modal; print(modal.Function.from_name('fusion-demo-galleries','embed_assets').spawn().object_id)"
"""
from __future__ import annotations

import modal

app = modal.App("fusion-demo-galleries")
volume = modal.Volume.from_name("fusion-data")
hf_secret = modal.Secret.from_name("huggingface")

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "p7zip-full")
    .pip_install("requests", "soundfile", "pillow", "numpy")
    .env({"PYTHONUTF8": "1"})
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0", "numpy>=1.24", "transformers>=4.46",
        "accelerate>=0.30", "soundfile>=0.12", "librosa>=0.10",
        "pillow>=10.0", "safetensors", "huggingface_hub",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"PYTHONUTF8": "1"})
)

ROOT = "/vol/demo_space"
MODEL = "EximiusLabs/fusion-embedding-2-2b-preview"
REVISION = "ad45028fbeb638fcc333690bd6119e02a66dc7d7"  # v0.2-preview tag, resolved 2026-07-21

N_AUDIO = 220
N_IMG_PER_TERM = 9

IMAGE_TERMS = [
    "dog barking", "cat", "rooster", "songbird", "ocean waves", "thunderstorm",
    "rain on window", "waterfall", "campfire", "fireworks", "helicopter",
    "airplane takeoff", "steam train", "motorcycle", "city traffic",
    "church bells", "acoustic guitar", "grand piano", "violin player",
    "drum kit", "trumpet player", "orchestra", "choir singing",
    "stadium crowd", "children playground", "typing keyboard", "chainsaw",
    "lawn mower", "kitchen blender", "vacuum cleaner", "washing machine",
    "grandfather clock", "wind chimes", "frog pond", "crickets meadow",
    "owl", "horse galloping", "cow", "sheep flock", "seagulls",
    "honey bees", "glass shattering", "wooden door", "footsteps stairs",
    "ambulance siren", "police car", "waterfall forest", "blacksmith forge",
    "sewing machine", "table tennis",
]


def _lic_ok(url: str) -> bool:
    u = (url or "").lower()
    if "publicdomain/zero" in u or "/zero/" in u:
        return True
    return ("/by/" in u) and ("nc" not in u) and ("sa" not in u) and ("nd" not in u)


@app.function(image=cpu_image, volumes={"/vol": volume}, timeout=3 * 3600,
              cpu=4, memory=8192)
def build_assets() -> dict:
    import csv
    import io
    import json
    import os
    import random
    import subprocess
    import time
    import zipfile

    import requests
    import soundfile as sf
    from PIL import Image

    os.makedirs(f"{ROOT}/audio_wav", exist_ok=True)
    os.makedirs(f"{ROOT}/audio", exist_ok=True)
    os.makedirs(f"{ROOT}/images", exist_ok=True)
    os.makedirs(f"{ROOT}/examples", exist_ok=True)
    marker = f"{ROOT}/_ASSETS_OK"
    if os.path.exists(marker):
        print("assets already built"); return json.load(open(marker))

    s = requests.Session(); s.headers["User-Agent"] = "fusion-embedding-demo-builder"

    # ---------------- audio: FSD50K eval, license-filtered ----------------
    def zget(name, dest):
        if os.path.exists(dest):
            return
        url = f"https://zenodo.org/records/4060432/files/{name}?download=1"
        print("downloading", name, flush=True)
        with s.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)

    tmp = "/tmp/fsd"
    os.makedirs(tmp, exist_ok=True)
    zget("FSD50K.metadata.zip", f"{tmp}/meta.zip")
    zget("FSD50K.ground_truth.zip", f"{tmp}/gt.zip")
    with zipfile.ZipFile(f"{tmp}/meta.zip") as z:
        info = json.load(io.TextIOWrapper(
            z.open("FSD50K.metadata/eval_clips_info_FSD50K.json"), "utf-8"))
    with zipfile.ZipFile(f"{tmp}/gt.zip") as z:
        rows = list(csv.DictReader(io.TextIOWrapper(
            z.open("FSD50K.ground_truth/eval.csv"), "utf-8")))

    by_label: dict[str, list] = {}
    for r in rows:
        cid = r["fname"]; meta = info.get(cid) or {}
        if not _lic_ok(meta.get("license", "")):
            continue
        labels = [x for x in r["labels"].split(",") if x]
        if not labels:
            continue
        by_label.setdefault(labels[0], []).append((cid, labels, meta))
    rng = random.Random(0)
    order = sorted(by_label); [rng.shuffle(by_label[k]) for k in order]
    candidates = []
    while len(candidates) < N_AUDIO * 3 and any(by_label.values()):
        for k in order:
            if by_label[k]:
                candidates.append(by_label[k].pop())
                if len(candidates) >= N_AUDIO * 3:
                    break
    want = {c[0] for c in candidates}
    cand_by_id = {c[0]: c for c in candidates}
    print(f"license-ok candidates: {len(candidates)} across {len(order)} labels", flush=True)

    # FSD50K eval audio is a SPLIT zip (.z01 + .zip): python zipfile cannot
    # open it; download both parts and extract with 7z (handles spanning).
    zget("FSD50K.eval_audio.z01", f"{tmp}/FSD50K.eval_audio.z01")
    zget("FSD50K.eval_audio.zip", f"{tmp}/FSD50K.eval_audio.zip")
    import subprocess
    subprocess.run(["7z", "x", "-y", f"{tmp}/FSD50K.eval_audio.zip",
                    f"-o{tmp}/eval_x"], check=True,
                   stdout=subprocess.DEVNULL, timeout=3600)
    import glob as _glob
    wavs = {os.path.basename(f).split(".")[0]: f
            for f in _glob.glob(f"{tmp}/eval_x/**/*.wav", recursive=True)}
    print(f"extracted {len(wavs)} eval wavs", flush=True)
    audio_manifest, kept = [], 0
    if True:
        names = wavs
        for cid in list(want):
            if kept >= N_AUDIO:
                break
            n = names.get(cid)
            if not n:
                continue
            raw = open(n, "rb").read()
            try:
                data, sr = sf.read(io.BytesIO(raw))
            except Exception:
                continue
            dur = len(data) / float(sr)
            if not (1.5 <= dur <= 22.0):
                continue
            wavp = f"{ROOT}/audio_wav/{cid}.wav"
            with open(wavp, "wb") as f:
                f.write(raw)
            oggp = f"{ROOT}/audio/{cid}.ogg"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wavp,
                            "-ac", "1", "-b:a", "64k", oggp], check=True)
            _, labels, meta = cand_by_id[cid]
            audio_manifest.append({
                "id": cid, "file": f"audio/{cid}.ogg",
                "labels": [x.replace("_", " ") for x in labels],
                "title": meta.get("title", ""), "uploader": meta.get("uploader", ""),
                "license": meta.get("license", ""),
                "source": f"https://freesound.org/s/{cid}/",
                "duration_s": round(dur, 1),
            })
            kept += 1
    json.dump(audio_manifest, open(f"{ROOT}/audio_attribution.json", "w",
                                   encoding="utf-8"), indent=1)
    volume.commit()
    print(f"audio kept: {kept}", flush=True)

    # ---------------- images: Openverse cc0/by ----------------
    img_manifest, idx = [], 0
    for term in IMAGE_TERMS:
        try:
            r = s.get("https://api.openverse.org/v1/images/",
                      params={"q": term, "license": "cc0,by", "page_size": 20,
                              "mature": "false"}, timeout=30)
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as e:
            print("openverse fail", term, e, flush=True)
            continue
        got = 0
        for it in results:
            if got >= N_IMG_PER_TERM:
                break
            if not _lic_ok(f"/{it.get('license','')}/"):
                continue
            url = it.get("url") or it.get("thumbnail")
            if not url:
                continue
            try:
                ir = s.get(url, timeout=25)
                ir.raise_for_status()
                im = Image.open(io.BytesIO(ir.content)).convert("RGB")
            except Exception:
                continue
            im.thumbnail((512, 512))
            fn = f"images/{idx:05d}.jpg"
            im.save(f"{ROOT}/{fn}", "JPEG", quality=87)
            img_manifest.append({
                "file": fn, "term": term, "title": it.get("title") or "",
                "creator": it.get("creator") or "",
                "license": f"CC {it.get('license','').upper()} {it.get('license_version','')}".strip(),
                "source": it.get("foreign_landing_url") or "",
                "provider": it.get("source") or "",
            })
            idx += 1; got += 1
        time.sleep(1.2)
    json.dump(img_manifest, open(f"{ROOT}/image_attribution.json", "w",
                                 encoding="utf-8"), indent=1)

    out = {"audio": kept, "images": idx}
    json.dump(out, open(marker, "w"))
    volume.commit()
    print("ASSETS_OK:", json.dumps(out), flush=True)
    return out


@app.function(image=gpu_image, volumes={"/vol": volume}, gpu="L4",
              secrets=[hf_secret], timeout=2 * 3600)
def embed_assets() -> dict:
    import glob
    import json
    import os
    import random

    import librosa
    import torch
    from PIL import Image
    from transformers import AutoModel

    marker = f"{ROOT}/_INDEX_OK"
    if os.path.exists(marker):
        print("indexes already built"); return json.load(open(marker))

    model = AutoModel.from_pretrained(MODEL, revision=REVISION,
                                      trust_remote_code=True,
                                      torch_dtype=torch.float32).to("cuda").eval()

    audio_manifest = json.load(open(f"{ROOT}/audio_attribution.json", encoding="utf-8"))
    embs, meta = [], []
    for i, rec in enumerate(audio_manifest):
        wav, sr = librosa.load(f"{ROOT}/audio_wav/{rec['id']}.wav",
                               sr=16000, mono=True)
        with torch.no_grad():
            e = model.embed_audio(wav, sr=16000)
        embs.append(e.detach().float().cpu())
        meta.append({"file": rec["file"], "caption": ", ".join(rec["labels"][:3]),
                     "title": rec["title"], "license": rec["license"],
                     "source": rec["source"]})
        if i % 25 == 0:
            print(f"audio emb {i}/{len(audio_manifest)}", flush=True)
    A = torch.nn.functional.normalize(torch.stack(embs), dim=-1)
    torch.save({"emb": A.half(), "meta": meta, "modality": "audio"},
               f"{ROOT}/audio_index.pt")
    volume.commit()

    img_manifest = json.load(open(f"{ROOT}/image_attribution.json", encoding="utf-8"))
    embs, meta = [], []
    for i, rec in enumerate(img_manifest):
        try:
            im = Image.open(f"{ROOT}/{rec['file']}").convert("RGB")
            with torch.no_grad():
                e = model.embed_image(im)
        except Exception as ex:
            print("img emb fail", rec["file"], ex, flush=True)
            continue
        embs.append(e.detach().float().cpu())
        meta.append({"file": rec["file"], "caption": rec["term"],
                     "title": rec["title"], "creator": rec["creator"],
                     "license": rec["license"], "source": rec["source"]})
        if i % 50 == 0:
            print(f"image emb {i}/{len(img_manifest)}", flush=True)
    I = torch.nn.functional.normalize(torch.stack(embs), dim=-1)
    torch.save({"emb": I.half(), "meta": meta, "modality": "image"},
               f"{ROOT}/image_index.pt")

    # six diverse example clips for the sound->images tab
    rng = random.Random(7)
    prefer = ["Bark", "Thunder", "Church bell", "Acoustic guitar", "Helicopter",
              "Ocean", "Bell", "Guitar", "Rain", "Siren"]
    picked, used = [], set()
    for p in prefer:
        for rec in audio_manifest:
            if len(picked) >= 6:
                break
            if rec["id"] in used:
                continue
            if any(p.lower() in l.lower() for l in rec["labels"]):
                picked.append(rec); used.add(rec["id"]); break
    while len(picked) < 6:
        rec = rng.choice(audio_manifest)
        if rec["id"] not in used:
            picked.append(rec); used.add(rec["id"])
    ex = []
    for rec in picked:
        src = f"{ROOT}/{rec['file']}"; dst = f"{ROOT}/examples/{os.path.basename(rec['file'])}"
        open(dst, "wb").write(open(src, "rb").read())
        ex.append({"file": f"examples/{os.path.basename(rec['file'])}",
                   "caption": ", ".join(rec["labels"][:2])})
    json.dump(ex, open(f"{ROOT}/examples.json", "w", encoding="utf-8"), indent=1)

    out = {"audio_emb": int(A.shape[0]), "image_emb": int(I.shape[0]),
           "dim": int(A.shape[1]), "examples": len(ex), "revision": REVISION}
    json.dump(out, open(marker, "w"))
    volume.commit()
    print("INDEX_OK:", json.dumps(out), flush=True)
    return out
