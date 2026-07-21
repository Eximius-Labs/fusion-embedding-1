"""Speech/music corpus delta ingestion (research memo
docs/research_speech_gap_solutions.md section 6.2) — feeds the ~1M-pair pretrain.

Own Modal app so the shared ``fusion-embedding`` app is not redeployed mid-flight.
Each source lands as its own frame shard dir (domain = shard membership, which the
trainer's domain-homogeneous batching keys on):

  msw_train          ~150K 1-s keyword clips     (MSW, CC BY 4.0)
  librispeech_train  ~100-150K <=10s utterances  (OpenSLR tarballs, CC BY 4.0)
  fma_train          ~20K 30-s music clips       (FMA, per-track CC-BY/BY-SA filter)

Common Voice requires a Mozilla Data Collective account (checked by ``peek_sources``);
if gated it is skipped and LibriSpeech train-other-500 backfills the sentence quota.

All shards store the standard ``{frames, text, task="sound"}`` records (the audio-side
instruction stays the sound instruction; domain lives in the TEXT templates, per GLAP),
plus a ``domain`` list in index.json for the manifest and future domain-restricted banks.

Deploy + spawn:
    PYTHONUTF8=1 uv run modal deploy scripts/ingest_speech_delta.py
    uv run python -c "import modal; print(modal.Function.from_name('fusion-speech-delta','peek_sources').remote())"
    # then smokes with limit=100, then full spawns
"""

from __future__ import annotations

import modal

app = modal.App("fusion-speech-delta")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "zip", "unzip", "p7zip-full", "wget")
    .pip_install(
        "torch==2.6.0",
        "numpy>=1.24",
        "transformers>=4.46",
        "accelerate>=0.30",
        "soundfile>=0.12",
        "librosa>=0.10",
        "datasets>=3.0",
        "pandas>=2.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_python_source("fusion_embedding")
)

hf_secret = modal.Secret.from_name("huggingface")
volume = modal.Volume.from_name("fusion-data")
VOL = "/vol"
AUDIO_MODEL = "Qwen/Qwen2.5-Omni-7B"
HF_ENV = {"FUSION_DATA_ROOT": VOL, "HF_HOME": f"{VOL}/hf-cache",
          "HF_XET_HIGH_PERFORMANCE": "1"}

MANIFEST_DIR = f"{VOL}/corpus_speech_delta"


# ---------------------------------------------------------------------------
# shared GPU sink: wav -> mel -> frozen tower -> frame shards (+ index.json)
# ---------------------------------------------------------------------------
class _FrameSink:
    """Mirrors ingest_wavcaps_zip's buffering: chunked tower forwards, host-side
    accumulation only inside the current shard, per-shard Volume commits, heartbeat
    prints (P0b host-RAM lessons applied)."""

    def __init__(self, frame_shard: str, domain: str, shard_size: int = 512,
                 batch: int = 16, feature_layer: str = "post_proj"):
        import torch
        from transformers import AutoFeatureExtractor

        from fusion_embedding.hf_components import load_audio_tower
        from fusion_embedding.paths import frames_dir

        self.torch = torch
        self.fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL,
                                                       trust_remote_code=True)
        self.sr = self.fe.sampling_rate
        self.enc, _, self.d_audio = load_audio_tower(
            device="cuda", dtype=torch.bfloat16, audio_feature_layer=feature_layer)
        self.out_dir = frames_dir(frame_shard)
        self.domain = domain
        self.shard_size = shard_size
        self.batch = batch
        self.mel_buf: list = []
        self.cap_buf: list = []
        self.recs: list = []
        self.captions: list = []
        self.shards: list = []
        self.kept = 0
        self.min_frames = 10 ** 9

    def add(self, wav, sr0: int, caption: str) -> None:
        import librosa
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr0 != self.sr:
            wav = librosa.resample(wav, orig_sr=sr0, target_sr=self.sr)
        feats = self.fe(wav, sampling_rate=self.sr, return_tensors="pt",
                        return_attention_mask=True, padding="max_length",
                        truncation=True)
        mel = feats["input_features"][0]
        am = feats.get("attention_mask")
        if am is not None:
            mel = mel[:, : int(am[0].sum().item())]
        self.mel_buf.append(mel)
        self.cap_buf.append(caption)
        self.kept += 1
        if len(self.mel_buf) >= self.batch:
            self._run_tower()
        if len(self.recs) >= self.shard_size:
            self._write_shard()
        if self.kept % 2000 == 0:
            print(f"  kept={self.kept} shards={len(self.shards)} "
                  f"min_frames={self.min_frames}", flush=True)

    def _run_tower(self):
        torch = self.torch
        if not self.mel_buf:
            return
        n_mels = self.mel_buf[0].shape[0]
        fmax = max(m.shape[1] for m in self.mel_buf)
        mb = torch.zeros(len(self.mel_buf), n_mels, fmax, device="cuda")
        mm = torch.zeros(len(self.mel_buf), fmax, dtype=torch.bool, device="cuda")
        for i, m in enumerate(self.mel_buf):
            mb[i, :, : m.shape[1]] = m.to("cuda")
            mm[i, : m.shape[1]] = True
        with torch.no_grad():
            frames, fmask = self.enc(mb, mm)
        for i, cap in enumerate(self.cap_buf):
            t = int(fmask[i].sum().item())
            assert t >= 1, f"degenerate frame count t={t} (sub-2s input path)"
            self.min_frames = min(self.min_frames, t)
            self.recs.append({"frames": frames[i, :t].cpu().contiguous(),
                              "text": cap, "task": "sound"})
            self.captions.append(cap)
        self.mel_buf.clear()
        self.cap_buf.clear()

    def _write_shard(self):
        from fusion_embedding.data import write_frame_shard
        if not self.recs:
            return
        name = f"shard-{len(self.shards):04d}.pt"
        write_frame_shard(self.out_dir / name, self.recs, half=True)
        self.shards.append(name)
        self.recs.clear()
        volume.commit()

    def finalize(self) -> dict:
        import json
        self._run_tower()
        self._write_shard()
        with open(str(self.out_dir / "index.json"), "w") as fh:
            json.dump({"d_audio": self.d_audio, "shard_size": self.shard_size,
                       "n_total": self.kept, "captions": self.captions,
                       "tasks": ["sound"] * self.kept,
                       "domain": [self.domain] * self.kept,
                       "shards": self.shards}, fh)
        volume.commit()
        return {"kept": self.kept, "shards": len(self.shards),
                "min_frames": self.min_frames, "d_audio": self.d_audio}


# ---------------------------------------------------------------------------
# CPU probes: source structure, gates, mirrors
# ---------------------------------------------------------------------------
@app.function(image=image, secrets=[hf_secret], volumes={VOL: volume},
              cpu=2.0, memory=4096, timeout=1800, env=HF_ENV)
def peek_sources() -> dict:
    import json

    from huggingface_hub import HfApi

    api = HfApi()
    out: dict = {}

    # MSW: can datasets>=3.0 load it, and what files exist for a direct route?
    try:
        files = api.list_repo_files("MLCommons/ml_spoken_words",
                                    repo_type="dataset")
        en = [f for f in files if "/en" in f or f.startswith("en")][:40]
        out["msw_files_sample"] = en
        out["msw_n_files"] = len(files)
    except Exception as e:                                       # noqa: BLE001
        out["msw_files_error"] = f"{type(e).__name__}: {e}"
    try:
        from datasets import load_dataset
        ds = load_dataset("MLCommons/ml_spoken_words", "en_wav", split="train",
                          streaming=True, trust_remote_code=True)
        rec = next(iter(ds))
        out["msw_stream"] = {"ok": True,
                             "keys": sorted(rec.keys()),
                             "sample": {k: str(rec[k])[:80] for k in rec
                                        if k != "audio"}}
    except Exception as e:                                       # noqa: BLE001
        out["msw_stream"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}

    # Common Voice: gated?
    for repo in ("mozilla-foundation/common_voice_17_0",
                 "mozilla-foundation/common_voice_16_1"):
        try:
            info = api.dataset_info(repo)
            out[f"cv_{repo.split('_')[-2]}_{repo.split('_')[-1]}"] = {
                "gated": info.gated, "disabled": info.disabled}
        except Exception as e:                                   # noqa: BLE001
            out[f"cv_{repo[-4:]}"] = f"{type(e).__name__}: {str(e)[:120]}"

    # Jamendo mirror on HF?
    hits = []
    for q in ("mtg-jamendo", "jamendo"):
        try:
            for d in api.list_datasets(search=q, limit=10):
                hits.append(d.id)
        except Exception:                                        # noqa: BLE001
            pass
    out["jamendo_hf_candidates"] = sorted(set(hits))

    # FMA: metadata zip reachable?
    import urllib.request
    for url in ("https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
                "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as r:
                out[f"fma_{url.rsplit('/', 1)[-1]}"] = {
                    "status": r.status,
                    "size_gb": round(int(r.headers.get("Content-Length", 0)) / 1e9, 2)}
        except Exception as e:                                   # noqa: BLE001
            out[f"fma_{url.rsplit('/', 1)[-1]}"] = f"{type(e).__name__}: {str(e)[:120]}"

    print("PEEK_SOURCES:", json.dumps(out, default=str)[:3000], flush=True)
    return out


# ---------------------------------------------------------------------------
# MSW: word-balanced ~150K English 1-s keyword clips
# ---------------------------------------------------------------------------
@app.function(gpu="L4", image=image, secrets=[hf_secret], volumes={VOL: volume},
              cpu=8.0, memory=32768, timeout=20 * 3600, ephemeral_disk=512 * 1024,
              env=HF_ENV)
def ingest_msw(target: int = 150_000, per_word_cap: int = 30, limit: int = 0,
               frame_shard: str = "msw_train") -> dict:
    """English subset, word-balanced (cap clips/keyword), template-rotated captions.
    MSW is derived from Common Voice by forced alignment (disjoint from Google's
    SpeechCommands recordings by construction; provenance recorded)."""
    import collections
    import json

    from fusion_embedding.speech_captions import caption_for_word

    import io
    import tarfile

    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    tgt = limit or target
    sink = _FrameSink(frame_shard, domain="speech_word")
    per_word: dict = collections.Counter()
    seen = skipped_cap = bad = 0
    # datasets>=3.0 removed script loading; stream the wav tars directly.
    # Layout: data/wav/en/train/audio/N.tar.gz, members <word>/<clip>.wav
    api = HfApi()
    tars = sorted(
        (f for f in api.list_repo_files("MLCommons/ml_spoken_words",
                                        repo_type="dataset")
         if f.startswith("data/wav/en/train/audio/") and f.endswith(".tar.gz")),
        key=lambda p: int(p.rsplit("/", 1)[-1].split(".")[0]))
    print(f"MSW: {len(tars)} train tars", flush=True)
    import os
    for tf_name in tars:
        if sink.kept >= tgt:
            break
        local = hf_hub_download("MLCommons/ml_spoken_words", tf_name,
                                repo_type="dataset")
        with tarfile.open(local, "r:gz") as tar:
            for m in tar:
                if sink.kept >= tgt:
                    break
                if not m.name.endswith((".wav", ".opus")):
                    continue
                seen += 1
                word = os.path.basename(os.path.dirname(m.name)).strip()
                if not word:
                    continue
                if per_word[word] >= per_word_cap:
                    skipped_cap += 1
                    continue
                try:
                    wav, sr0 = sf.read(io.BytesIO(tar.extractfile(m).read()),
                                       dtype="float32")
                except Exception:                                # noqa: BLE001
                    bad += 1
                    continue
                per_word[word] += 1
                sink.add(wav, sr0, caption_for_word(word, sink.kept))
        os.remove(local)
    stats = sink.finalize()
    result = {"frame_shard": frame_shard, "source": "MLCommons/ml_spoken_words:en",
              "license": "CC-BY-4.0", "seen": seen, "skipped_word_cap": skipped_cap,
              "decode_fail": bad, "distinct_words": len(per_word), **stats}
    print("INGEST_MSW:", json.dumps(result), flush=True)
    return result


# ---------------------------------------------------------------------------
# LibriSpeech: OpenSLR tarballs -> <=10s utterances, transcript-as-caption
# ---------------------------------------------------------------------------
_OPENSLR = "https://www.openslr.org/resources/12/{name}.tar.gz"


@app.function(gpu="L4", image=image, secrets=[hf_secret], volumes={VOL: volume},
              cpu=8.0, memory=32768, timeout=20 * 3600, ephemeral_disk=512 * 1024,
              env=HF_ENV)
def ingest_librispeech(target: int = 100_000, limit: int = 0, max_dur_s: float = 10.0,
                       subsets: str = "train-clean-100,train-clean-360",
                       frame_shard: str = "librispeech_train") -> dict:
    """OpenSLR tarballs (CC BY 4.0) streamed to ephemeral disk; keeps utterances
    <= max_dur_s; transcripts normalized to sentence case; template rotation."""
    import json
    import os
    import subprocess
    import tarfile

    import soundfile as sf

    from fusion_embedding.speech_captions import caption_for_transcript

    tgt = limit or target
    sink = _FrameSink(frame_shard, domain="speech_sentence")
    too_long = bad = 0
    for name in [s.strip() for s in subsets.split(",") if s.strip()]:
        if sink.kept >= tgt:
            break
        tar_path = f"/tmp/{name}.tar.gz"
        url = _OPENSLR.format(name=name)
        print(f"downloading {url} ...", flush=True)
        subprocess.run(["wget", "-q", "--tries=3", "-O", tar_path, url], check=True,
                       timeout=4 * 3600)
        print(f"  {name}: {os.path.getsize(tar_path) / 1e9:.1f} GB", flush=True)
        # extract sequentially (tar is seek-hostile); transcripts arrive near their flacs
        with tarfile.open(tar_path, "r:gz") as tar:
            trans: dict = {}
            for m in tar:
                if sink.kept >= tgt:
                    break
                if m.name.endswith(".trans.txt"):
                    fh = tar.extractfile(m)
                    for line in fh.read().decode("utf-8").splitlines():
                        uid, _, txt = line.partition(" ")
                        trans[uid] = txt
                elif m.name.endswith(".flac"):
                    uid = os.path.basename(m.name)[:-5]
                    txt = trans.get(uid)
                    if txt is None:
                        continue                       # transcript block not seen yet
                    fh = tar.extractfile(m)
                    try:
                        wav, sr0 = sf.read(fh, dtype="float32")
                    except Exception:                            # noqa: BLE001
                        bad += 1
                        continue
                    if len(wav) / sr0 > max_dur_s:
                        too_long += 1
                        continue
                    sink.add(wav, sr0, caption_for_transcript(txt, sink.kept))
        os.remove(tar_path)
    stats = sink.finalize()
    result = {"frame_shard": frame_shard, "source": f"OpenSLR-12:{subsets}",
              "license": "CC-BY-4.0", "too_long": too_long, "decode_fail": bad,
              **stats}
    print("INGEST_LIBRISPEECH:", json.dumps(result), flush=True)
    return result


# ---------------------------------------------------------------------------
# FMA: license-filtered music with genre template captions
# ---------------------------------------------------------------------------
@app.function(gpu="L4", image=image, secrets=[hf_secret], volumes={VOL: volume},
              cpu=8.0, memory=32768, timeout=20 * 3600, ephemeral_disk=512 * 1024,
              env=HF_ENV)
def ingest_fma(target: int = 20_000, limit: int = 0, size: str = "medium",
               frame_shard: str = "fma_train") -> dict:
    """fma_<size>.zip + fma_metadata.zip; keeps tracks whose per-track AUDIO license
    is CC-BY / CC-BY-SA / public domain (license_allowed); genre template captions."""
    import io
    import json
    import os
    import subprocess
    import zipfile

    import pandas as pd
    import soundfile as sf

    from fusion_embedding.speech_captions import caption_for_genre, license_allowed

    tgt = limit or target
    # 1) metadata: per-track license + top genre
    subprocess.run(["wget", "-q", "--tries=3", "-O", "/tmp/fma_metadata.zip",
                    "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"],
                   check=True, timeout=3600)
    zf = zipfile.ZipFile("/tmp/fma_metadata.zip")
    with zf.open("fma_metadata/raw_tracks.csv") as fh:
        raw = pd.read_csv(fh, index_col=0, low_memory=False)
    with zf.open("fma_metadata/tracks.csv") as fh:
        tracks = pd.read_csv(fh, index_col=0, header=[0, 1], low_memory=False)
    genre_top = tracks[("track", "genre_top")]
    lic_col = "track_license" if "track_license" in raw.columns else "license_title"
    lic_url = raw["license_url"] if "license_url" in raw.columns else None
    allowed: dict = {}
    for tid, lic in raw[lic_col].items():
        cand = f"{lic} {lic_url.get(tid, '')}" if lic_url is not None else str(lic)
        g = genre_top.get(tid)
        if license_allowed(str(cand)) and isinstance(g, str) and g:
            allowed[int(tid)] = g
    print(f"FMA: {len(allowed)} tracks pass license+genre filter "
          f"(of {len(raw)})", flush=True)

    # 2) audio zip
    subprocess.run(["wget", "-q", "--tries=3", "-O", f"/tmp/fma_{size}.zip",
                    f"https://os.unil.cloud.switch.ch/fma/fma_{size}.zip"],
                   check=True, timeout=6 * 3600)
    sink = _FrameSink(frame_shard, domain="music")
    bad = not_allowed = 0
    with zipfile.ZipFile(f"/tmp/fma_{size}.zip") as az:
        for nm in az.namelist():
            if sink.kept >= tgt:
                break
            if not nm.endswith(".mp3"):
                continue
            try:
                tid = int(os.path.splitext(os.path.basename(nm))[0])
            except ValueError:
                continue
            genre = allowed.get(tid)
            if genre is None:
                not_allowed += 1
                continue
            try:
                data = az.read(nm)
                wav, sr0 = sf.read(io.BytesIO(data), dtype="float32")
            except Exception:                                    # noqa: BLE001
                bad += 1
                continue
            sink.add(wav, sr0, caption_for_genre(genre, sink.kept))
    stats = sink.finalize()
    result = {"frame_shard": frame_shard, "source": f"FMA:{size}",
              "license": "per-track CC-BY/CC-BY-SA (filtered)",
              "license_pass": len(allowed), "skipped_license": not_allowed,
              "decode_fail": bad, **stats}
    print("INGEST_FMA:", json.dumps(result), flush=True)
    return result


# ---------------------------------------------------------------------------
# MTG-Jamendo via the rkstgr/mtg-jamendo webdataset mirror, license-filtered
# ---------------------------------------------------------------------------
@app.function(gpu="L4", image=image, secrets=[hf_secret], volumes={VOL: volume},
              cpu=8.0, memory=32768, timeout=20 * 3600, ephemeral_disk=512 * 1024,
              env=HF_ENV)
def ingest_jamendo(target: int = 40_000, limit: int = 0,
                   frame_shard: str = "jamendo_train") -> dict:
    """Streams data/train/N.tar from rkstgr/mtg-jamendo (mp3 + inline genre/
    instrument/mood tags), keeping only track ids whose audio license in MTG's
    audio_licenses.txt is CC-BY / CC-BY-SA (license_allowed). Caption from tags:
    genre template extended with instruments/mood when present. Diligence note:
    MTG's tag METADATA file is CC BY-NC-SA; we use tags as facts about the audio,
    and record the caveat in the manifest (research memo section 4)."""
    import io as _io
    import json
    import os
    import tarfile
    import urllib.request

    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    from fusion_embedding.speech_captions import (caption_for_genre,
                                                  license_allowed)

    tgt = limit or target
    # 1) per-track license map from MTG (3-line blocks: NN/<id>.mp3 / by / license)
    lic_url = ("https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/"
               "master/audio_licenses.txt")
    allowed: set = set()
    total_ids = 0
    with urllib.request.urlopen(lic_url, timeout=120) as r:
        block: list = []
        for raw in _io.TextIOWrapper(r, encoding="utf-8", errors="replace"):
            line = raw.strip()
            if not line:
                block = []
                continue
            block.append(line)
            if len(block) == 3 and block[0].endswith(".mp3"):
                total_ids += 1
                tid = os.path.splitext(os.path.basename(block[0]))[0]
                if license_allowed(block[2]):
                    allowed.add(tid)
    print(f"JAMENDO: {len(allowed)}/{total_ids} track ids license-allowed",
          flush=True)

    sink = _FrameSink(frame_shard, domain="music")
    api = HfApi()
    tars = sorted((f for f in api.list_repo_files("rkstgr/mtg-jamendo",
                                                  repo_type="dataset")
                   if f.startswith("data/train/") and f.endswith(".tar")),
                  key=lambda p: int(p.rsplit("/", 1)[-1].split(".")[0]))
    print(f"JAMENDO: {len(tars)} mirror tars", flush=True)
    skipped_lic = bad = 0
    for tf_name in tars:
        if sink.kept >= tgt:
            break
        local = hf_hub_download("rkstgr/mtg-jamendo", tf_name, repo_type="dataset")
        with tarfile.open(local, "r:") as tar:
            meta: dict = {}
            wav_bytes: dict = {}
            for m in tar:
                stem, ext = os.path.splitext(os.path.basename(m.name))
                if ext == ".json":
                    meta[stem] = json.loads(tar.extractfile(m).read())
                elif ext in (".mp3", ".flac", ".ogg", ".wav"):
                    wav_bytes[stem] = tar.extractfile(m).read()
            for stem, data in wav_bytes.items():
                if sink.kept >= tgt:
                    break
                info = meta.get(stem, {})
                tid = str(info.get("id", stem)).lstrip("0") or stem
                if tid not in allowed:
                    skipped_lic += 1
                    continue
                genres = info.get("genres") or []
                if not genres:
                    continue
                try:
                    wav, sr0 = sf.read(_io.BytesIO(data), dtype="float32")
                except Exception:                                # noqa: BLE001
                    bad += 1
                    continue
                cap = caption_for_genre(str(genres[0]), sink.kept)
                instr = info.get("instruments") or []
                if instr and sink.kept % 2 == 0:
                    cap = cap.rstrip(".") + f", featuring {instr[0]}."
                sink.add(wav, sr0, cap)
        os.remove(local)
    stats = sink.finalize()
    result = {"frame_shard": frame_shard, "source": "rkstgr/mtg-jamendo (MTG mirror)",
              "license": "per-track CC-BY/CC-BY-SA (audio_licenses.txt filtered)",
              "license_allowed_ids": len(allowed), "skipped_license": skipped_lic,
              "decode_fail": bad, **stats}
    print("INGEST_JAMENDO:", json.dumps(result), flush=True)
    return result


# ---------------------------------------------------------------------------
# manifest over whatever delta shards exist
# ---------------------------------------------------------------------------
@app.function(image=image, secrets=[hf_secret], volumes={VOL: volume},
              cpu=2.0, memory=8192, timeout=1800, env=HF_ENV)
def write_manifest(shards: str = "msw_train,librispeech_train,jamendo_train,fma_train") -> dict:
    import json
    import os

    from fusion_embedding.paths import frames_dir

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    entries = {}
    for s in [x.strip() for x in shards.split(",") if x.strip()]:
        ip = str(frames_dir(s) / "index.json")
        if not os.path.exists(ip):
            entries[s] = {"status": "MISSING"}
            continue
        with open(ip) as fh:
            ix = json.load(fh)
        dom = ix.get("domain", ["?"])
        entries[s] = {"n_total": ix.get("n_total"), "shards": len(ix.get("shards", [])),
                      "domain": dom[0] if dom else "?",
                      "text_cache": bool(ix.get("text_cache", False)),
                      "caption_samples": ix.get("captions", [])[:3]}
    manifest = {"corpus": "speech_delta_v1", "entries": entries,
                "licenses": {"msw_train": "CC-BY-4.0 (MLCommons/ml_spoken_words, en)",
                             "librispeech_train": "CC-BY-4.0 (OpenSLR 12)",
                             "fma_train": "per-track CC-BY/CC-BY-SA/PD, NC+ND dropped",
                             "common_voice": "SKIPPED if MDC-gated (see ingest report)"},
                "recipe": "docs/research_speech_gap_solutions.md sec 6.2-6.3"}
    with open(f"{MANIFEST_DIR}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    volume.commit()
    print("MANIFEST:", json.dumps(manifest)[:1500], flush=True)
    return manifest
