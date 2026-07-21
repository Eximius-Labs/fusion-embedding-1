"""Run fusion-embedding-2 on the nine MAEB tasks from the FE1 submission.

Standalone Modal app: installs mteb from the local checkout (branch
model/add-fusion-embedding-2, which carries the FE2 ModelMeta), loads the
model at the pinned FE2 revision through mteb.get_model, and evaluates with
the mteb-format result cache rooted on the fusion-data Volume so the WHOLE
output folder (per-task TaskResult JSONs + model_meta.json) can be pulled
back intact with `modal volume get`.

FSD2019Kaggle stays excluded: 13.6% of its test clips sit in FSD50K dev,
which is in the training corpus (same reason as the FE1 submission).

Probe first, then the full run:
    PYTHONUTF8=1 uv run modal run --detach scripts/maeb_fe2_run.py::run_tasks --tasks BeijingOpera
    PYTHONUTF8=1 uv run modal run --detach scripts/maeb_fe2_run.py::run_tasks
Pull results:
    PYTHONUTF8=1 uv run modal volume get fusion-data maeb_fe2/results <local_dir>
"""

from __future__ import annotations

import modal

app = modal.App("fusion-maeb-fe2-v02")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchcodec==0.2.1",
        "numpy>=1.24",
        "transformers>=4.46",
        "accelerate>=0.30",
        "soundfile>=0.12",
        "librosa>=0.10",
        "pillow>=10.0",
        "av>=12",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir("D:/oss/.tmp/mteb-pr", "/root/mteb-src", copy=True,
                   ignore=["**/.git/**", "**/.venv/**", "**/tests/**",
                           "**/docs/**", "**/__pycache__/**"])
    .run_commands("pip install '/root/mteb-src[audio,image]'  # pin-9451b840")
)

hf_secret = modal.Secret.from_name("huggingface")
volume = modal.Volume.from_name("fusion-data")

MODEL = "EximiusLabs/fusion-embedding-2-2b-preview"
ALL_TASKS = [
    "BeijingOpera",
    "ClothoT2ARetrieval",
    "GTZANAudioReranking",
    "GTZANGenre",
    "MACST2ARetrieval",
    "RavdessZeroshot",
    "SpeechCommandsZeroshotv0.02",
    "UrbanSound8KT2ARetrieval",
    "VehicleSoundClustering",
]
CACHE_DIR = "/vol/maeb_fe2_v02/results_cache"
REVISION = "9451b840f0d1d95440f3836e9a7f600a833d663f"  # v0.2-preview


@app.function(gpu="L4", image=image, secrets=[hf_secret],
              volumes={"/vol": volume}, timeout=6 * 3600)
def run_tasks(tasks: str = "") -> dict:
    import json
    import os
    import time

    import mteb

    names = [t for t in tasks.split(",") if t] or ALL_TASKS
    t0 = time.time()
    assert REVISION, 'set REVISION to the v0.2 commit hash before deploying'
    # the baked registry may pin an older revision; override the meta at runtime
    meta = mteb.get_model_meta(MODEL)
    meta.revision = REVISION
    model = meta.load_model()
    cache = mteb.ResultCache(cache_path=CACHE_DIR)
    scores = {}
    # one task at a time, committing the Volume after each, so a killed run
    # loses at most the in-flight task
    for n in names:
        result = mteb.evaluate(model, tasks=[mteb.get_task(n)], cache=cache,
                               overwrite_strategy="only-missing",
                               show_progress_bar=True)
        volume.commit()
        for tr in result.task_results:
            d = tr.to_dict()
            split = next(iter(d["scores"]))
            scores[d["task_name"]] = d["scores"][split][0].get("main_score")
        print(f"TASK_DONE: {n} -> {scores.get(n)}  elapsed={round(time.time()-t0)}s",
              flush=True)
    payload = {"model": MODEL, "tasks": names,
               "runtime_s": round(time.time() - t0, 1),
               "main_scores": scores}

    # show what landed on the volume, for the pull step
    for root, _dirs, files in os.walk("/vol/maeb_fe2"):
        for f in files:
            payload.setdefault("files", []).append(
                os.path.join(root, f).replace("/vol/maeb_fe2/", ""))
    print("MAEB_FE2_RESULT:", json.dumps(payload))
    return payload
