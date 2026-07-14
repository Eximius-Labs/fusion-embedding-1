"""Run one real MVEB video task end-to-end with the staged mteb wrapper.

Standalone Modal app: installs mteb from the local PR checkout, loads the model
through mteb.get_model (Hub remote code at the pinned revision), and runs
MSVDT2VRetrieval (the smallest video task in the maintainer's video benchmarks:
0.64 GB download, 1,320 test rows) via mteb.evaluate. Prints the task's scores
as one JSON line for retrieval from the logs.

Run:
    PYTHONUTF8=1 uv run modal run --detach scripts/mveb_video_run.py::run_task
"""

from __future__ import annotations

import modal

app = modal.App("fusion-mveb-video")

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
    .run_commands("pip install '/root/mteb-src[audio,image]'")
)

hf_secret = modal.Secret.from_name("huggingface")

TASK = "MSVDT2VRetrieval"
MODEL = "EximiusLabs/fusion-embedding-1-2b-preview"


@app.function(gpu="L4", image=image, secrets=[hf_secret], timeout=3 * 3600)
def run_task() -> dict:
    import json
    import time

    import mteb

    t0 = time.time()
    model = mteb.get_model(MODEL)
    task = mteb.get_task(TASK)
    result = mteb.evaluate(model, tasks=[task], cache=None,
                           show_progress_bar=True)
    dt = time.time() - t0

    task_result = result.task_results[0]
    payload = {
        "task": TASK,
        "model": MODEL,
        "runtime_s": round(dt, 1),
        "scores": task_result.to_dict()["scores"],
    }
    print("MVEB_VIDEO_RESULT:", json.dumps(payload))
    return payload
