"""Run fusion-embedding-1 on the MVEB video-suite tasks (mteb-format results).

Standalone Modal app on the maeb_fe2_run.py pattern: deploy + spawn (never
client-tied), mteb installed from the local checkout, per-task Volume commits
so a killed run loses at most the in-flight task, ResultCache rooted on the
fusion-data Volume so the complete mteb-format folder (TaskResult JSONs +
model_meta.json) is pulled back intact.

Scope note: this covers the MVEB(text, video, beta) and MVEB(video, beta)
task lists that fit the approved budget; the largest tasks (ActivityNet,
Panda70M, Kinetics400/600/700, VGGSoundV, OmniVideoBench, MusicAVQA and
RAVDESSAV pair classification) are enumerated in GROUPS["monsters"] and are
not launched by default.

Deploy + spawn:
    PYTHONUTF8=1 uv run modal deploy scripts/mveb_fe1_run.py
    uv run python -c "import modal; f = modal.Function.from_name('fusion-mveb-fe2', 'run_tasks'); print(f.spawn('probe').object_id)"
Pull results:
    PYTHONUTF8=1 uv run modal volume get fusion-data mveb_fe1/results_cache/results <local_dir>
"""

from __future__ import annotations

import modal

app = modal.App("fusion-mveb-fe2")

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
volume = modal.Volume.from_name("fusion-data")

MODEL = "EximiusLabs/fusion-embedding-2-2b-preview"
CACHE_DIR = "/vol/mveb_fe2_video/results_cache"

GROUPS = {
    "probe": ["MSVDV2TRetrieval"],
    "g1": ["AVMemeExamT2VRetrieval", "AudioCapsAVT2VRetrieval",
           "DiDeMoV2TRetrieval", "VALOR32KT2VRetrieval"],
    "g2": ["VATEXT2VRetrieval", "UCF101VideoZeroShotClassification",
           "MELDVideoZeroShot", "MELDVideoClassification"],
    "g3": ["HMDB51Classification", "HumanAnimalCartoonVPairClassification",
           "BreakfastClassification", "AVMemeVideoClassification",
           "RAVDESSVideoClustering"],
    "g4": ["WorldSenseVideoZeroShot", "WorldSenseVideoClassification"],
    # over the approved budget; enumerate only
    "monsters": ["ActivityNetCaptionsT2VRetrieval", "Panda70MT2VRetrieval",
                 "Kinetics700V", "Kinetics600V", "Kinetics400ZeroShot",
                 "VGGSoundV", "OmniVideoBenchVideoCentricQA",
                 "MusicAVQAVPairClassification",
                 "RAVDESSAVVPairClassification"],
}


@app.function(gpu="L4", image=image, secrets=[hf_secret],
              volumes={"/vol": volume}, timeout=20 * 3600)
def run_tasks(group: str = "probe") -> dict:
    import json
    import time

    import mteb

    names = GROUPS[group] if group in GROUPS else [t for t in group.split(",") if t]
    t0 = time.time()
    model = mteb.get_model(MODEL)
    cache = mteb.ResultCache(cache_path=CACHE_DIR)
    scores = {}
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
    payload = {"model": MODEL, "group": group, "tasks": names,
               "runtime_s": round(time.time() - t0, 1), "main_scores": scores}
    print("MVEB_FE1_RESULT:", json.dumps(payload), flush=True)
    return payload
