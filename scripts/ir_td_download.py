"""Download the IR-TD early-access release (5-part tar.gz, ~8.4 GB) from the
authors' Google Drive folder straight to the fusion-data Volume, extract, and
report counts. Runs Modal-side so it does not compete with local bandwidth.

License: IR-TD is academic research use only; the thermal pack trained on it is
released under a research/non-commercial license, separate from the permissive
core (docs/research_thermal_corpus_solutions.md).

Deploy + spawn:
    PYTHONUTF8=1 uv run modal deploy scripts/ir_td_download.py
    uv run python -c "import modal; print(modal.Function.from_name('fusion-ir-td-download','fetch').spawn().object_id)"
"""
from __future__ import annotations

import modal

app = modal.App("fusion-ir-td-download")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("gdown")
)

volume = modal.Volume.from_name("fusion-data")

FOLDER = "https://drive.google.com/drive/folders/10AQ0nZ6V3mdRvTtXW3XbKKSGOs37dS0W"
DEST = "/vol/thermal/ir_td"


@app.function(image=image, volumes={"/vol": volume}, timeout=4 * 3600,
              memory=8192)
def fetch() -> dict:
    import glob
    import json
    import os
    import subprocess
    import tarfile

    os.makedirs(DEST, exist_ok=True)
    marker = os.path.join(DEST, "_EXTRACTED_OK")
    if os.path.exists(marker):
        print("IR_TD: already extracted, nothing to do")
        return json.load(open(marker))

    dl = os.path.join(DEST, "_parts")
    os.makedirs(dl, exist_ok=True)
    # gdown the whole folder (resumable at file granularity: skips existing)
    subprocess.run(["gdown", "--folder", FOLDER, "-O", dl, "--continue"],
                   check=True, timeout=3 * 3600)
    volume.commit()

    parts = sorted(glob.glob(os.path.join(dl, "**", "IR-TD-version*.tar.gz*"),
                             recursive=True))
    assert parts, f"no tarball parts found under {dl}"
    print(f"IR_TD: {len(parts)} part(s):", [os.path.basename(p) for p in parts],
          flush=True)

    # multi-part = split archive: concatenate then extract
    joined = os.path.join(DEST, "ir_td.tar.gz")
    with open(joined, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while chunk := f.read(1 << 24):
                    out.write(chunk)
    n_img, n_json = 0, 0
    with tarfile.open(joined, "r:gz") as tar:
        for i, m in enumerate(tar):
            tar.extract(m, DEST, filter="data")
            if m.name.lower().endswith((".jpg", ".jpeg", ".png")):
                n_img += 1
            elif m.name.endswith(".json"):
                n_json += 1
            if i % 5000 == 0:
                print(f"IR_TD extract: {i} entries ({n_img} images)", flush=True)
                volume.commit()
    os.remove(joined)

    caps = glob.glob(os.path.join(DEST, "**", "new_relative.json"),
                     recursive=True)
    result = {"images": n_img, "jsons": n_json,
              "captions_file": caps[0] if caps else None}
    if caps:
        recs = json.load(open(caps[0], encoding="utf-8"))
        result["caption_records"] = len(recs)
        sample = recs[0] if isinstance(recs, list) else next(iter(recs.items()))
        result["sample_keys"] = (sorted(sample.keys())
                                 if isinstance(sample, dict) else str(sample)[:120])
    json.dump(result, open(marker, "w"))
    volume.commit()
    print("IR_TD_DONE:", json.dumps(result), flush=True)
    return result
