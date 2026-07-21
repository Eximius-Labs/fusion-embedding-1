"""Create the PRIVATE demo Space and upload code + assets.

Reads the HF token from .env (never printed). Assets are pulled from the
fusion-data Volume to a local staging dir first (pull_assets.ps1 or the modal
CLI); this script uploads staging + code files, then attempts ZeroGPU
hardware and reports the outcome (falls back to cpu-basic silently).

Run:  PYTHONUTF8=1 uv run python scripts/demo_space/push_space.py <staging_dir>
"""
import os
import sys

from huggingface_hub import HfApi

REPO = "EximiusLabs/fusion-embedding-demo"
HERE = os.path.dirname(os.path.abspath(__file__))


def token() -> str:
    for line in open(os.path.join(HERE, "..", "..", ".env"), encoding="utf-8"):
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("HF_TOKEN not found in .env")


def main(staging: str) -> None:
    api = HfApi(token=token())
    api.create_repo(REPO, repo_type="space", space_sdk="gradio",
                    private=True, exist_ok=True)
    api.upload_folder(repo_id=REPO, repo_type="space", folder_path=staging,
                      commit_message="Gallery assets and indexes")
    for src, dst in [("app.py", "app.py"),
                     ("requirements.txt", "requirements.txt"),
                     ("README_space.md", "README.md")]:
        api.upload_file(repo_id=REPO, repo_type="space",
                        path_or_fileobj=os.path.join(HERE, src),
                        path_in_repo=dst,
                        commit_message=f"Add {dst}")
    hw = "unknown"
    try:
        api.request_space_hardware(REPO, "zero-a10g")
        hw = "zero-a10g requested/attached"
    except Exception as e:  # eligibility gate expected on free orgs
        hw = f"cpu-basic (zero-a10g rejected: {str(e)[:120]})"
    rt = api.get_space_runtime(REPO)
    print("SPACE:", f"https://huggingface.co/spaces/{REPO}")
    print("hardware:", hw, "| runtime stage:", rt.stage)


if __name__ == "__main__":
    main(sys.argv[1])
