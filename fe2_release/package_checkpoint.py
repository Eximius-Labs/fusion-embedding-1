"""Package a training checkpoint into the HF release artifact.

Pulls the chosen ckpt + its result JSON from the Modal volume (via the modal CLI), strips
it to inference essentials, embeds the readout protocol + benchmark numbers, and writes
release/out/. Run: uv run --env-file .env python release/package_checkpoint.py --run-tag _a0native_384_800
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch

DEFAULT_SHARD = "audiocaps10k_sharded,fsd50k_train,wavcaps_audioset_sl_full"
OUT = os.path.join(os.path.dirname(__file__), "out")
DEFAULT_NAME = "fusion-embedding-2-2b-preview"
PROTOCOL = {
    "text_query_template": ("<|im_start|>system\n{instruction}<|im_end|>\n"
                            "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"),
    "image_doc_template": ("<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
                           "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
                           "<|im_start|>assistant\n"),
    "default_query_instruction": "Retrieve images or text relevant to the user's query.",
    "pooling": "last non-pad token",
    "cross_modal_readout": "per-modality mean-centering recommended (see inference.center)",
}


def _pull(remote: str, local: str) -> None:
    cmd = ["uv", "run", "--env-file", ".env", "modal", "volume", "get", "fusion-data",
           remote, local, "--force"]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONUTF8": "1"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="_a0native_384_800")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--shard", default=DEFAULT_SHARD)
    ap.add_argument("--version", default="0.1-preview")
    ap.add_argument("--name", default=DEFAULT_NAME,
                    help="artifact basename (fusion-embedding-1-... or -2-...)")
    args = ap.parse_args()
    global SHARD
    SHARD = args.shard
    os.makedirs(OUT, exist_ok=True)

    ckpt_name = f"p1frames_{SHARD}_step{args.steps}{args.run_tag}.pt"
    result_name = f"result_frames_{SHARD}_step{args.steps}{args.run_tag}.json"
    local_ckpt = os.path.join(OUT, "_raw.pt")
    local_result = os.path.join(OUT, "_result.json")
    _pull(f"checkpoints/{ckpt_name}", local_ckpt)
    _pull(f"checkpoints/{result_name}", local_result)

    ck = torch.load(local_ckpt, map_location="cpu", weights_only=False)
    result = json.load(open(local_result, encoding="utf-8"))
    # Prefer the standalone rescore (native-protocol eval) over the run's auto-score: for
    # native-target ckpts the in-run auto-816 may predate the native-eval fix (0.370 vs 0.626).
    audiocaps = result.get("score816")
    rescore_name = f"score816_audiocaps_test816__{SHARD}_step{args.steps}{args.run_tag}.json"
    local_rescore = os.path.join(OUT, "_rescore.json")
    try:
        _pull(f"checkpoints/{rescore_name}", local_rescore)
        audiocaps = json.load(open(local_rescore, encoding="utf-8"))
        print(f"using native rescore numbers from {rescore_name}")
    except subprocess.CalledProcessError:
        print("no standalone rescore found — using the run's auto-816 score")
    numbers = {"audiocaps_883_minrank5": audiocaps,
               "in_domain_eval": {k: result[k] for k in
                                  ("a2t_R@1", "a2t_R@10", "a2t_mAP@10") if k in result}}

    packaged = {
        "resampler": ck["resampler"],
        "text_whitening": ck["text_whitening"],
        "logit_scale": ck["logit_scale"],
        # fusion-embedding-2: modality-gated deep adapters ride along; loaders hard-fail
        # on presence mismatch so an adapter ckpt can never run unadapted.
        **({"adapters": ck["adapters"]} if "adapters" in ck else {}),
        "config": ck["config"],
        "base_4bit": ck.get("base_4bit", False),
        "protocol": PROTOCOL,
        "benchmarks": numbers,
        "source_run": {"ckpt": ckpt_name, "steps": args.steps, "run_tag": args.run_tag},
        "base_model": "Qwen/Qwen3-VL-Embedding-2B",
        "audio_model": "Qwen/Qwen2.5-Omni-7B",
        "version": args.version,
    }
    out_path = os.path.join(OUT, f"{args.name}.pt")
    torch.save(packaged, out_path)
    n_params = sum(v.numel() for v in ck["resampler"].values())
    n_params += sum(v.numel() for v in ck.get("adapters", {}).values())

    # safetensors export: the HF Hub param-count chip reads model.safetensors. Only the
    # TRAINED tensors ship (resampler + adapters + whitening + logit_scale); the frozen
    # towers are named in metadata.
    from safetensors.torch import save_file
    flat = {}
    for prefix, sd in (("resampler", ck["resampler"]),
                       ("audio_adapters", ck.get("adapters", {})),
                       ("text_whitening", ck["text_whitening"])):
        for k, v in sd.items():
            flat[f"{prefix}.{k}"] = v.contiguous()
    flat["logit_scale"] = torch.as_tensor(ck["logit_scale"]).float().reshape(1)
    save_file(flat, os.path.join(OUT, "model.safetensors"),
              metadata={"frozen_base": "Qwen/Qwen3-VL-Embedding-2B (not included; loaded at runtime)",
                        "frozen_audio_tower": "Qwen/Qwen2.5-Omni-7B audio tower (not included)",
                        "format": "pt"})
    print(f"safetensors: {sum(v.numel() for v in flat.values())/1e6:.1f}M params exported")
    # NOTE (v0.2 lesson): this config.json is the PACKAGER PAYLOAD only. The HF repo's
    # live config.json also carries auto_map/architectures and the model-construction
    # fields the trust_remote_code path requires — when pushing a weights update,
    # MERGE this payload into the repo's existing config.json (see the v0.2 fixup
    # 9451b840f0d1); replacing it wholesale breaks AutoModel loading.
    with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"model_type": "fusion-embedding-connector",
                   "adapter_rank": ck["config"].get("adapter_rank", 0),
                   "d_resampler": ck["config"].get("d_resampler"),
                   "n_query": ck["config"].get("n_query"),
                   "d_llm": ck["config"].get("d_llm"),
                   "mrl_dims": ck["config"].get("mrl_dims"),
                   "trained_params": n_params,
                   "protocol": PROTOCOL, "benchmarks": numbers}, fh, indent=2)
    os.remove(local_ckpt)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"packaged {out_path} ({size_mb:.1f} MB, {n_params/1e6:.1f}M trained params)")
    print("benchmarks embedded:", json.dumps(numbers, indent=2)[:400])


if __name__ == "__main__":
    sys.exit(main())
