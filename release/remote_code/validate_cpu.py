"""CPU validation for the remote-code files, run BEFORE anything is uploaded.

Checks, per repo (FE1, FE2):
  1. the modeling/configuration files import cleanly under a package context
     (mimicking transformers' dynamic-module loading of Hub repos);
  2. the extended config.json (existing fields + auto_map + remote-code fields)
     round-trips through FusionEmbeddingConfig;
  3. the repo's actual model.safetensors loads into the model with strict=True
     (exact key/shape agreement, no remapping);
  4. writes the extended config.json to out_fe{1,2}_config.json for upload.

Run: uv run --with python-dotenv --with huggingface_hub --with safetensors \
       --with "transformers>=4.46" python release/remote_code/validate_cpu.py
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

REPOS = {
    "fe1": "EximiusLabs/fusion-embedding-1-2b-preview",
    "fe2": "EximiusLabs/fusion-embedding-2-2b-preview",
}
REMOTE_FIELDS = {
    "d_audio": 3584,
    "resampler_depth": 6,
    "resampler_heads": 8,
    "resampler_ffn_mult": 4,
    "resampler_dropout": 0.0,
    "adapter_act": "silu",
    "mrl_default": 1024,
    "audio_pad_id": 151654,
    "eos_id": 151645,
    "pad_id": 151643,
    "audio_pad_token": "<|audio_pad|>",
    "base_model": "Qwen/Qwen3-VL-Embedding-2B",
    "audio_model": "Qwen/Qwen2.5-Omni-7B",
    "max_text_tokens": 512,
    "n_decoder_layers": 28,
    "architectures": ["FusionEmbeddingModel"],
    "auto_map": {
        "AutoConfig": "configuration_fusion_embedding.FusionEmbeddingConfig",
        "AutoModel": "modeling_fusion_embedding.FusionEmbeddingModel",
    },
}


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(HERE, "..", "..", ".env"))
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # -- 1. import under a synthetic package (relative imports must resolve) --
    pkg_dir = tempfile.mkdtemp(prefix="fe_remote_")
    pkg = os.path.join(pkg_dir, "fe_remote_pkg")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for fn in ("configuration_fusion_embedding.py", "modeling_fusion_embedding.py"):
        shutil.copy(os.path.join(HERE, fn), os.path.join(pkg, fn))
    sys.path.insert(0, pkg_dir)
    cfg_mod = importlib.import_module("fe_remote_pkg.configuration_fusion_embedding")
    mdl_mod = importlib.import_module("fe_remote_pkg.modeling_fusion_embedding")
    print("import OK")

    token = os.environ.get("HF_TOKEN")
    for key, repo in REPOS.items():
        cj_path = hf_hub_download(repo, "config.json", token=token)
        current = json.load(open(cj_path, encoding="utf-8"))
        extended = {**current, **REMOTE_FIELDS}
        # adapter_rank: present on FE2's config.json already; FE1 gets an explicit 0
        extended.setdefault("adapter_rank", 0)

        # -- 2. config round-trip --
        cfg = cfg_mod.FusionEmbeddingConfig(**{k: v for k, v in extended.items()
                                               if k not in ("architectures", "auto_map",
                                                            "model_type")})
        assert cfg.d_resampler == current["d_resampler"] == 384
        assert cfg.mrl_default == 1024 and cfg.audio_pad_id == 151654
        assert cfg.adapter_rank == (384 if key == "fe2" else 0)

        # -- 3. strict state-dict agreement with the shipped safetensors --
        model = mdl_mod.FusionEmbeddingModel(cfg)
        st_path = hf_hub_download(repo, "model.safetensors", token=token)
        sd = load_file(st_path)
        model.load_state_dict(sd, strict=True)
        n = sum(p.numel() for p in model.parameters())
        expected = current["trained_params"] + 1  # + logit_scale scalar
        assert n == expected, f"{key}: param count {n} != {expected}"
        # model state dict must contain nothing beyond the shipped keys
        extra = set(model.state_dict()) - set(sd)
        assert not extra, f"{key}: extra keys {sorted(extra)[:5]}"
        print(f"{key}: strict load OK ({n/1e6:.2f}M params, "
              f"{len(sd)} tensors, adapters={'yes' if cfg.adapter_rank else 'no'})")

        out = os.path.join(HERE, f"out_{key}_config.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(extended, fh, indent=2)
        print(f"{key}: wrote {out}")

    print("ALL CPU CHECKS PASSED")


if __name__ == "__main__":
    main()
