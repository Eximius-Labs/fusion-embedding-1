"""Real frozen-Qwen wiring — the HLD §10 ``load_components`` seam, implemented.

Discovered against the live models (Modal CPU introspection):

Base — ``Qwen/Qwen3-VL-Embedding-2B`` -> ``Qwen3VLModel``
  * children ``[visual, language_model]``; ``language_model.embed_tokens`` is the
    input embedding; ``forward`` accepts ``inputs_embeds``.
  * ``text_config.hidden_size = 2048`` (= d_llm). EOS = ``<|im_end|>`` id 151645,
    pad id 151643. No ``<|audio_pad|>`` -> we add it (its row is always overwritten
    by the resampler, so its value is irrelevant and the base's original rows stay
    byte-identical).

Audio — ``Qwen/Qwen2.5-Omni-7B`` audio tower -> ``Qwen2_5OmniAudioEncoder``
  * ``forward(input_features, feature_lens, ...) -> BaseModelOutputWithPooling`` whose
    ``last_hidden_state`` is **packed** ``[sum_i T_i, output_dim]`` (varlen, FlashAttn
    style), NOT ``[B,T,D]``. ``output_dim = 3584`` (post stride-2 pool + projection;
    ~25 fps / ~750 frames per 30 s). num_mel_bins 128. FE = WhisperFeatureExtractor.

  ==> the exposed frame dim is **3584**, not the 1280 the HLD assumed (1280 is the
      encoder's internal d_model). We set ``cfg.d_audio`` from the real config so the
      FusionResampler's ``in_proj`` is sized correctly (3584 -> d_resampler).

The three callables returned here satisfy the exact contract the model + tiny
stand-ins already use, so ``FusionEmbeddingModel`` runs unchanged on the real base.
"""

from __future__ import annotations

import glob
import os
from dataclasses import replace
from typing import Optional

import torch
import torch.nn as nn

from .config import FusionConfig

AUDIO_PAD_TOKEN = "<|audio_pad|>"


def _hf_token() -> Optional[str]:
    """The HF token from the environment (Modal injects it from the `huggingface` secret).

    Returns None (not "") for missing/empty so callers pass token=None for anonymous access
    rather than an empty Bearer header.
    """
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return tok or None


# --------------------------------------------------------------------------- #
# Audio-tower adapter: packed varlen output -> [B, T, d_audio] + frame_mask
# --------------------------------------------------------------------------- #
class OmniAudioAdapter(nn.Module):
    """Wrap the frozen Qwen2.5-Omni audio encoder to the (mel, mel_mask) -> (frames, mask) contract.

    Correctness-first: processes one clip at a time so we never have to reverse-engineer
    the encoder's internal chunk/pool length arithmetic to un-pack its varlen output —
    for B=1 the packed ``last_hidden_state`` IS that clip's ``[T_i, d_audio]`` frames.
    The encoder is frozen and runs under ``no_grad``; batch-packing is a later optimization.
    """

    def __init__(self, encoder: nn.Module, d_audio: int):
        super().__init__()
        self.encoder = encoder
        self.d_audio = d_audio

    @torch.no_grad()
    def forward(self, mel: torch.Tensor, mel_mask: Optional[torch.Tensor] = None):
        # mel: [B, n_mels, F]; mel_mask: [B, F] (1=valid). feature_lens = valid mel frames.
        B, n_mels, F = mel.shape
        if mel_mask is None:
            feat_lens = torch.full((B,), F, dtype=torch.long, device=mel.device)
        else:
            feat_lens = mel_mask.long().sum(dim=1)
        dtype = next(self.encoder.parameters()).dtype

        per_item = []
        for i in range(B):
            Li = int(feat_lens[i].item())
            Li = max(Li, 1)
            # The Omni encoder wants packed 2D mel [n_mels, frames] + feature_lens; it does
            # its own chunk/pad to [n_chunks, n_mels, n_window] internally. Passing 3D here
            # makes that 4D -> conv1d fails. So feed one clip as 2D.
            feats = mel[i, :, :Li].to(dtype)                         # [n_mels, Li]
            out = self.encoder(
                input_features=feats,
                feature_lens=torch.tensor([Li], device=mel.device),
            )
            frames = out.last_hidden_state                          # [Ti, d_audio] (packed, B=1)
            if frames.dim() == 3:                                   # tolerate [1, Ti, d] variants
                frames = frames[0]
            per_item.append(frames.float())

        T_max = max(f.shape[0] for f in per_item)
        frames_out = mel.new_zeros(B, T_max, self.d_audio)
        frame_mask = torch.zeros(B, T_max, dtype=torch.bool, device=mel.device)
        for i, f in enumerate(per_item):
            t = f.shape[0]
            frames_out[i, :t] = f
            frame_mask[i, :t] = True
        return frames_out, frame_mask


# --------------------------------------------------------------------------- #
# base_lm adapter: drive only the language_model with inputs_embeds -> hidden
# --------------------------------------------------------------------------- #
class BaseLMAdapter(nn.Module):
    """(inputs_embeds, attention_mask) -> last_hidden_state [B,S,d_llm] via the frozen text LM."""

    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.language_model = language_model

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


# --------------------------------------------------------------------------- #
# Weight loading: pull ONLY the audio tower out of the 7B checkpoint
# --------------------------------------------------------------------------- #
def _load_audio_encoder(audio_model: str, audio_cfg, dtype, hf_home: Optional[str]):
    """Instantiate Qwen2_5OmniAudioEncoder and load just the `thinker.audio_tower.*`
    weights from the sharded safetensors — avoids materializing the full 7B model."""
    from safetensors.torch import load_file
    from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as mod
    from huggingface_hub import snapshot_download

    snap = snapshot_download(audio_model, cache_dir=hf_home, token=_hf_token())
    encoder = mod.Qwen2_5OmniAudioEncoder(audio_cfg)

    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    prefix = "thinker.audio_tower."
    collected: dict[str, torch.Tensor] = {}
    for shard in shards:
        sd = load_file(shard)
        for k, v in sd.items():
            if k.startswith(prefix):
                collected[k[len(prefix):]] = v
    missing, unexpected = encoder.load_state_dict(collected, strict=False)
    # informative, non-fatal: some buffers (e.g. positional) may be registered, not stored
    print(f"[audio] loaded {len(collected)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    return encoder.to(dtype).eval()


# --------------------------------------------------------------------------- #
# The seam itself
# --------------------------------------------------------------------------- #
def load_components(
    cfg: FusionConfig,
    base_model: str = "Qwen/Qwen3-VL-Embedding-2B",
    audio_model: str = "Qwen/Qwen2.5-Omni-7B",
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = True,
    gradient_checkpointing: bool = True,
):
    """Load the frozen Qwen base + Omni audio tower; return everything to build the model.

    Returns ``(cfg, embed_tokens, base_lm, audio_encoder, tokenizer, feature_extractor)``
    where cfg has resolved ``d_audio`` (3584), ``d_llm`` (2048), ``audio_pad_id``,
    ``eos_id``, ``pad_id``. Hand the three callables straight to ``FusionEmbeddingModel``.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoFeatureExtractor

    hf_home = os.environ.get("HF_HOME")
    token = _hf_token()
    print(f"[hf] using HF token: {'yes (' + token[:6] + '…)' if token else 'NO — set the huggingface secret'}")

    # --- base: Qwen3VLModel ---
    quant = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModel.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=quant,
        device_map={"": device} if quant is not None else None,
        token=token,
    )
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, token=token)
    # Audio placeholder = an existing INERT special token used purely as a position marker.
    # We drive language_model with inputs_embeds and OVERWRITE these positions with audio
    # tokens, so the marker is never semantically processed (the vision path is bypassed —
    # no pixel_values). Reusing a token avoids resizing embeddings (fragile under 4-bit) and
    # keeps the base byte-identical — the strongest form of the MMEB-V2 regression guard.
    audio_pad_id = None
    for cand in ("<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"):
        tid = tokenizer.convert_tokens_to_ids(cand)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            audio_pad_id = tid
            print(f"[base] audio placeholder marker = {cand} (id {tid}), reused inert; base unmodified")
            break
    if audio_pad_id is None:                                        # fallback: add a token + resize
        tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_PAD_TOKEN]})
        base.resize_token_embeddings(len(tokenizer))
        audio_pad_id = tokenizer.convert_tokens_to_ids(AUDIO_PAD_TOKEN)

    language_model = base.language_model
    if gradient_checkpointing and hasattr(language_model, "gradient_checkpointing_enable"):
        # weights are frozen, but we backprop activations THROUGH the LM to reach the
        # connector — checkpointing trades compute for a large activation-memory cut (8GB).
        language_model.gradient_checkpointing_enable()
    embed_tokens = base.get_input_embeddings()
    base_lm = BaseLMAdapter(language_model)

    d_llm = base.config.text_config.hidden_size

    # --- audio: Qwen2_5OmniAudioEncoder (audio tower only) ---
    acfg = AutoConfig.from_pretrained(audio_model, trust_remote_code=True, token=token)
    audio_cfg = acfg.thinker_config.audio_config
    d_audio = audio_cfg.output_dim                                  # 3584 (post-projection)
    encoder = _load_audio_encoder(audio_model, audio_cfg, dtype, hf_home).to(device)
    audio_encoder = OmniAudioAdapter(encoder, d_audio=d_audio)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        audio_model, trust_remote_code=True, token=token
    )

    # --- resolve cfg to the real dims/ids ---
    cfg = replace(
        cfg,
        d_llm=d_llm,
        d_audio=d_audio,
        audio_pad_id=int(audio_pad_id),
        eos_id=int(tokenizer.eos_token_id),
        pad_id=int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0),
    )
    cfg._validate()
    return cfg, embed_tokens, base_lm, audio_encoder, tokenizer, feature_extractor
