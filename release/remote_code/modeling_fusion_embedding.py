"""HF remote-code modeling for the fusion-embedding family (AutoModel + trust_remote_code).

One embedding space for text, images, and audio. The checkpoint on this repository holds
ONLY the trained components (perceiver-resampler connector, diagonal text whitening,
logit scale and — generation 2 — the modality-gated deep adapters); the frozen
Qwen3-VL-Embedding-2B base and the frozen Qwen2.5-Omni audio tower are downloaded from
their own repositories on first use and are byte-identical to their releases.

    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "EximiusLabs/fusion-embedding-1-2b-preview", trust_remote_code=True)
    t = model.embed_text("a dog barks in the distance")
    a = model.embed_audio("dog.wav")
    i = model.embed_image("dog.jpg")

The embed_* methods reproduce the repository's reference ``inference.py`` exactly (same
chat templates, truncation, pooling, whitening, Matryoshka truncation and normalization);
outputs are bitwise-identical to that loader on the same hardware. Non-audio inputs never
execute the generation-2 adapter branch (the gate returns the frozen layers' output
untouched), so text/image outputs are bit-for-bit those of generation 1 and of the base's
computation path.

Requires: transformers>=4.46 (with the Qwen2.5-Omni model classes), torchvision, pillow,
soundfile, librosa. A CUDA GPU is recommended (~14 GB at bf16).
"""

from __future__ import annotations

import math
import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_fusion_embedding import FusionEmbeddingConfig

DEFAULT_QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
DOC_INSTRUCTION = "Represent the user's input."

_ACTS = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}


# --------------------------------------------------------------------------- #
# helpers (mirrors of the training package, kept self-contained on purpose)
# --------------------------------------------------------------------------- #
def _chat(instruction: str, user_content: str) -> str:
    """The base's official embedding format: system-turn instruction, assistant opener."""
    return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def sinusoidal_positions(length: int, dim: int, device, dtype) -> torch.Tensor:
    if dim % 2 != 0:
        pe = sinusoidal_positions(length, dim + 1, device, dtype)
        return pe[:, :dim]
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, idx).squeeze(1)


def mrl_truncate_normalize(x: torch.Tensor, dim: int) -> torch.Tensor:
    return F.normalize(x[..., :dim], p=2, dim=-1)


class TextWhitening(nn.Module):
    """Diagonal (per-dim, MRL-safe) standardization of frozen text embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.uint8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(self.fitted) == 0:
            return x
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std


class _ResamplerBlock(nn.Module):
    """Pre-norm: latent self-attention -> cross-attention -> FFN."""

    def __init__(self, dim: int, heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, q, kv, key_padding_mask):
        h = self.norm_sa(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        q = q + self.cross_attn(h, kv_n, kv_n, key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        q = q + self.ffn(self.norm_ff(q))
        return q


class FusionResampler(nn.Module):
    """Perceiver-resampler: variable-length audio frames -> N fixed latent tokens."""

    def __init__(self, cfg: FusionEmbeddingConfig):
        super().__init__()
        dr = cfg.d_resampler
        self.in_proj = nn.Linear(cfg.d_audio, dr)
        self.queries = nn.Parameter(torch.empty(cfg.n_query, dr))
        nn.init.normal_(self.queries, std=0.02)
        self.blocks = nn.ModuleList(
            _ResamplerBlock(dr, cfg.resampler_heads, cfg.resampler_ffn_mult,
                            cfg.resampler_dropout)
            for _ in range(cfg.resampler_depth)
        )
        self.out_proj = nn.Linear(dr, cfg.d_llm)
        self.out_norm = nn.LayerNorm(cfg.d_llm)

    def forward(self, frames: torch.Tensor, frame_mask: Optional[torch.Tensor] = None):
        B, T, _ = frames.shape
        if frame_mask is None:
            frame_mask = torch.ones(B, T, dtype=torch.bool, device=frames.device)
        kv = self.in_proj(frames)
        kv = kv + sinusoidal_positions(T, kv.size(-1), kv.device, kv.dtype).unsqueeze(0)
        key_padding = ~frame_mask
        fully_masked = key_padding.all(dim=1)
        if fully_masked.any():
            key_padding = key_padding.clone()
            key_padding[fully_masked, 0] = False
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            q = block(q, kv, key_padding)
        return self.out_norm(self.out_proj(q))


class AdapterGate:
    """Depth-counted on/off switch shared by every adapter hook (generation 2)."""

    __slots__ = ("_depth",)

    def __init__(self) -> None:
        self._depth = 0

    @property
    def active(self) -> bool:
        return self._depth > 0

    def __enter__(self) -> "AdapterGate":
        self._depth += 1
        return self

    def __exit__(self, *exc) -> None:
        self._depth -= 1
        if self._depth < 0:
            raise RuntimeError("AdapterGate depth underflow — unbalanced enter/exit")


class GatedAdapter(nn.Module):
    """Parallel bottleneck adapter: ``h + up(act(down(LN(h))))``, computed in fp32."""

    def __init__(self, d_model: int, rank: int, act: str = "silu"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, rank, bias=False)
        self.act = _ACTS[act]()
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(h.float())))).to(h.dtype)


def _make_hook(adapter: GatedAdapter, gate: AdapterGate):
    def hook(_module, _inputs, output):
        if not gate.active:
            return None                                    # keep original output — bitwise no-op
        if isinstance(output, tuple):                      # HF decoder layers -> (hidden, ...)
            h = output[0]
            return (h + adapter(h),) + tuple(output[1:])
        return output + adapter(output)
    return hook


class OmniAudioAdapter(nn.Module):
    """Frozen Qwen2.5-Omni audio encoder -> (frames [B,T,d_audio], frame_mask [B,T])."""

    def __init__(self, encoder: nn.Module, d_audio: int):
        super().__init__()
        self.encoder = encoder
        self.d_audio = d_audio

    @torch.no_grad()
    def forward(self, mel: torch.Tensor, mel_mask: Optional[torch.Tensor] = None):
        B, n_mels, Fdim = mel.shape
        if mel_mask is None:
            feat_lens = torch.full((B,), Fdim, dtype=torch.long, device=mel.device)
        else:
            feat_lens = mel_mask.long().sum(dim=1)
        dtype = next(self.encoder.parameters()).dtype
        per_item = []
        for i in range(B):
            Li = max(int(feat_lens[i].item()), 1)
            feats = mel[i, :, :Li].to(dtype)
            out = self.encoder(input_features=feats,
                               feature_lens=torch.tensor([Li], device=mel.device))
            frames = out.last_hidden_state
            if frames.dim() == 3:
                frames = frames[0]
            per_item.append(frames.float())
        T_max = max(f.shape[0] for f in per_item)
        frames_out = mel.new_zeros(B, T_max, self.d_audio)
        frame_mask = torch.zeros(B, T_max, dtype=torch.bool, device=mel.device)
        for i, f in enumerate(per_item):
            frames_out[i, : f.shape[0]] = f
            frame_mask[i, : f.shape[0]] = True
        return frames_out, frame_mask


# --------------------------------------------------------------------------- #
# the AutoModel entry point
# --------------------------------------------------------------------------- #
class FusionEmbeddingModel(PreTrainedModel):
    """fusion-embedding for transformers AutoModel (trust_remote_code).

    The registered submodules are exactly the trained components shipped in this
    repository's ``model.safetensors`` (resampler + text whitening + logit scale
    + generation-2 adapters). The frozen base and audio tower load lazily from
    their own repositories on the first ``embed_*`` call, onto the device the
    trained components are on at that moment — call ``.to("cuda")`` (or pass
    ``device_map``) before embedding.
    """

    config_class = FusionEmbeddingConfig
    base_model_prefix = "fusion_embedding"
    main_input_name = "input_ids"
    _supports_flash_attn_2 = False

    def __init__(self, config: FusionEmbeddingConfig):
        super().__init__(config)
        self.resampler = FusionResampler(config)
        self.text_whitening = TextWhitening(config.d_llm)
        self.logit_scale = nn.Parameter(torch.zeros(1))
        self.audio_adapters: Optional[nn.ModuleList] = None
        if config.adapter_rank and config.adapter_rank > 0:
            self.audio_adapters = nn.ModuleList(
                GatedAdapter(config.d_llm, config.adapter_rank, config.adapter_act)
                for _ in range(config.n_decoder_layers)
            )
        # runtime-only state (plain dict: never in state_dict / parameters / save)
        self._rt: dict = {}
        self.post_init()

    def _init_weights(self, module):  # trained weights always come from the checkpoint
        pass

    # ------------------------------------------------------------- backbones
    @property
    def _device(self) -> torch.device:
        return self.resampler.out_proj.weight.device

    def _ensure_backbones(self) -> None:
        if "full" in self._rt:
            return
        from transformers import (AutoConfig, AutoFeatureExtractor, AutoModel,
                                  AutoProcessor)

        device, dtype = self._device, torch.bfloat16
        full = AutoModel.from_pretrained(self.config.base_model, trust_remote_code=True,
                                         dtype=dtype)
        full = full.to(device).eval()
        for p in full.parameters():
            p.requires_grad_(False)
        proc = AutoProcessor.from_pretrained(self.config.base_model, trust_remote_code=True)

        acfg = AutoConfig.from_pretrained(self.config.audio_model, trust_remote_code=True)
        audio_cfg = acfg.thinker_config.audio_config
        tower = self._load_audio_encoder(audio_cfg, dtype).to(device)
        fe_audio = AutoFeatureExtractor.from_pretrained(self.config.audio_model,
                                                        trust_remote_code=True)

        self._rt.update(full=full, proc=proc, tok=proc.tokenizer,
                        tower=OmniAudioAdapter(tower, self.config.d_audio),
                        fe_audio=fe_audio, gate=AdapterGate(), adapter_handles=[])

        if self.audio_adapters is not None:
            layers = self._find_decoder_layers(full.language_model)
            if len(layers) != len(self.audio_adapters):
                raise RuntimeError(
                    f"decoder has {len(layers)} layers but the checkpoint carries "
                    f"{len(self.audio_adapters)} adapters")
            gate = self._rt["gate"]
            self._rt["adapter_handles"] = [
                layer.register_forward_hook(_make_hook(ad, gate))
                for layer, ad in zip(layers, self.audio_adapters)
            ]

    def _load_audio_encoder(self, audio_cfg, dtype):
        """Instantiate the Omni audio encoder and load only ``thinker.audio_tower.*``."""
        import glob

        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as mod

        snap = snapshot_download(self.config.audio_model,
                                 allow_patterns=["*.safetensors", "*.json"])
        encoder = mod.Qwen2_5OmniAudioEncoder(audio_cfg)
        prefix = "thinker.audio_tower."
        collected = {}
        for shard in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
            for k, v in load_file(shard).items():
                if k.startswith(prefix):
                    collected[k[len(prefix):]] = v
        encoder.load_state_dict(collected, strict=False)
        return encoder.to(dtype).eval()

    @staticmethod
    def _find_decoder_layers(base_lm: nn.Module) -> nn.ModuleList:
        best = None
        for name, mod_ in base_lm.named_modules():
            if isinstance(mod_, nn.ModuleList) and name.rsplit(".", 1)[-1] == "layers":
                if best is None or len(mod_) > len(best):
                    best = mod_
        if best is None or len(best) == 0:
            raise ValueError("no decoder ModuleList named 'layers' found in the base")
        return best

    # ------------------------------------------------------------- internals
    def _finish(self, pooled: torch.Tensor, dim: Optional[int]) -> torch.Tensor:
        dim = dim or self.config.mrl_default
        return mrl_truncate_normalize(pooled.float(), dim).squeeze(0).cpu()

    def _encode_text_ids(self, ids_t: torch.Tensor) -> torch.Tensor:
        full = self._rt["full"]
        embeds = full.get_input_embeddings()(ids_t)
        out = full.language_model(inputs_embeds=embeds,
                                  attention_mask=torch.ones_like(ids_t))
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return last_token_pool(hidden, torch.ones_like(ids_t))

    # ------------------------------------------------------------- embedding
    @torch.no_grad()
    def embed_text(self, text: str, instruction: str = DEFAULT_QUERY_INSTRUCTION,
                   dim: Optional[int] = None) -> torch.Tensor:
        self._ensure_backbones()
        if self._rt["gate"].active:
            raise RuntimeError("adapter gate is open during a text encode — "
                               "non-audio inputs must run with the gate closed")
        ids = self._rt["tok"].encode(_chat(instruction, text),
                                     add_special_tokens=False)[: self.config.max_text_tokens]
        ids_t = torch.tensor([ids], device=self._device)
        pooled = self._encode_text_ids(ids_t)
        return self._finish(self.text_whitening(pooled), dim)

    @torch.no_grad()
    def embed_audio(self, audio: Union[str, "object"], sr: Optional[int] = None,
                    dim: Optional[int] = None) -> torch.Tensor:
        import librosa
        import soundfile as sf

        self._ensure_backbones()
        if isinstance(audio, (str, os.PathLike)):
            wav, sr = sf.read(str(audio), dtype="float32")
        else:
            wav = audio
            assert sr is not None, "pass sr= when embedding a raw array"
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        fe_audio = self._rt["fe_audio"]
        target_sr = fe_audio.sampling_rate
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        feats = fe_audio(wav, sampling_rate=target_sr, return_tensors="pt",
                         return_attention_mask=True, padding="max_length", truncation=True)
        mel = feats["input_features"][0]
        am = feats.get("attention_mask")
        if am is not None:
            mel = mel[:, : int(am[0].sum().item())]
        frames, frame_mask = self._rt["tower"](
            mel.unsqueeze(0).to(self._device),
            torch.ones(1, mel.shape[1], dtype=torch.bool, device=self._device))
        audio_tok = self.resampler(frames, frame_mask)
        cfg = self.config
        ids = torch.tensor([[cfg.audio_pad_id] * cfg.n_query + [cfg.eos_id]],
                           device=self._device)
        attention_mask = torch.ones_like(ids)
        full = self._rt["full"]
        embeds = full.get_input_embeddings()(ids).clone()
        embeds[ids == cfg.audio_pad_id] = (
            audio_tok.reshape(-1, audio_tok.size(-1)).to(embeds.dtype))
        with self._rt["gate"]:                      # adapters ON for audio (gen 2)
            out = full.language_model(inputs_embeds=embeds, attention_mask=attention_mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = last_token_pool(hidden, attention_mask)
        return self._finish(pooled, dim)

    @torch.no_grad()
    def embed_image(self, image, dim: Optional[int] = None) -> torch.Tensor:
        from PIL import Image

        self._ensure_backbones()
        if self._rt["gate"].active:
            # The vision path runs through the same (hook-carrying) decoder layers;
            # non-audio inputs must execute with the gate closed so the adapter
            # branch never runs.
            raise RuntimeError("adapter gate is open during an image embed — "
                               "non-audio inputs must run with the gate closed")
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(str(image))
        image = image.convert("RGB")
        text = _chat(DOC_INSTRUCTION, "<|vision_start|><|image_pad|><|vision_end|>")
        inputs = self._rt["proc"](text=[text], images=[image],
                                  return_tensors="pt").to(self._device)
        h = self._rt["full"](**inputs).last_hidden_state
        pooled = last_token_pool(h, inputs["attention_mask"])
        return self._finish(pooled, dim)

    # ------------------------------------------------------------- batched
    @torch.no_grad()
    def embed_text_batch(self, texts, instruction: str = DEFAULT_QUERY_INSTRUCTION,
                         dim: Optional[int] = None,
                         max_tokens: Optional[int] = None) -> torch.Tensor:
        """Batch text embedding [B, dim] (right-padded, mask-aware last-token pooling)."""
        self._ensure_backbones()
        if self._rt["gate"].active:
            raise RuntimeError("adapter gate is open during a text encode — "
                               "non-audio inputs must run with the gate closed")
        cfg, tok = self.config, self._rt["tok"]
        max_tokens = max_tokens or cfg.max_text_tokens
        seqs = [tok.encode(_chat(instruction, t), add_special_tokens=False)[:max_tokens]
                for t in texts]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), cfg.pad_id, dtype=torch.long, device=self._device)
        mask = torch.zeros(len(seqs), L, dtype=torch.long, device=self._device)
        for b, s in enumerate(seqs):
            ids[b, : len(s)] = torch.tensor(s, device=self._device)
            mask[b, : len(s)] = 1
        full = self._rt["full"]
        out = full.language_model(inputs_embeds=full.get_input_embeddings()(ids),
                                  attention_mask=mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = self.text_whitening(last_token_pool(hidden, mask))
        return mrl_truncate_normalize(pooled.float(), dim or cfg.mrl_default).cpu()

    @torch.no_grad()
    def embed_audio_batch(self, wavs, sr: int, dim: Optional[int] = None) -> torch.Tensor:
        """Batch audio embedding [B, dim] from raw waveform arrays at a common rate."""
        import librosa
        import numpy as np

        self._ensure_backbones()
        cfg, fe_audio = self.config, self._rt["fe_audio"]
        target_sr = fe_audio.sampling_rate
        prepped = []
        for wav in wavs:
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            prepped.append(wav)
        feats = fe_audio(prepped, sampling_rate=target_sr, return_tensors="pt",
                         return_attention_mask=True, padding="max_length", truncation=True)
        mel, am = feats["input_features"], feats.get("attention_mask")
        if am is not None:
            tmax = int(am.sum(dim=1).max().item())
            mel, am = mel[:, :, :tmax], am[:, :tmax]
        fmask = (am.bool() if am is not None
                 else torch.ones(mel.shape[0], mel.shape[2], dtype=torch.bool))
        frames, frame_mask = self._rt["tower"](mel.to(self._device),
                                               fmask.to(self._device))
        audio_tok = self.resampler(frames, frame_mask)
        ids = torch.tensor([[cfg.audio_pad_id] * cfg.n_query + [cfg.eos_id]] * mel.shape[0],
                           device=self._device)
        attention_mask = torch.ones_like(ids)
        full = self._rt["full"]
        embeds = full.get_input_embeddings()(ids).clone()
        embeds[ids == cfg.audio_pad_id] = (
            audio_tok.reshape(-1, audio_tok.size(-1)).to(embeds.dtype))
        with self._rt["gate"]:
            out = full.language_model(inputs_embeds=embeds, attention_mask=attention_mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = last_token_pool(hidden, attention_mask)
        return mrl_truncate_normalize(pooled.float(), dim or cfg.mrl_default).cpu()

    # ------------------------------------------------------------- read-out
    @staticmethod
    def center(embs: torch.Tensor) -> torch.Tensor:
        """Per-modality mean-centering + renormalization (cross-modal ranking readout)."""
        c = embs - embs.mean(dim=0, keepdim=True)
        return F.normalize(c, dim=-1)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "fusion-embedding is an embedding model: use embed_text(str), "
            "embed_audio(path_or_array, sr=...), embed_image(path_or_PIL), and "
            "center(embs) for cross-modal ranking.")
