"""MTEB/MAEB wrapper for fusion-embedding-1 — implements the mteb audio encoder interface.

Wraps the released checkpoint (connector + frozen towers) behind
``get_audio_embeddings`` / ``get_text_embeddings`` so ``mteb`` can run MAEB tasks.
Protocol notes (must match training/eval exactly, see the model card):
  - text goes through the base's native chat template with the sound instruction,
    last-token pooled, then diagonal whitening and MRL truncate+renormalize;
  - audio goes tower -> resampler -> <|vision_pad|> splice -> base -> last-token pool
    -> MRL truncate+renormalize (no whitening on the audio side);
  - checkpoint precision is bf16 (load_in_4bit must stay False — the -5.6 trap).

Run via ``modal_app.py::maeb_eval`` or locally with a CUDA GPU (~14 GB for bf16).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

SAMPLING_RATE = 16_000
INSTRUCTION = "Retrieve images or text relevant to the user's query."  # sound task default
MAX_TEXT_TOKENS = 254


class FusionEmbeddingMTEB:
    """mteb audio-encoder interface over a loaded FusionEmbeddingModel."""

    def __init__(self, model, cfg, feature_extractor, tokenizer, device: str = "cuda",
                 dim: int = 0, audio_batch: int = 8, text_batch: int = 64):
        self.model = model
        self.cfg = cfg
        self.fe = feature_extractor
        self.tok = tokenizer
        self.device = device
        self.dim = int(dim) or cfg.mrl_default
        self.audio_batch = audio_batch
        self.text_batch = text_batch

    # ---- mteb interface -------------------------------------------------------
    def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
               prompt_type=None, **kwargs: Any) -> np.ndarray:
        """mteb EncoderProtocol entry point — dispatch by the batch's modality columns
        (mirrors the ClapZeroShotWrapper dispatcher; audio+text fusion unsupported)."""
        feats = inputs.dataset.features
        has_text, has_audio = "text" in feats, "audio" in feats
        if has_text and has_audio:
            raise NotImplementedError("fusion-embedding-1 encodes one modality per input")
        if has_audio:
            return self.get_audio_embeddings(inputs, **kwargs)
        if has_text:
            return self.get_text_embeddings(inputs, **kwargs)
        raise ValueError(f"no supported modality in {list(feats)}")

    def get_audio_embeddings(self, inputs, show_progress_bar: bool = True,
                             **kwargs: Any) -> np.ndarray:
        from mteb.models.modality_collators import AudioCollator

        from fusion_embedding.model import mrl_truncate_normalize

        inputs.collate_fn = AudioCollator(target_sampling_rate=SAMPLING_RATE)
        out = []
        self.model.eval()
        buf: list = []

        def _flush():
            if not buf:
                return
            feats = self.fe([w.astype(np.float32) for w in buf], sampling_rate=SAMPLING_RATE,
                            return_tensors="pt", return_attention_mask=True,
                            padding="max_length", truncation=True)
            mel = feats["input_features"]                          # [B, n_mels, T] (30s-padded)
            am = feats.get("attention_mask")                       # [B, T] 1s on real frames
            if am is not None:                                     # trim to batch max real length
                tmax = int(am.sum(dim=1).max().item())
                mel, am = mel[:, :, :tmax], am[:, :tmax]
            with torch.no_grad():
                fmask = (am.bool() if am is not None
                         else torch.ones(mel.shape[0], mel.shape[2], dtype=torch.bool))
                audio_tok = self.model.audio_tokens(mel.to(self.device), fmask.to(self.device))
                ids = torch.tensor([[self.cfg.audio_pad_id] * self.cfg.n_query
                                    + [self.cfg.eos_id]] * mel.shape[0], device=self.device)
                pooled = self.model.encode_audio(ids, torch.ones_like(ids), audio_tok)
                out.append(mrl_truncate_normalize(pooled, self.dim).float().cpu().numpy())
            buf.clear()

        for batch in inputs:
            for audio in batch["audio"]:
                arr = audio["array"] if isinstance(audio, dict) else audio
                sr = audio.get("sampling_rate", SAMPLING_RATE) if isinstance(audio, dict) else SAMPLING_RATE
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                if sr != SAMPLING_RATE:
                    import librosa
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLING_RATE)
                buf.append(arr)
                if len(buf) >= self.audio_batch:
                    _flush()
        _flush()
        return np.vstack(out)

    def get_text_embeddings(self, inputs, show_progress_bar: bool = True,
                            **kwargs: Any) -> np.ndarray:
        from fusion_embedding.model import mrl_truncate_normalize

        texts: list[str] = []
        for batch in inputs:
            texts.extend(batch["text"])
        out = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.text_batch):
                chunk = texts[i:i + self.text_batch]
                bodies = [f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
                          f"<|im_start|>user\n{c}<|im_end|>\n<|im_start|>assistant\n"
                          for c in chunk]
                seqs = [self.tok.encode(b, add_special_tokens=False)[:MAX_TEXT_TOKENS]
                        for b in bodies]
                L = max(len(s) for s in seqs)
                ids = torch.full((len(seqs), L), self.cfg.pad_id, dtype=torch.long)
                mask = torch.zeros(len(seqs), L, dtype=torch.long)
                for b, s in enumerate(seqs):
                    ids[b, : len(s)] = torch.tensor(s)
                    mask[b, : len(s)] = 1
                raw = self.model.encode_text(ids.to(self.device), mask.to(self.device))
                emb = mrl_truncate_normalize(self.model.text_whitening(raw), self.dim)
                out.append(emb.float().cpu().numpy())
        return np.vstack(out)

    # Fused/interleaved inputs are not supported by this architecture (audio OR text).
    def get_fused_embeddings(self, *args, **kwargs):
        raise NotImplementedError("fusion-embedding-1 encodes one modality per input")

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b.T

    def similarity_pairwise(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (np.asarray(a) * np.asarray(b)).sum(axis=-1)


def build_model_meta(revision: str = "v0.2-preview"):
    """ModelMeta for mteb — declares modalities/params; also the leaderboard-PR entry."""
    from mteb.models import ModelMeta

    return ModelMeta(
        loader=None,
        name="EximiusLabs/fusion-embedding-1-2b-preview",
        languages=["eng-Latn"],
        revision=revision,
        release_date="2026-07-06",
        modalities=["audio", "text"],
        n_parameters=2_200_000_000,
        n_embedding_parameters=16_400_000,
        memory_usage_mb=14000,
        max_tokens=254,
        embed_dim=2048,
        license="cc-by-nc-4.0",
        open_weights=True,
        public_training_code="https://github.com/Eximius-Labs/fusion-embedding",
        public_training_data=None,
        framework=["PyTorch"],
        reference="https://huggingface.co/EximiusLabs/fusion-embedding-1-2b-preview",
        similarity_fn_name="cosine",
        use_instructions=True,
        training_datasets=set(),
    )


def load_for_mteb(ckpt_name: str, device: str = "cuda", dim: int = 0):
    """Build FusionEmbeddingMTEB from a training checkpoint on the Volume (Modal-side)."""
    import dataclasses
    import json

    from fusion_embedding.config import FusionConfig
    from fusion_embedding.hf_components import load_audio_tower, load_base
    from fusion_embedding.model import FusionEmbeddingModel
    from fusion_embedding.paths import checkpoints_dir

    ckpt = torch.load(str(checkpoints_dir() / ckpt_name), map_location=device)
    flds = {f.name for f in dataclasses.fields(FusionConfig)}
    saved = {k: v for k, v in ckpt.get("config", {}).items() if k in flds}
    cfg0 = FusionConfig(**saved)
    cfg, embed_tokens, base_lm, tokenizer = load_base(
        cfg0, device=device, dtype=torch.bfloat16,
        load_in_4bit=bool(ckpt.get("base_4bit", False)), d_audio=cfg0.d_audio)
    tower, fe, _ = load_audio_tower(device=device, dtype=torch.bfloat16)
    model = FusionEmbeddingModel(cfg, embed_tokens, base_lm, audio_encoder=tower)
    model.resampler.to(device).float()
    model.resampler.load_state_dict(ckpt["resampler"])
    # fusion-embedding-2: adapter presence must match, and the weights must load —
    # an adapter ckpt scored without them silently evaluates the unadapted model.
    if ("adapters" in ckpt) != (model.audio_adapters is not None):
        raise RuntimeError(f"adapter presence mismatch: ckpt has_adapters={'adapters' in ckpt} "
                           f"but config adapter_rank={cfg.adapter_rank}")
    if model.audio_adapters is not None:
        model.audio_adapters.to(device).float()
        model.audio_adapters.load_state_dict(ckpt["adapters"])
    if isinstance(model.logit_scale, torch.nn.Parameter):
        model.logit_scale.data = ckpt["logit_scale"].to(device)
    if "text_whitening" in ckpt:
        model.text_whitening.load_state_dict(ckpt["text_whitening"])
    hf_tok = getattr(tokenizer, "hf", tokenizer)
    wrapper = FusionEmbeddingMTEB(model, cfg, fe, hf_tok, device=device, dim=dim)
    try:
        wrapper.mteb_model_meta = build_model_meta()
    except Exception as e:                                         # noqa: BLE001
        print(f"ModelMeta attach skipped: {e}")
    return wrapper
