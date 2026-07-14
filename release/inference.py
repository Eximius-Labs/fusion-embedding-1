"""fusion-embedding-1 inference — one embedding space for text, images, and audio.

Loads the frozen Qwen3-VL-Embedding base (native paths for text and images), the frozen
Qwen2.5-Omni audio tower, and this repository's trained connector checkpoint. All inputs
use the base model's official chat-template format; embedding quality is sensitive to
this formatting, so use the templates provided here rather than constructing your own.

    from inference import FusionEmbedder
    fe = FusionEmbedder.from_pretrained("EximiusLabs/fusion-embedding-1-2b-preview")
    a, t, i = fe.embed_audio("dog.wav"), fe.embed_text("a dog barks"), fe.embed_image("dog.jpg")

Requires: fusion_embedding (pip install git+https://github.com/Eximius-Labs/fusion-embedding-1),
transformers>=4.46, torchvision, pillow, soundfile, librosa.
"""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import numpy as np

import torch

BASE_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
AUDIO_MODEL = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
DOC_INSTRUCTION = "Represent the user's input."
CKPT_FILE = "fusion-embedding-1-2b-preview.pt"


def _chat(instruction: str, user_content: str) -> str:
    """The base's official embedding format: system-turn instruction, assistant opener."""
    return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n")


class FusionEmbedder:
    def __init__(self, ckpt_path: str, device: str = "cuda", dtype=torch.bfloat16):
        from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

        from fusion_embedding.config import FusionConfig
        from fusion_embedding.hf_components import BaseLMAdapter, load_audio_tower
        from fusion_embedding.model import FusionEmbeddingModel, last_token_pool

        self.device = device
        self._pool = last_token_pool
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        flds = {f.name for f in dataclasses.fields(FusionConfig)}
        self.cfg = FusionConfig(**{k: v for k, v in ck["config"].items() if k in flds})

        self.full = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True, dtype=dtype)
        self.full = self.full.to(device).eval()
        for p in self.full.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        self.tok = self.proc.tokenizer

        tower, _, _ = load_audio_tower(AUDIO_MODEL, device=device, dtype=dtype)
        self.fe_audio = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL, trust_remote_code=True)

        self.model = FusionEmbeddingModel(self.cfg, self.full.get_input_embeddings(),
                                          BaseLMAdapter(self.full.language_model),
                                          audio_encoder=tower)
        self.model.resampler.to(device).float()
        self.model.resampler.load_state_dict(ck["resampler"])
        self.model.text_whitening.load_state_dict(ck["text_whitening"])   # identity if unfitted
        self.model.eval()

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, repo_or_path: str, device: str = "cuda",
                        revision: Optional[str] = None, **kw) -> "FusionEmbedder":
        """Load from a local checkpoint path or an HF repo. ``revision`` pins a repo
        tag/commit (e.g. ``"v0.1-preview"``, ``"v0.2-preview"``); default is latest."""
        if os.path.exists(repo_or_path):
            path = repo_or_path
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_or_path, CKPT_FILE, revision=revision)
        return cls(path, device=device, **kw)

    # ------------------------------------------------------------------ helpers
    def _finish(self, pooled: torch.Tensor, dim: Optional[int]) -> torch.Tensor:
        from fusion_embedding.model import mrl_truncate_normalize
        return mrl_truncate_normalize(pooled.float(), dim or self.cfg.mrl_default).squeeze(0).cpu()

    # ------------------------------------------------------------------ audio
    @torch.no_grad()
    def embed_audio(self, audio: Union[str, "np.ndarray"], sr: Optional[int] = None,
                    dim: Optional[int] = None) -> torch.Tensor:
        import librosa
        import soundfile as sf
        if isinstance(audio, (str, os.PathLike)):
            wav, sr = sf.read(str(audio), dtype="float32")
        else:
            wav = audio
            assert sr is not None, "pass sr= when embedding a raw array"
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        target_sr = self.fe_audio.sampling_rate
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        feats = self.fe_audio(wav, sampling_rate=target_sr, return_tensors="pt",
                              return_attention_mask=True, padding="max_length", truncation=True)
        mel = feats["input_features"][0]
        am = feats.get("attention_mask")
        if am is not None:
            mel = mel[:, : int(am[0].sum().item())]
        audio_tok = self.model.audio_tokens(
            mel.unsqueeze(0).to(self.device),
            torch.ones(1, mel.shape[1], dtype=torch.bool, device=self.device))
        ids = torch.tensor([[self.cfg.audio_pad_id] * self.cfg.n_query + [self.cfg.eos_id]],
                           device=self.device)
        pooled = self.model.encode_audio(ids, torch.ones_like(ids), audio_tok)
        return self._finish(pooled, dim)

    # ------------------------------------------------------------------ text
    @torch.no_grad()
    def embed_text(self, text: str, instruction: str = DEFAULT_QUERY_INSTRUCTION,
                   dim: Optional[int] = None) -> torch.Tensor:
        ids = self.tok.encode(_chat(instruction, text), add_special_tokens=False)[:512]
        ids_t = torch.tensor([ids], device=self.device)
        pooled = self.model.encode_text(ids_t, torch.ones_like(ids_t))
        return self._finish(self.model.text_whitening(pooled), dim)

    # ------------------------------------------------------------------ image
    @torch.no_grad()
    def embed_image(self, image, dim: Optional[int] = None) -> torch.Tensor:
        from PIL import Image
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(str(image))
        image = image.convert("RGB")
        text = _chat(DOC_INSTRUCTION, "<|vision_start|><|image_pad|><|vision_end|>")
        inputs = self.proc(text=[text], images=[image], return_tensors="pt").to(self.device)
        h = self.full(**inputs).last_hidden_state
        pooled = self._pool(h, inputs["attention_mask"])
        return self._finish(pooled, dim)

    # ------------------------------------------------------------------ video
    @torch.no_grad()
    def embed_video(self, video, fps: Optional[float] = None,
                    max_frames: Optional[int] = None,
                    dim: Optional[int] = None) -> torch.Tensor:
        """Embed a video through the frozen base model's own video path.

        ``video`` is a file path/URL, or a pre-extracted frame sequence (a list of
        PIL images and/or frame-image paths). Preprocessing follows the base model's
        official usage (the Qwen3-VL-Embedding reference scripts):
        ``qwen_vl_utils.process_vision_info`` with ``image_patch_size=16``, 1 fps
        sampling up to 64 frames for path inputs, uniform temporal sampling of frame
        sequences, a 7,864,320 total-pixel budget, and ``do_resize=False`` at the
        processor because the vision utility already smart-resizes. Like images,
        video is a non-audio input and takes the frozen path (no whitening).
        Requires qwen-vl-utils>=0.0.14; path inputs additionally need a video
        decoder backend supported by it.
        """
        import numpy as np

        gate = getattr(self.model, "_adapter_gate", None)
        if gate is not None and gate.active:
            # The video path runs through the same (hook-carrying) decoder layers;
            # non-audio inputs must run with the gate closed so the adapter branch
            # never runs. Mirrors the encode_text/embed_image guards.
            raise RuntimeError("adapter gate is open during a video embed — "
                               "non-audio inputs must run with the gate closed")
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "video embedding uses the base model's own preprocessing "
                "package: pip install 'qwen-vl-utils>=0.0.14'") from e

        # Constants from the base's reference implementation.
        video_total_pixels = 10 * 768 * 32 * 32
        default_fps, default_max_frames = 1.0, 64

        if isinstance(video, (str, os.PathLike)):
            v = str(video)
            content = v if v.startswith(("http://", "https://")) \
                else "file://" + os.path.abspath(v)
            vkw = {"fps": fps or default_fps,
                   "max_frames": max_frames or default_max_frames,
                   "total_pixels": video_total_pixels}
        else:
            frames = list(video)
            if not frames:
                raise ValueError("empty frame sequence")
            mf = max_frames or default_max_frames
            if len(frames) > mf:
                # Uniform temporal sampling, as in the base's sample_frames.
                idx = np.linspace(0, len(frames) - 1, mf, dtype=int)
                frames = [frames[i] for i in idx]
            content = ["file://" + os.path.abspath(str(f))
                       if isinstance(f, (str, os.PathLike)) else f
                       for f in frames]
            vkw = {"total_pixels": video_total_pixels}

        conversation = [{"role": "user",
                         "content": [{"type": "video", "video": content, **vkw}]}]
        _, video_inputs, video_kwargs = process_vision_info(
            conversation, image_patch_size=16,
            return_video_metadata=True, return_video_kwargs=True)
        videos, video_metadata = zip(*video_inputs)
        text = _chat(DOC_INSTRUCTION, "<|vision_start|><|video_pad|><|vision_end|>")
        inputs = self.proc(text=[text], videos=list(videos),
                           video_metadata=list(video_metadata),
                           do_resize=False, return_tensors="pt",
                           **video_kwargs).to(self.device)
        h = self.full(**inputs).last_hidden_state
        pooled = self._pool(h, inputs["attention_mask"])
        return self._finish(pooled, dim)

    # ------------------------------------------------------------------ cross-modal readout
    @staticmethod
    def center(embs: torch.Tensor) -> torch.Tensor:
        """Per-modality mean-centering followed by renormalization. Recommended when ranking
        a gallery of one modality against queries of another; improves cross-modal R@1 by
        roughly two points across modality pairs in our evaluation."""
        c = embs - embs.mean(dim=0, keepdim=True)
        return torch.nn.functional.normalize(c, dim=-1)
