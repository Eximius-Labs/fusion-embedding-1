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
import math
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


# --------------------------------------------------------------------------- #
# video preprocessing (native)
#
# Faithful reimplementation of the base model's reference video preprocessing
# (the Qwen3-VL-Embedding scripts' vision pipeline at image_patch_size=16), so
# no extra vision package is needed: frame selection, the per-frame image
# resize applied to frame-sequence inputs, the per-video smart resize under the
# total-pixel budget, and the processor kwargs (do_resize=False,
# do_sample_frames=False, video_metadata) match the reference exactly; outputs
# are verified bitwise-equal against the reference implementation on identical
# inputs. Decoded-video inputs (frame tensors, file paths via torchcodec) take
# the reference path-input treatment: a single per-video resize, no per-frame
# image resize.
# --------------------------------------------------------------------------- #
_V_PATCH_FACTOR = 32                       # image_patch_size 16 x spatial merge 2
_V_FRAME_FACTOR = 2
_V_DEFAULT_FPS = 1.0
_V_DEFAULT_MAX_FRAMES = 64
_V_MIN_PIXELS = 128 * _V_PATCH_FACTOR ** 2           # per-frame floor
_V_MAX_PIXELS = 768 * _V_PATCH_FACTOR ** 2           # per-frame ceiling
_V_TOTAL_PIXELS = 10 * _V_MAX_PIXELS                 # per-video budget
_V_IMG_MIN_PIXELS = 4 * _V_PATCH_FACTOR ** 2         # per-frame image defaults
_V_IMG_MAX_PIXELS = 16384 * _V_PATCH_FACTOR ** 2
_V_FPS_MIN_FRAMES = 4
_V_MAX_RATIO = 200


def _v_round(n: float, f: int) -> int:
    return round(n / f) * f


def _v_ceil(n: float, f: int) -> int:
    return math.ceil(n / f) * f


def _v_floor(n: float, f: int) -> int:
    return math.floor(n / f) * f


def _v_smart_resize(height: int, width: int, factor: int,
                    min_pixels: int, max_pixels: int):
    if max(height, width) / min(height, width) > _V_MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {_V_MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}")
    h_bar = max(factor, _v_round(height, factor))
    w_bar = max(factor, _v_round(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _v_floor(height / beta, factor)
        w_bar = _v_floor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _v_ceil(height * beta, factor)
        w_bar = _v_ceil(width * beta, factor)
    return h_bar, w_bar


def _v_frame_to_image(frame):
    """Frame-sequence element -> resized RGB PIL image (reference fetch_image)."""
    from PIL import Image

    if isinstance(frame, (str, os.PathLike)):
        image = Image.open(str(frame))
    else:
        image = frame
    if image.mode == "RGBA":
        white = Image.new("RGB", image.size, (255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")
    width, height = image.size
    rh, rw = _v_smart_resize(height, width, _V_PATCH_FACTOR,
                             _V_IMG_MIN_PIXELS, _V_IMG_MAX_PIXELS)
    return image.resize((rw, rh))


def _v_prepare(video, fps, max_frames):
    """Normalize any supported video input to (uint8 tensor [T,C,H,W], metadata).

    Frame sequences follow the reference list-input treatment (per-frame image
    resize, pad to an even count by repeating the last frame, synthetic
    metadata at 2 fps). Frame tensors and file paths follow the reference
    decoded-video treatment (frame selection only; single per-video resize).
    """
    import numpy as np

    if isinstance(video, torch.Tensor):
        if video.ndim != 4 or video.shape[1] not in (1, 3):
            raise ValueError(
                f"expected a [T, C, H, W] frame tensor, got {list(video.shape)}")
        frames = video
        if frames.shape[1] == 1:
            frames = frames.expand(-1, 3, -1, -1)
        if frames.dtype != torch.uint8:
            frames = frames.clamp(0, 255).to(torch.uint8)
        t = frames.shape[0]
        mf = max_frames or _V_DEFAULT_MAX_FRAMES
        if t > mf:
            idx = np.linspace(0, t - 1, mf, dtype=int)
            frames = frames[torch.as_tensor(idx.copy())]
            t = mf
        n = _v_ceil(t, _V_FRAME_FACTOR)
        if t < n:
            frames = torch.cat([frames, frames[-1:].expand(n - t, -1, -1, -1)])
        metadata = dict(fps=2.0, frames_indices=list(range(n)),
                        total_num_frames=float(n))
        return frames, metadata

    if isinstance(video, (str, os.PathLike)):
        v = str(video)
        if v.startswith("file://"):
            v = v[7:]
        try:
            from torchcodec.decoders import VideoDecoder
        except ImportError as e:
            raise ImportError(
                "embedding a video by file path requires torchcodec "
                "(pip install torchcodec); alternatively pass decoded frames "
                "(a [T, C, H, W] tensor or a list of PIL images)") from e
        decoder = VideoDecoder(v)
        video_fps = decoder.metadata.average_fps
        total = decoder.metadata.num_frames
        want_fps = fps or _V_DEFAULT_FPS
        min_frames = _v_ceil(_V_FPS_MIN_FRAMES, _V_FRAME_FACTOR)
        max_f = _v_floor(max_frames or _V_DEFAULT_MAX_FRAMES, _V_FRAME_FACTOR)
        n = total / video_fps * want_fps
        n = min(min(max(n, min_frames), max_f), total)
        n = _v_floor(n, _V_FRAME_FACTOR)
        if not (_V_FRAME_FACTOR <= n <= total):
            raise ValueError(
                f"video too short: {total} frames; need >= {_V_FRAME_FACTOR}")
        idx = torch.linspace(0, total - 1, n).round().long().tolist()
        frames = decoder.get_frames_at(indices=idx).data
        metadata = dict(fps=video_fps, frames_indices=idx,
                        total_num_frames=total, video_backend="torchcodec")
        return frames, metadata

    # frame sequence (PIL images and/or paths)
    frames = list(video)
    if not frames:
        raise ValueError("empty frame sequence")
    mf = max_frames or _V_DEFAULT_MAX_FRAMES
    if len(frames) > mf:
        idx = np.linspace(0, len(frames) - 1, mf, dtype=int)
        frames = [frames[i] for i in idx]
    images = [_v_frame_to_image(f) for f in frames]
    n = _v_ceil(len(images), _V_FRAME_FACTOR)
    if len(images) < n:
        images.extend([images[-1]] * (n - len(images)))
    tensor = torch.stack([
        torch.from_numpy(np.array(image).transpose(2, 0, 1)) for image in images
    ])
    metadata = dict(fps=2.0, frames_indices=list(range(n)),
                    total_num_frames=float(n))
    return tensor, metadata


def _v_resize_video(frames: torch.Tensor) -> torch.Tensor:
    """Per-video smart resize under the total-pixel budget (reference exact)."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    n, _, height, width = frames.shape
    max_pixels = max(min(_V_MAX_PIXELS, _V_TOTAL_PIXELS / n * _V_FRAME_FACTOR),
                     int(_V_MIN_PIXELS * 1.05))
    rh, rw = _v_smart_resize(height, width, _V_PATCH_FACTOR,
                             _V_MIN_PIXELS, max_pixels)
    return TF.resize(frames, [rh, rw],
                     interpolation=InterpolationMode.BICUBIC,
                     antialias=True).float()


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

        ``video`` is a decoded frame tensor ([T, C, H, W], e.g. straight from
        a torchcodec ``VideoDecoder``), a file path/URL (decoded with
        torchcodec, 1 fps up to 64 frames), or a pre-extracted frame sequence
        (PIL images and/or frame paths, sampled uniformly to 64).
        Preprocessing natively reimplements the base model's reference
        scripts (see the module-level helpers above); no extra vision package
        is required. Like images, video is a non-audio input: it takes the
        frozen path (no whitening, no adapters).
        """
        gate = getattr(self.model, "_adapter_gate", None)
        if gate is not None and gate.active:
            # The video path runs through the same (hook-carrying) decoder layers;
            # non-audio inputs must run with the gate closed so the adapter branch
            # never runs. Mirrors the encode_text/embed_image guards.
            raise RuntimeError("adapter gate is open during a video embed — "
                               "non-audio inputs must run with the gate closed")
        frames, metadata = _v_prepare(video, fps, max_frames)
        frames = _v_resize_video(frames)
        text = _chat(DOC_INSTRUCTION, "<|vision_start|><|video_pad|><|vision_end|>")
        inputs = self.proc(text=[text], videos=[frames],
                           video_metadata=[metadata],
                           do_resize=False, do_sample_frames=False,
                           return_tensors="pt").to(self.device)
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
