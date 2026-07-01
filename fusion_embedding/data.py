"""Data pipeline: instruction taxonomy (HLD §6) + manifest dataset + collator.

The instruction lives on the *text/query* side; the audio side is neutral
(``[N × <|audio_pad|>] + <eos>``), so the same audio embeds differently per task.

Two interchangeable backends behind the same interface:
  * synthetic (``SyntheticAudioProcessor`` + ``HashingTokenizer``) — deterministic,
    dependency-free, used by the CPU E2E tests; and
  * real (``load_audio`` / ``RealAudioProcessor`` / a HF tokenizer) — lazy-imported
    behind the HLD §10 seams.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import torch
from torch.utils.data import Dataset

from .config import FusionConfig, TASK_INSTRUCTIONS, TASK_KEYS


def instruction_for(task: str) -> str:
    if task not in TASK_INSTRUCTIONS:
        raise KeyError(f"unknown task '{task}'; valid: {TASK_KEYS}")
    return TASK_INSTRUCTIONS[task]


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------- #
# Tokenizer interface + a dependency-free hashing tokenizer for tests
# ---------------------------------------------------------------------------- #
class FusionTokenizer(Protocol):
    pad_id: int
    eos_id: int
    audio_pad_id: int

    def encode(self, text: str) -> list[int]: ...


@dataclass
class HashingTokenizer:
    """Deterministic whitespace tokenizer (no vocab file). Ordinary tokens map to
    ``[first_real_id, vocab)`` by hashing; special ids match ``FusionConfig.tiny()``."""

    vocab: int = 64
    pad_id: int = 0
    audio_pad_id: int = 1
    eos_id: int = 2
    first_real_id: int = 3

    def encode(self, text: str) -> list[int]:
        toks = text.split()
        span = self.vocab - self.first_real_id
        return [self.first_real_id + (_stable_seed(tok) % span) for tok in toks]


# ---------------------------------------------------------------------------- #
# Audio processor interface (waveform -> mel) + synthetic backend
# ---------------------------------------------------------------------------- #
class AudioProcessor(Protocol):
    def __call__(self, record: dict) -> torch.Tensor: ...   # -> mel [n_mels, F]


@dataclass
class SyntheticAudioProcessor:
    """Deterministic mel from a record id — distinct per item, identical across epochs
    (so the toy overfit in the train E2E can actually memorize the alignment)."""

    cfg: FusionConfig
    min_frames: int = 12
    max_frames: int = 40

    def __call__(self, record: dict) -> torch.Tensor:
        key = str(record.get("audio", record.get("id", record.get("text", ""))))
        g = torch.Generator().manual_seed(_stable_seed("audio::" + key))
        span = self.max_frames - self.min_frames
        F = self.min_frames + int(torch.randint(0, span + 1, (1,), generator=g).item())
        return torch.randn(self.cfg.n_mels, F, generator=g)


def load_audio(path: str, target_sr: int = 16_000):
    """SEAM (HLD §10.4): load `path` as 16 kHz mono waveform. Real deps imported lazily."""
    try:
        import librosa  # type: ignore
    except ImportError as e:  # pragma: no cover - exercised only in the real pipeline
        raise ImportError("load_audio needs the 'hf' extra (librosa/soundfile).") from e
    wav, _ = librosa.load(path, sr=target_sr, mono=True)
    return torch.from_numpy(wav)


@dataclass
class RealAudioProcessor:
    """SEAM (HLD §10.4): wrap the Qwen2.5-Omni feature extractor (16 kHz mono -> 128-mel)."""

    cfg: FusionConfig
    feature_extractor: object = None     # transformers feature extractor, injected at load time

    def __call__(self, record: dict) -> torch.Tensor:  # pragma: no cover - real pipeline only
        if self.feature_extractor is None:
            raise NotImplementedError(
                "RealAudioProcessor seam: inject the Omni feature extractor in load_components."
            )
        wav = load_audio(record["audio"], self.cfg.sample_rate)
        feats = self.feature_extractor(
            wav.numpy(), sampling_rate=self.cfg.sample_rate, return_tensors="pt"
        )
        mel = feats["input_features"][0]                  # [n_mels, F]
        return mel


# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
class CachedFeatureDataset(Dataset):
    """Dataset over precomputed mel features saved as ``.pt`` dicts ``{mel, text, task}``.

    This is the production training source: ``preprocess`` decodes audio -> mel once and
    writes one file per clip; training reads them back here (no audio decode on the GPU
    box). Emits the same item dict shape as ``FusionAudioTextManifest`` so the collator
    and model consume it unchanged.
    """

    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("CachedFeatureDataset got no feature files")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        d = torch.load(self.paths[i], map_location="cpu", weights_only=False)
        task = d.get("task", "sound")
        if task not in TASK_INSTRUCTIONS:
            task = "sound"
        return {
            "mel": d["mel"],
            "text": d["text"],
            "task": task,
            "instruction": instruction_for(task),
        }


class CachedFrameDataset(Dataset):
    """Dataset over precomputed frozen-encoder frames saved as ``{frames, text, task}`` ``.pt``.

    The Option 2 fast path: the frozen audio tower ran once in ``precompute_frames``, so training
    reads frames directly (no audio decode AND no encoder forward). Emits ``frames`` [T, d_audio].
    """

    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("CachedFrameDataset got no frame files")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        d = torch.load(self.paths[i], map_location="cpu", weights_only=False)
        task = d.get("task", "sound")
        if task not in TASK_INSTRUCTIONS:
            task = "sound"
        return {
            "frames": d["frames"],
            "text": d["text"],
            "task": task,
            "instruction": instruction_for(task),
        }


class InMemoryFrameDataset(Dataset):
    """Frames held in RAM (list of ``{frames, text, task}``) — no per-step disk I/O.

    Precomputed frames are ~7x larger than mel, so reading them from a network Volume every
    step starves the GPU. Loading them once into RAM (a few GB) and serving from memory keeps
    the connector fed. Build via ``from_paths`` (one sequential read) or pass items directly.
    """

    def __init__(self, items: Sequence[dict]):
        self.items = list(items)

    @classmethod
    def from_paths(cls, paths: Sequence[str], half: bool = True, log_every: int = 0) -> "InMemoryFrameDataset":
        items = []
        for i, p in enumerate(paths):
            d = torch.load(p, map_location="cpu", weights_only=False)
            fr = d["frames"]
            if half:
                fr = fr.half()                          # halve RAM/copy cost; cast back at use
            items.append({"frames": fr, "text": d["text"], "task": d.get("task", "sound")})
            if log_every and i % log_every == 0:
                print(f"  preloaded {i}/{len(paths)}", flush=True)
        return cls(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        d = self.items[i]
        task = d.get("task", "sound")
        if task not in TASK_INSTRUCTIONS:
            task = "sound"
        return {
            "frames": d["frames"].float(),
            "text": d["text"],
            "task": task,
            "instruction": instruction_for(task),
        }


class FusionAudioTextManifest(Dataset):
    """Audio↔text pairs tagged with a §6 task. Each record: {audio, text, task[, id]}."""

    def __init__(self, records: Sequence[dict], audio_processor: AudioProcessor):
        self.records = list(records)
        self.audio_processor = audio_processor
        for r in self.records:
            if r.get("task", "sound") not in TASK_INSTRUCTIONS:
                raise KeyError(f"record has unknown task: {r.get('task')!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        task = r.get("task", "sound")
        return {
            "mel": self.audio_processor(r),               # [n_mels, F]
            "text": r["text"],
            "task": task,
            "instruction": instruction_for(task),
        }


# ---------------------------------------------------------------------------- #
# Collator: builds model-ready batches (HLD §4.1 token layout)
# ---------------------------------------------------------------------------- #
@dataclass
class FusionCollator:
    cfg: FusionConfig
    tokenizer: FusionTokenizer
    max_text_len: int = 64

    def _pad_mel(self, mels: list[torch.Tensor]):
        n_mels = mels[0].shape[0]
        Fmax = max(m.shape[1] for m in mels)
        B = len(mels)
        out = torch.zeros(B, n_mels, Fmax)
        mask = torch.zeros(B, Fmax, dtype=torch.bool)
        for b, m in enumerate(mels):
            out[b, :, : m.shape[1]] = m
            mask[b, : m.shape[1]] = True
        return out, mask

    def _audio_token_ids(self, B: int):
        """Neutral audio side: N <|audio_pad|> placeholders + <eos>."""
        N = self.cfg.n_query
        ids = torch.full((B, N + 1), self.cfg.audio_pad_id, dtype=torch.long)
        ids[:, -1] = self.cfg.eos_id
        mask = torch.ones_like(ids)
        return ids, mask

    def _text_ids(self, items: list[dict]):
        seqs = []
        for it in items:
            body = f"{it['instruction']} {it['text']}".strip()
            ids = self.tokenizer.encode(body)[: self.max_text_len - 1]
            ids = ids + [self.cfg.eos_id]                 # always end on <eos> for pooling
            seqs.append(ids)
        L = max(len(s) for s in seqs)
        B = len(seqs)
        out = torch.full((B, L), self.cfg.pad_id, dtype=torch.long)
        mask = torch.zeros(B, L, dtype=torch.long)
        for b, s in enumerate(seqs):
            out[b, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[b, : len(s)] = 1
        return out, mask

    def __call__(self, items: list[dict]) -> dict:
        mel, mel_mask = self._pad_mel([it["mel"] for it in items])
        audio_ids, audio_mask = self._audio_token_ids(len(items))
        text_ids, text_mask = self._text_ids(items)
        return {
            "mel": mel,
            "mel_mask": mel_mask,
            "audio_input_ids": audio_ids,
            "audio_attention_mask": audio_mask,
            "text_input_ids": text_ids,
            "text_attention_mask": text_mask,
            "tasks": [it["task"] for it in items],
        }


@dataclass
class FrameCollator(FusionCollator):
    """Collator for the precomputed-frames path: pads frames [B,T,d_audio] + frame_mask,
    reusing the audio-token-id and instruction-templated text logic from FusionCollator."""

    def _pad_frames(self, frames: list[torch.Tensor]):
        d = frames[0].shape[-1]
        Tmax = max(f.shape[0] for f in frames)
        B = len(frames)
        out = torch.zeros(B, Tmax, d)
        mask = torch.zeros(B, Tmax, dtype=torch.bool)
        for b, f in enumerate(frames):
            t = f.shape[0]
            out[b, :t] = f
            mask[b, :t] = True
        return out, mask

    def __call__(self, items: list[dict]) -> dict:
        frames, frame_mask = self._pad_frames([it["frames"] for it in items])
        audio_ids, audio_mask = self._audio_token_ids(len(items))
        text_ids, text_mask = self._text_ids(items)
        return {
            "frames": frames,
            "frame_mask": frame_mask,
            "audio_input_ids": audio_ids,
            "audio_attention_mask": audio_mask,
            "text_input_ids": text_ids,
            "text_attention_mask": text_mask,
            "tasks": [it["task"] for it in items],
        }


# ---------------------------------------------------------------------------- #
# Synthetic dataset builder (tests + demo)
# ---------------------------------------------------------------------------- #
def make_synthetic_records(n: int, tasks: Optional[Sequence[str]] = None) -> list[dict]:
    """n distinct audio↔text pairs, round-robin over the taxonomy, content keyed to id."""
    tasks = list(tasks or TASK_KEYS)
    records = []
    for i in range(n):
        task = tasks[i % len(tasks)]
        records.append(
            {
                "id": f"item-{i}",
                "audio": f"synthetic://item-{i}",
                "text": f"caption number {i} describing the {task} sample token{i}",
                "task": task,
            }
        )
    return records


def make_synthetic_dataset(cfg: FusionConfig, n: int = 16, vocab: int = 64, tasks=None):
    """Return (manifest, collator) wired with the synthetic backend."""
    records = make_synthetic_records(n, tasks=tasks)
    manifest = FusionAudioTextManifest(records, SyntheticAudioProcessor(cfg))
    tokenizer = HashingTokenizer(
        vocab=vocab, pad_id=cfg.pad_id, audio_pad_id=cfg.audio_pad_id, eos_id=cfg.eos_id
    )
    collator = FusionCollator(cfg, tokenizer)
    return manifest, collator
