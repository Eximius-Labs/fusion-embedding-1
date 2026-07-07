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
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Protocol, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

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
        rec = {
            "frames": d["frames"].float(),
            "text": d["text"],
            "task": task,
            "instruction": instruction_for(task),
        }
        if d.get("text_emb") is not None:                         # cached RAW text (Step 2), if present
            rec["text_emb"] = d["text_emb"].float()
        return rec


# ---------------------------------------------------------------------------- #
# Sharded frames: many clips per file → one big sequential read, no per-clip
# network round-trips, no full-RAM preload. Replaces the per-clip `.pt` + preload
# path for large corpora (the read bottleneck was per-file latency, not bandwidth).
# ---------------------------------------------------------------------------- #
def write_frame_shard(path, records: Sequence[dict], half: bool = True) -> None:
    """Write a list of ``{frames, text, task}`` as ONE shard file (parallel lists).

    ``half`` stores frames as fp16 (halves shard size / read cost; cast back to float at use).
    """
    frames = [(r["frames"].half() if half else r["frames"]).contiguous() for r in records]
    torch.save({"frames": frames,
                "text": [r["text"] for r in records],
                "task": [r.get("task", "sound") for r in records]}, str(path))


def text_emb_shard_path(frame_shard_path, tag: str = "") -> str:
    """Sibling path holding a frame shard's precomputed RAW text embeddings (Step 2 cache).

    Kept SEPARATE from the frame shard on disk (``shard-0000.pt`` → ``shard-0000.txtemb.pt``) but
    joined by clip position, so the text cache can be (re)built independently of the frames.
    ``tag`` names an ALTERNATE cache variant (e.g. ``_native`` for chat-template targets) so
    variants coexist: ``shard-0000.txtemb_native.pt``.
    """
    p = str(frame_shard_path)
    suffix = f".txtemb{tag}.pt"
    return p[:-3] + suffix if p.endswith(".pt") else p + suffix


def write_text_emb_shard(frame_shard_path, embs: torch.Tensor, tag: str = "") -> None:
    """Write ``[n_clips, d_llm]`` fp16 RAW (pre-whitening) text embeddings beside a frame shard."""
    torch.save({"text_emb": embs.half().contiguous()}, text_emb_shard_path(frame_shard_path, tag))


def _shard_record(shard: dict, off: int, text_embs: Optional[torch.Tensor] = None) -> dict:
    task = shard["task"][off]
    if task not in TASK_INSTRUCTIONS:
        task = "sound"
    rec = {"frames": shard["frames"][off].float(), "text": shard["text"][off],
           "task": task, "instruction": instruction_for(task)}
    if text_embs is not None:
        rec["text_emb"] = text_embs[off].float()               # cached RAW pooled text (whitened at use)
    return rec


def shard_starts_from(n_shards: int, shard_size: int, n_total: int) -> list[int]:
    """Global start index of each shard for a SINGLE source (only its last shard is partial)."""
    return [min(p * shard_size, n_total) for p in range(n_shards)]


def load_frame_clips(shard_paths: Sequence, shard_starts: Sequence[int],
                     global_indices: Iterable[int], *, with_text_emb: bool = False,
                     text_emb_tag: str = "") -> list[dict]:
    """Materialise specific clips (by global index) into a list — for the small held-out eval set.

    ``shard_starts[p]`` is the global index of clip 0 in shard ``p`` (robust to partial shards when
    several sources are concatenated). Reads only the shards that actually hold a wanted clip.
    ``with_text_emb`` also loads the sibling text-emb cache (Step 2) and attaches ``text_emb`` per clip.
    """
    import bisect
    import os
    from collections import defaultdict
    starts = list(shard_starts)
    by_shard: dict = defaultdict(list)
    for g in global_indices:
        pos = bisect.bisect_right(starts, g) - 1                 # shard whose start is ≤ g
        by_shard[pos].append(g - starts[pos])
    out: list[dict] = []
    for pos in sorted(by_shard):
        shard = torch.load(str(shard_paths[pos]), map_location="cpu", weights_only=False)
        tembs = None
        if with_text_emb:
            tp = text_emb_shard_path(shard_paths[pos], text_emb_tag)
            if not os.path.exists(tp):
                raise FileNotFoundError(f"text cache missing: {tp} — run precompute_text_cache first")
            tembs = torch.load(tp, map_location="cpu", weights_only=False)["text_emb"]
        for off in sorted(by_shard[pos]):
            out.append(_shard_record(shard, off, tembs))
    return out


class ShardedFrameDataset(IterableDataset):
    """Streams frames from N-clip shard files with a reservoir shuffle buffer.

    One ``torch.load`` per shard is a single big sequential read (hundreds of clips), so this
    avoids both the full-RAM preload and the per-clip network latency of ``CachedFrameDataset``.
    ``shard_starts[p]`` gives the global index of shard ``p``'s first clip (so several sources'
    shards can be concatenated even with partial last shards); ``exclude`` is a set of those global
    indices to skip (the held-out eval clips). Shards are split disjointly across DataLoader workers,
    and re-iterating yields a fresh (reshuffled) pass. Emits ``{frames, text, task, instruction}``.
    """

    def __init__(self, shard_paths: Sequence, shard_starts: Sequence[int], *,
                 exclude: Optional[set] = None, shuffle_buffer: int = 2048,
                 shuffle_shards: bool = True, seed: int = 0, use_text_emb: bool = False,
                 text_emb_tag: str = "", max_frames: int = 0):
        self.shard_paths = [str(p) for p in shard_paths]
        self.shard_starts = list(shard_starts)
        self.exclude = set(exclude or ())
        self.shuffle_buffer = max(1, shuffle_buffer)
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.use_text_emb = use_text_emb                          # attach cached RAW text (Step 2)
        self.text_emb_tag = text_emb_tag
        # Random-crop long clips to max_frames (~25 frames/s; 250 = 10 s, the CLAP-standard
        # training window). 0 = no crop. Cuts I/O and worker RAM ~2.5x on long-clip corpora
        # (FreeSound averages ~25 s) and doubles as light temporal augmentation.
        self.max_frames = int(max_frames)
        self._epoch = 0

    def __iter__(self):
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        nworkers = worker.num_workers if worker is not None else 1
        order = list(range(len(self.shard_paths)))
        if self.shuffle_shards:
            random.Random(self.seed + self._epoch).shuffle(order)
        self._epoch += 1
        order = order[wid::nworkers]                              # disjoint shard subset per worker
        buf_rng = random.Random(self.seed * 7919 + wid + self._epoch)
        buffer: list = []
        for pos in order:
            shard = torch.load(self.shard_paths[pos], map_location="cpu", weights_only=False)
            tembs = None
            if self.use_text_emb:
                import os
                tp = text_emb_shard_path(self.shard_paths[pos], self.text_emb_tag)
                if not os.path.exists(tp):
                    raise FileNotFoundError(f"text cache missing: {tp} — run precompute_text_cache first")
                tembs = torch.load(tp, map_location="cpu", weights_only=False)["text_emb"]
            start = self.shard_starts[pos]
            for off in range(len(shard["text"])):
                if (start + off) in self.exclude:
                    continue
                rec = _shard_record(shard, off, tembs)
                if self.max_frames and rec["frames"].shape[0] > self.max_frames:
                    t0 = buf_rng.randrange(rec["frames"].shape[0] - self.max_frames + 1)
                    rec["frames"] = rec["frames"][t0: t0 + self.max_frames].contiguous()
                if len(buffer) < self.shuffle_buffer:
                    buffer.append(rec)
                else:
                    j = buf_rng.randrange(len(buffer))
                    buffer[j], rec = rec, buffer[j]
                    yield rec
        buf_rng.shuffle(buffer)
        yield from buffer


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
        batch = {
            "frames": frames,
            "frame_mask": frame_mask,
            "audio_input_ids": audio_ids,
            "audio_attention_mask": audio_mask,
            "text_input_ids": text_ids,
            "text_attention_mask": text_mask,
            "tasks": [it["task"] for it in items],
            "texts": [it["text"] for it in items],   # raw captions (bank own-positive masking)
        }
        if all(it.get("text_emb") is not None for it in items):  # Step 2: RAW text cache → skip base
            batch["text_emb_cached"] = torch.stack([it["text_emb"] for it in items])
        return batch


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
