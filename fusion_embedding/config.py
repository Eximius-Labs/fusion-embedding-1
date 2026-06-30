"""``FusionConfig`` — the single place every confirmed number from the HLD lives.

All anchors in HLD §3 are encoded here as defaults. ``FusionConfig.tiny()`` returns
a miniature config with identical *structure* but tiny dims so the whole pipeline
runs end-to-end on CPU in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# --- Instruction taxonomy (HLD §6). Query/text side carries the instruction;
#     the audio side is neutral. The same audio embeds differently per task. ---
TASK_INSTRUCTIONS = {
    "sound": "Retrieve audio by sound description.",
    "speech_content": "Retrieve audio by spoken content.",
    "music": "Retrieve music by description.",
    "speech_language": "Retrieve speech by language.",
    "speech_paralinguistic": "Retrieve speech by speaker and emotion.",
}
TASK_KEYS = tuple(TASK_INSTRUCTIONS.keys())


@dataclass
class FusionConfig:
    """Confirmed dims + Stage-1 hyperparameters.

    Every default here is a *locked decision* from the HLD. The ``tiny`` factory
    overrides only the magnitudes, never the shape of the computation, so tests
    exercise the production code paths.
    """

    # --- Dimensions (HLD §3) ---
    d_audio: int = 1280          # Qwen2.5-Omni audio tower output dim
    d_llm: int = 2048            # Qwen3-VL-Embedding-2B hidden == embedding dim
    n_query: int = 64            # FusionResampler latent queries (N). P1=64, P3 speech=128-200

    # --- FusionResampler (HLD §4.2) ---
    # The resampler runs at an internal bottleneck width `d_resampler`, projecting
    # audio frames d_audio -> d_resampler on input and d_resampler -> d_llm on output.
    # This reconciles "L=6 blocks" with the HLD's "single-digit millions / ~0.35%
    # trained" budget: 6 full-width blocks at d_llm=2048 would be ~405M params (~20%
    # of the 2B base), contradicting the whole frozen-base thesis. At d_resampler=256
    # the connector is ~7M params. See HLD §4.2 / Flamingo's resampler.
    d_resampler: int = 256
    resampler_depth: int = 6     # L = 6 pre-norm blocks
    resampler_heads: int = 8
    resampler_ffn_mult: int = 4  # FFN expansion (4x)
    resampler_dropout: float = 0.0

    # --- Matryoshka ladder (HLD §3) ---
    mrl_dims: tuple[int, ...] = (2048, 1536, 1024, 512, 256, 128, 64)
    mrl_default: int = 1024
    # Per-rung loss weights; None => equal weight on every rung.
    mrl_weights: Optional[tuple[float, ...]] = None

    # --- Long / variable audio (HLD §4.2, §5.3) ---
    max_windows: int = 8         # cap on concatenated 30 s windows
    use_global_summary: bool = True  # LAION-CLAP feature-fusion of a global window

    # --- Contrastive loss (HLD §5.1) ---
    # logit_scale = log(1/temperature). CLIP init temp 0.07 -> log(1/0.07).
    logit_scale_init: float = math.log(1.0 / 0.07)
    logit_scale_max: float = math.log(100.0)   # clamp (CLIP convention)
    learnable_temperature: bool = True
    lambda_coral: float = 0.05   # light in P1, raised in P2
    debias_gamma: float = 0.0    # P1 off; P2 ~0.1 (debiased contrastive)
    use_hard_negatives: bool = False  # P1 off; P2 mines hard negatives

    # --- Optimization (HLD §5.3) ---
    lr: float = 1e-4
    weight_decay: float = 0.0    # connector-only; no decay on the resampler
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    warmup_ratio: float = 0.05
    max_steps: int = 10_000
    micro_batch_size: int = 64
    grad_accum_steps: int = 8    # effective batch = micro * accum * world_size
    grad_clip: float = 1.0
    use_bf16: bool = True        # autocast on CUDA only (guarded at call site)

    # --- Audio frontend (HLD §3, §5.3) ---
    sample_rate: int = 16_000
    n_mels: int = 128
    win_ms: float = 25.0
    hop_ms: float = 10.0
    window_seconds: float = 30.0
    audio_frames_per_second: int = 25   # after Omni stride-2 pooling (~40 ms/frame)

    # --- Special tokens (ids resolved at component-load time, HLD §10) ---
    audio_pad_token: str = "<|audio_pad|>"
    audio_pad_id: int = -1       # filled in by load_components / tiny builder
    eos_id: int = -1             # filled in by load_components / tiny builder
    pad_id: int = 0

    seed: int = 0

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        if self.d_resampler % self.resampler_heads != 0:
            raise ValueError(
                f"d_resampler={self.d_resampler} must be divisible by resampler_heads={self.resampler_heads}"
            )
        if self.d_resampler > self.d_llm:
            raise ValueError(
                f"d_resampler={self.d_resampler} should not exceed d_llm={self.d_llm}"
            )
        if not self.mrl_dims:
            raise ValueError("mrl_dims must be non-empty")
        if any(d > self.d_llm for d in self.mrl_dims):
            raise ValueError(f"every MRL rung must be <= d_llm={self.d_llm}; got {self.mrl_dims}")
        if any(d <= 0 for d in self.mrl_dims):
            raise ValueError(f"MRL rungs must be positive; got {self.mrl_dims}")
        if len(set(self.mrl_dims)) != len(self.mrl_dims):
            raise ValueError(f"MRL rungs must be unique; got {self.mrl_dims}")
        if self.mrl_default not in self.mrl_dims:
            raise ValueError(
                f"mrl_default={self.mrl_default} must be one of mrl_dims={self.mrl_dims}"
            )
        if self.mrl_weights is not None and len(self.mrl_weights) != len(self.mrl_dims):
            raise ValueError(
                f"mrl_weights length {len(self.mrl_weights)} != mrl_dims length {len(self.mrl_dims)}"
            )
        if self.n_query <= 0:
            raise ValueError("n_query must be positive")

    # ------------------------------------------------------------------ #
    @property
    def normalized_mrl_weights(self) -> tuple[float, ...]:
        """Per-rung weights, normalized to sum 1 (equal weight if unspecified)."""
        raw = self.mrl_weights if self.mrl_weights is not None else tuple(1.0 for _ in self.mrl_dims)
        total = float(sum(raw))
        if total <= 0:
            raise ValueError("mrl_weights must sum to a positive value")
        return tuple(w / total for w in raw)

    @property
    def frames_per_window(self) -> int:
        """~750 frames per 30 s at 25 fps (HLD §3)."""
        return int(round(self.audio_frames_per_second * self.window_seconds))

    def with_tokens(self, *, audio_pad_id: int, eos_id: int, pad_id: Optional[int] = None) -> "FusionConfig":
        """Return a copy with token ids resolved (called after a tokenizer is known)."""
        from dataclasses import replace

        kw = dict(audio_pad_id=audio_pad_id, eos_id=eos_id)
        if pad_id is not None:
            kw["pad_id"] = pad_id
        return replace(self, **kw)

    # ------------------------------------------------------------------ #
    @classmethod
    def tiny(cls, **overrides) -> "FusionConfig":
        """Miniature config: same structure, tiny magnitudes — for CPU E2E tests.

        Token ids (audio_pad_id=1, eos_id=2) are pre-filled so the tiny builder can
        construct a self-contained vocab without a real tokenizer.
        """
        defaults = dict(
            d_audio=16,
            d_llm=32,
            d_resampler=16,
            n_query=4,
            resampler_depth=2,
            resampler_heads=2,
            resampler_ffn_mult=2,
            mrl_dims=(32, 16, 8),
            mrl_default=16,
            max_windows=3,
            n_mels=8,
            audio_frames_per_second=25,
            micro_batch_size=4,
            grad_accum_steps=1,
            max_steps=50,
            use_bf16=False,
            audio_pad_id=1,
            eos_id=2,
            pad_id=0,
            seed=0,
        )
        defaults.update(overrides)
        return cls(**defaults)
