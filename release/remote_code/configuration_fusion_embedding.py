"""HF remote-code configuration for the fusion-embedding family.

Lives on the model repos (EximiusLabs/fusion-embedding-1-2b-preview and
EximiusLabs/fusion-embedding-2-2b-preview) next to modeling_fusion_embedding.py so the
models load with plain transformers:

    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "EximiusLabs/fusion-embedding-1-2b-preview", trust_remote_code=True)

The config carries the trained-connector dimensions plus the frozen-component repo
names; the frozen Qwen3-VL-Embedding base and Qwen2.5-Omni audio tower are NOT part of
this checkpoint — they are fetched from their own repositories at first use.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class FusionEmbeddingConfig(PretrainedConfig):
    model_type = "fusion-embedding-connector"

    def __init__(
        self,
        d_audio: int = 3584,
        d_llm: int = 2048,
        n_query: int = 64,
        d_resampler: int = 384,
        resampler_depth: int = 6,
        resampler_heads: int = 8,
        resampler_ffn_mult: int = 4,
        resampler_dropout: float = 0.0,
        adapter_rank: int = 0,
        adapter_act: str = "silu",
        mrl_dims=(2048, 1536, 1024, 512, 256, 128, 64),
        mrl_default: int = 1024,
        audio_pad_id: int = 151654,
        eos_id: int = 151645,
        pad_id: int = 151643,
        audio_pad_token: str = "<|audio_pad|>",
        base_model: str = "Qwen/Qwen3-VL-Embedding-2B",
        audio_model: str = "Qwen/Qwen2.5-Omni-7B",
        max_text_tokens: int = 512,
        n_decoder_layers: int = 28,
        **kwargs,
    ):
        self.d_audio = d_audio
        self.d_llm = d_llm
        self.n_query = n_query
        self.d_resampler = d_resampler
        self.resampler_depth = resampler_depth
        self.resampler_heads = resampler_heads
        self.resampler_ffn_mult = resampler_ffn_mult
        self.resampler_dropout = resampler_dropout
        self.adapter_rank = adapter_rank or 0
        self.adapter_act = adapter_act or "silu"
        self.mrl_dims = list(mrl_dims)
        self.mrl_default = mrl_default
        self.audio_pad_id = audio_pad_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.audio_pad_token = audio_pad_token
        self.base_model = base_model
        self.audio_model = audio_model
        self.max_text_tokens = max_text_tokens
        self.n_decoder_layers = n_decoder_layers
        super().__init__(**kwargs)
