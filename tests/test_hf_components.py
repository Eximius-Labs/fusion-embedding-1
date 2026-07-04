"""Device-placement gate for the frozen base loader.

The 4-bit path is placed by ``device_map={"": device}`` at load; the bf16 path loads on CPU and
must be moved explicitly. That bf16 branch was untested (everything ran in 4-bit) and crashed the
first bf16 run (2026-07-02: embed_tokens on CPU vs cuda inputs). These tests pin the branch logic
WITHOUT a real model / GPU by recording ``.to()`` calls on a stand-in.
"""

from fusion_embedding.hf_components import _place_frozen_base


class _FakeBase:
    def __init__(self):
        self.moved_to = None

    def to(self, device):          # HF modules return self from .to(...)
        self.moved_to = device
        return self


def test_place_frozen_base_moves_non_quantised_to_device():
    b = _FakeBase()
    out = _place_frozen_base(b, quant=None, device="cuda:0")
    assert out is b and b.moved_to == "cuda:0"        # bf16: CPU-loaded -> must be moved


def test_place_frozen_base_leaves_4bit_untouched():
    b = _FakeBase()
    out = _place_frozen_base(b, quant=object(), device="cuda:0")   # object() stands in for BnB config
    assert out is b and b.moved_to is None            # 4-bit: device_map placed it; .to() would error
