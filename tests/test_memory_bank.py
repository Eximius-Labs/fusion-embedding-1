"""Stage #3 gate: the frozen-text memory bank — FIFO mechanics, loss math, and the
small-micro-batch payoff (the lever that makes 8GB contrastive training viable)."""

import itertools

import torch
from torch.utils.data import DataLoader

from fusion_embedding.config import FusionConfig
from fusion_embedding.losses import FusionContrastiveLoss
from fusion_embedding.memory_bank import TextMemoryBank, precompute_text_bank
from fusion_embedding.data import (
    FusionAudioTextManifest,
    SyntheticAudioProcessor,
    make_synthetic_dataset,
)
from fusion_embedding.train_stage1 import (
    build_optimizer,
    encode_dataset,
    retrieval_report,
    set_seed,
)
from fusion_embedding._tiny import build_tiny_model


# ----------------------------- FIFO mechanics ------------------------------ #
def test_bank_starts_empty():
    bank = TextMemoryBank(dim=8, capacity=5)
    assert bank.is_empty and len(bank) == 0 and bank.get() is None


def test_bank_enqueue_and_get():
    bank = TextMemoryBank(dim=4, capacity=6)
    bank.enqueue(torch.ones(3, 4))
    assert len(bank) == 3 and bank.get().shape == (3, 4)


def test_bank_wraparound_keeps_most_recent():
    bank = TextMemoryBank(dim=1, capacity=4)
    for v in range(6):                       # enqueue scalars 0..5 one at a time
        bank.enqueue(torch.full((1, 1), float(v)))
    assert len(bank) == 4
    got = set(bank.get().flatten().tolist())
    assert got == {2.0, 3.0, 4.0, 5.0}       # oldest (0,1) evicted


def test_bank_batch_larger_than_capacity():
    bank = TextMemoryBank(dim=2, capacity=3)
    emb = torch.arange(10 * 2, dtype=torch.float32).reshape(10, 2)
    bank.enqueue(emb)
    assert len(bank) == 3
    assert torch.equal(bank.get(), emb[-3:])  # keeps the last `capacity`


def test_bank_stores_detached():
    bank = TextMemoryBank(dim=3, capacity=4)
    x = torch.randn(2, 3, requires_grad=True)
    bank.enqueue(x * 2)
    assert not bank.get().requires_grad


# ------------------------------- loss math --------------------------------- #
def test_bank_negatives_increase_loss():
    cfg = FusionConfig.tiny(lambda_coral=0.0)
    loss_fn = FusionContrastiveLoss(cfg)
    a = torch.randn(4, cfg.d_llm)
    t = torch.randn(4, cfg.d_llm)
    ls = torch.tensor(3.0)
    base, _ = loss_fn(a, t, ls)
    bank = torch.randn(32, cfg.d_llm)                 # 32 shared negatives
    withbank, _ = loss_fn(a, t, ls, bank_text=bank)
    assert withbank.item() > base.item()              # more true negatives -> higher loss


def test_bank_gradients_flow_to_audio_and_scale():
    cfg = FusionConfig.tiny(lambda_coral=0.0)
    loss_fn = FusionContrastiveLoss(cfg)
    a = torch.randn(2, cfg.d_llm, requires_grad=True)
    t = torch.randn(2, cfg.d_llm, requires_grad=True)
    ls = torch.tensor(2.0, requires_grad=True)
    bank = torch.randn(50, cfg.d_llm)
    loss, _ = loss_fn(a, t, ls, bank_text=bank)
    loss.backward()
    assert a.grad.abs().sum() > 0 and ls.grad.abs() > 0
    assert torch.isfinite(loss)


# -------------------- the small-micro-batch payoff ------------------------- #
def _distractor_manifest(cfg, n):
    """A bank source whose texts are disjoint from the training captions (clean negatives)."""
    recs = [
        {"id": f"distract-{i}", "audio": f"synthetic://distract-{i}",
         "text": f"unrelated distractor utterance {i} xyzzy", "task": "sound"}
        for i in range(n)
    ]
    return FusionAudioTextManifest(recs, SyntheticAudioProcessor(cfg))


def _train_micro_batch(use_bank: bool, *, steps: int, seed: int, n_train: int, batch_size: int):
    cfg = FusionConfig.tiny(max_steps=steps, d_resampler=32)
    set_seed(seed)
    model = build_tiny_model(cfg, seed=seed)
    manifest, collator = make_synthetic_dataset(cfg, n=n_train)
    loss_fn = FusionContrastiveLoss(cfg)

    bank = None
    if use_bank:
        bank = precompute_text_bank(model, _distractor_manifest(cfg, 24), collator)  # [24, d_llm], frozen

    loader = DataLoader(manifest, batch_size=batch_size, collate_fn=collator, shuffle=True)
    opt = build_optimizer(model, cfg)
    di = itertools.cycle(loader)
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(next(di))
        loss, _ = loss_fn(out["audio"], out["text"], out["logit_scale"], bank_text=bank)
        loss.backward()
        opt.step()

    a, t = encode_dataset(model, manifest, collator)
    return retrieval_report(a, t)["a2t_R@1"]


def test_bank_helps_small_micro_batch():
    """At micro-batch 2 the in-batch signal is starved (1 negative); the frozen-text bank
    restores many negatives, so it should retrieve at least as well — and meaningfully well."""
    common = dict(steps=160, seed=0, n_train=12, batch_size=2)
    without = _train_micro_batch(False, **common)
    withbank = _train_micro_batch(True, **common)
    assert withbank >= without                 # the bank never hurts
    assert withbank >= 0.75                     # and yields strong retrieval despite micro-batch 2
