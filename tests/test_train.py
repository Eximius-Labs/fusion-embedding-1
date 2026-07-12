"""Stage F gate: optimizer/schedule/guard primitives + the connector-only loop mechanics."""

import torch

from fusion_embedding.config import FusionConfig
from fusion_embedding.train_stage1 import (
    RegressionGuard,
    average_precision_at_k,
    build_optimizer,
    build_scheduler,
    cosine_warmup,
    encode_dataset,
    filter_clips_by_allowlist,
    flatten_caption_groups,
    lexical_relevance,
    multicaption_relevance,
    recall_at_k,
    retrieval_report,
    semantic_relevance,
)
from fusion_embedding.model import mrl_truncate_normalize
from fusion_embedding._tiny import build_tiny_model
from .conftest import make_batch


def test_cosine_warmup_shape():
    warm, total = 10, 100
    assert cosine_warmup(0, warm, total) < cosine_warmup(5, warm, total)   # warming up
    assert abs(cosine_warmup(warm - 1, warm, total) - 1.0) < 1e-6          # peak at end of warmup
    assert cosine_warmup(total, warm, total) < 1e-6                        # decays to ~0


def test_optimizer_only_holds_trainable_params(tiny_model):
    opt = build_optimizer(tiny_model, tiny_model.cfg)
    opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert opt_param_ids == {id(p) for p in tiny_model.trainable_parameters()}
    # no frozen base param is in the optimizer
    frozen_ids = {id(p) for comp in tiny_model.frozen_modules() for p in comp.parameters()}
    assert opt_param_ids.isdisjoint(frozen_ids)


def test_regression_guard_detects_drift(tiny_model):
    guard = RegressionGuard(tiny_model)
    assert guard.max_drift(tiny_model) == 0.0
    # mutate a frozen base param
    with torch.no_grad():
        next(tiny_model.base_lm.parameters()).add_(1.0)
    assert guard.max_drift(tiny_model) > 0
    import pytest
    with pytest.raises(RuntimeError):
        guard.check(tiny_model)


def test_recall_at_k_perfect_and_worst():
    perfect = torch.eye(5) * 10
    r = recall_at_k(perfect, ks=(1,))
    assert r["R@1"] == 1.0
    # anti-diagonal of EVEN size is a derangement -> positive never top-1
    bad = torch.eye(4).flip(1) * 10
    assert recall_at_k(bad, ks=(1,))["R@1"] == 0.0


def test_retrieval_report_keys():
    a = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    rep = retrieval_report(a, a)            # identical -> perfect both directions
    assert rep["a2t_R@1"] == 1.0 and rep["t2a_R@1"] == 1.0
    # new metrics present and perfect on the identity case
    for key in ("a2t_R@5", "a2t_R@10", "a2t_mAP@10", "t2a_mAP@10"):
        assert key in rep, key
    assert rep["a2t_mAP@10"] == 1.0


def test_map_at_k_equals_mrr_for_single_relevant():
    # put the one relevant item at a known rank per query, mAP@k should equal mean(1/rank)
    Q = 4
    sims = torch.zeros(Q, 6)
    for q in range(Q):
        sims[q] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])  # ranking is columns 0..5
    relevance = torch.zeros(Q, 6, dtype=torch.bool)
    relevant_col = [0, 1, 2, 9 % 6]     # ranks 1,2,3,... -> reciprocal ranks
    for q, c in enumerate([0, 1, 2, 4]):
        relevance[q, c] = True
    got = average_precision_at_k(sims, relevance, k=10)
    expected = (1 / 1 + 1 / 2 + 1 / 3 + 1 / 5) / Q
    assert abs(got - expected) < 1e-6, (got, expected)


def test_map_at_k_zero_when_relevant_outside_topk():
    sims = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    relevance = torch.tensor([[False, False, False, False, True]])  # relevant is rank 5
    assert average_precision_at_k(sims, relevance, k=3) == 0.0      # not in top-3
    assert abs(average_precision_at_k(sims, relevance, k=5) - 1 / 5) < 1e-6


def test_semantic_relevance_credits_duplicate_captions():
    # two identical caption vectors (clips 0 and 1) + two distinct ones
    v = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    text = torch.cat([v, v, torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)])
    rel = semantic_relevance(text, threshold=0.99)
    assert rel[0, 1] and rel[1, 0]                 # duplicates are mutually relevant
    assert rel[2, 2] and not rel[2, 0]             # distinct captions only self-relevant
    # a perfectly-aligned audio side: grouped mAP >= diagonal mAP (duplicates no longer penalised)
    audio = text.clone()
    diag = retrieval_report(audio, text)["a2t_mAP@10"]
    grouped = retrieval_report(audio, text, relevance=rel)["a2t_mAP@10"]
    assert grouped >= diag


def test_multicaption_relevance_min_rank_over_refs():
    # 3 audio clips, 2 captions each (caption j -> its audio via group ids)
    group_ids = [0, 0, 1, 1, 2, 2]
    rel = multicaption_relevance(group_ids)
    assert rel.shape == (3, 6)
    assert rel[0, 0] and rel[0, 1] and not rel[0, 2]          # clip 0 owns captions 0,1
    # audio embeds match ONE of their two captions perfectly, the other is far:
    d = 8
    caps = torch.nn.functional.normalize(torch.randn(6, d), dim=-1)
    audio = caps[[0, 2, 4]].clone()                           # audio i aligns with its FIRST caption
    rep = retrieval_report(audio, caps, relevance=rel)
    assert rep["a2t_R@1"] == 1.0                              # each audio's top-1 is a valid ref -> min-rank hit
    assert rep["a2t_mAP@10"] > 0.0
    # T→A: 6 caption-queries retrieve among 3 audio; the aligned ones rank their audio first
    assert rep["t2a_R@1"] >= 0.5


def test_lexical_relevance_by_word_overlap():
    caps = ["a dog is barking loudly", "a dog barking loudly", "rain on a tin roof", "birds chirping"]
    rel = lexical_relevance(caps, threshold=0.5)
    assert rel[0, 1] and rel[1, 0]                 # high word overlap -> mutually relevant
    assert not rel[0, 2] and not rel[2, 3]         # disjoint captions -> not relevant
    assert rel[3, 3]                               # diagonal always on
    # symmetric and boolean
    assert rel.dtype == torch.bool and torch.equal(rel, rel.t())


def test_flatten_caption_groups_pairs_with_multicaption_relevance():
    captions_multi = [["a dog barks", "barking dog"], ["rain falls"], ["a bell rings", "ringing", "chime"]]
    flat, gids = flatten_caption_groups(captions_multi)
    assert flat == ["a dog barks", "barking dog", "rain falls", "a bell rings", "ringing", "chime"]
    assert gids == [0, 0, 1, 2, 2, 2]
    # feeds straight into the protocol relevance matrix: [n_clips, n_caps]
    rel = multicaption_relevance(gids, n_audio=len(captions_multi))
    assert rel.shape == (3, 6)
    assert rel[0, 0] and rel[0, 1] and not rel[0, 2]      # clip 0 owns its two captions only
    assert rel[2, 3] and rel[2, 4] and rel[2, 5]          # clip 2 owns its three


def test_filter_clips_by_allowlist_restricts_to_canonical_split():
    caps_multi = [["a"], ["b", "b2"], ["c"], ["d"]]
    clip_ids = ["yt0|0", "yt1|0", "yt2|5", "yt3|0"]
    kept, kept_caps = filter_clips_by_allowlist(caps_multi, clip_ids, {"yt1|0", "yt3|0", "notpresent"})
    assert kept == [1, 3]                                       # on-disk positions preserved (aligned to frames)
    assert kept_caps == [["b", "b2"], ["d"]]
    # empty allowlist -> nothing kept; full -> everything, order stable
    assert filter_clips_by_allowlist(caps_multi, clip_ids, set())[0] == []
    assert filter_clips_by_allowlist(caps_multi, clip_ids, set(clip_ids))[0] == [0, 1, 2, 3]


def test_multicaption_eval_flow_end_to_end(tiny_model):
    """Mirror rescore_816's plumbing on the tiny model: encode audio (one per clip) and ALL
    reference captions SEPARATELY, then score with a non-square multicaption relevance matrix."""
    import os
    import tempfile

    from fusion_embedding.data import CachedFrameDataset, FrameCollator, HashingTokenizer
    from fusion_embedding._tiny import TINY_VOCAB

    cfg = tiny_model.cfg
    tok = HashingTokenizer(vocab=TINY_VOCAB, pad_id=cfg.pad_id,
                           audio_pad_id=cfg.audio_pad_id, eos_id=cfg.eos_id)
    collator = FrameCollator(cfg, tok)
    dim = cfg.mrl_default

    # 4 clips, 2 captions each; audio frames stored one per clip (first caption as the stored text)
    captions_multi = [[f"clip {i} caption {j}" for j in range(2)] for i in range(4)]
    d = tempfile.mkdtemp()
    g = torch.Generator().manual_seed(0)
    recs = []
    for i in range(4):
        p = os.path.join(d, f"f{i}.pt")
        torch.save({"frames": torch.randn(4 + i, cfg.d_audio, generator=g),
                    "text": captions_multi[i][0], "task": "sound"}, p)
        recs.append(p)
    audio_emb, _ = encode_dataset(tiny_model, CachedFrameDataset(recs), collator, dim=dim)

    # encode every reference caption separately (encode_text -> whiten -> MRL-normalise)
    flat, gids = flatten_caption_groups(captions_multi)
    ids, mask = collator._text_ids([{"instruction": "describe the sound", "text": c} for c in flat])
    raw = tiny_model.encode_text(ids, mask)
    text_emb = mrl_truncate_normalize(tiny_model.text_whitening(raw), dim)

    assert audio_emb.shape == (4, dim) and text_emb.shape == (8, dim)   # non-square: 4 clips vs 8 caps
    rel = multicaption_relevance(gids, n_audio=4)
    rep = retrieval_report(audio_emb, text_emb, relevance=rel)
    for key in ("a2t_R@1", "a2t_R@10", "a2t_mAP@10", "t2a_R@1"):
        assert 0.0 <= rep[key] <= 1.0                                  # valid, finite scores


def test_peak_lr_override_reaches_optimizer():
    """The Stage-3 LR fix: FusionConfig(lr=...) must flow into every optimizer param group."""
    cfg = FusionConfig.tiny(lr=3e-4)
    assert cfg.lr == 3e-4
    model = build_tiny_model(cfg)
    opt = build_optimizer(model, cfg)
    assert all(abs(g["lr"] - 3e-4) < 1e-12 for g in opt.param_groups)


def _train_n_steps(model, opt, sched, n, seed0=0):
    from fusion_embedding.losses import FusionContrastiveLoss
    loss_fn = FusionContrastiveLoss(model.cfg)
    last = -1
    for step in range(n):
        batch = make_batch(model.cfg, batch_size=4, seed=seed0 + step)
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        loss, _ = loss_fn(out["audio"], out["text"], out["logit_scale"])
        loss.backward(); opt.step(); sched.step()
        last = step
    return last


def test_resume_ckpt_roundtrip_restores_trainable_state():
    """Preemption-resilience gate: save mid-run, then load into FRESH objects and get back the
    exact resampler weights, logit_scale, scheduler LR position, optimizer momentum, and step+1."""
    import os
    import tempfile
    from fusion_embedding.train_stage1 import save_resume_ckpt, load_resume_ckpt

    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, max_steps=20)
    last = _train_n_steps(model, opt, sched, 5)                    # populate momentum + advance sched

    path = os.path.join(tempfile.mkdtemp(), "resume.pt")
    save_resume_ckpt(path, model, opt, sched, step=last, total_steps=20)
    saved_resampler = [p.detach().clone() for p in model.resampler.parameters()]
    saved_scale = model.logit_scale.detach().clone()
    saved_lr = sched.get_last_lr()[0]

    # fresh, differently-initialised objects — resume must overwrite them exactly
    model2 = build_tiny_model(cfg); opt2 = build_optimizer(model2, cfg)
    sched2 = build_scheduler(opt2, cfg, max_steps=20)
    start = load_resume_ckpt(path, model2, opt2, sched2, total_steps=20)

    assert start == last + 1                                       # resume AFTER last completed step
    for a, b in zip(model2.resampler.parameters(), saved_resampler):
        assert torch.equal(a.detach(), b)                         # exact resampler restore
    assert torch.equal(model2.logit_scale.detach(), saved_scale)  # temperature restored
    assert abs(sched2.get_last_lr()[0] - saved_lr) < 1e-9         # LR schedule position restored
    assert any(opt2.state.values())                               # optimizer momentum restored, not empty


def test_load_resume_ckpt_absent_or_config_mismatch_starts_fresh():
    """No file, or a checkpoint from a different total_steps, must return 0 (train from scratch)
    rather than silently resuming a mismatched run."""
    import os
    import tempfile
    from fusion_embedding.train_stage1 import save_resume_ckpt, load_resume_ckpt

    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg); opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, max_steps=20)

    assert load_resume_ckpt(os.path.join(tempfile.mkdtemp(), "nope.pt"), model, opt, sched,
                            total_steps=20) == 0                  # absent -> fresh
    path = os.path.join(tempfile.mkdtemp(), "r.pt")
    save_resume_ckpt(path, model, opt, sched, step=7, total_steps=20)
    assert load_resume_ckpt(path, model, opt, sched, total_steps=30) == 0   # different run -> fresh
    assert load_resume_ckpt(path, model, opt, sched, total_steps=20) == 8   # same run -> resume @ 8


def test_load_resume_ckpt_refuses_config_key_mismatch():
    """A resume must never cross A/B arms: same total_steps but a different config fingerprint
    (e.g. another d_resampler or lr arm sharing a run_tag by mistake) must start fresh."""
    import os
    import tempfile
    from fusion_embedding.train_stage1 import save_resume_ckpt, load_resume_ckpt

    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg); opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, max_steps=20)

    path = os.path.join(tempfile.mkdtemp(), "r.pt")
    save_resume_ckpt(path, model, opt, sched, step=7, total_steps=20, config_key="arm|d256|lr1e-4")
    assert load_resume_ckpt(path, model, opt, sched, total_steps=20,
                            config_key="arm|d384|lr1e-4") == 0    # different arm -> fresh
    assert load_resume_ckpt(path, model, opt, sched, total_steps=20,
                            config_key="arm|d256|lr1e-4") == 8    # same arm -> resume @ 8
    # legacy ckpt (no key) + keyed run must also refuse (never guess)
    save_resume_ckpt(path, model, opt, sched, step=7, total_steps=20)
    assert load_resume_ckpt(path, model, opt, sched, total_steps=20,
                            config_key="arm|d256|lr1e-4") == 0


def test_single_train_step_updates_only_connector():
    cfg = FusionConfig.tiny()
    model = build_tiny_model(cfg)
    from fusion_embedding.losses import FusionContrastiveLoss
    loss_fn = FusionContrastiveLoss(cfg)
    opt = build_optimizer(model, cfg)

    before_resampler = [p.detach().clone() for p in model.resampler.parameters()]
    before_scale = model.logit_scale.detach().clone()

    batch = make_batch(cfg, batch_size=6, seed=3)
    out = model(batch)
    loss, _ = loss_fn(out["audio"], out["text"], out["logit_scale"])
    loss.backward()
    opt.step()

    # connector + temp moved
    assert any(not torch.equal(b, a) for b, a in zip(before_resampler, model.resampler.parameters()))
    assert not torch.equal(before_scale, model.logit_scale.detach())
    # frozen base received no grad
    for comp in model.frozen_modules():
        for p in comp.parameters():
            assert p.grad is None


# ---------------------------------------------------------------------------- #
# Seed reproducibility (multi-seed A/B arms depend on this contract)
# ---------------------------------------------------------------------------- #
def _first_step_state(seed: int):
    """Build the tiny setup at ``seed`` and take one optimizer step; return the
    first-batch loss and a trainable-param fingerprint after the step."""
    from fusion_embedding.train_stage1 import build_tiny_training_setup

    cfg = FusionConfig.tiny(max_steps=4, d_resampler=32)
    s = build_tiny_training_setup(cfg, n_train=8, batch_size=8, seed=seed)
    opt = build_optimizer(s.model, cfg)
    batch = next(iter(s.train_loader))
    out = s.model(batch)
    loss, _metrics = s.loss_fn(out["audio"], out["text"], out["logit_scale"],
                               out.get("hard_neg_text"))
    loss.backward()
    opt.step()
    fp = torch.cat([p.detach().flatten() for p in s.model.trainable_parameters()])
    return float(loss), fp


def test_same_seed_reproduces_first_step():
    loss_a, fp_a = _first_step_state(seed=1)
    loss_b, fp_b = _first_step_state(seed=1)
    assert loss_a == loss_b
    assert torch.equal(fp_a, fp_b)


def test_different_seeds_diverge():
    loss_a, fp_a = _first_step_state(seed=1)
    loss_b, fp_b = _first_step_state(seed=2)
    # different init + shuffle: the first-step loss and the resulting trainables differ
    assert loss_a != loss_b
    assert not torch.equal(fp_a, fp_b)
