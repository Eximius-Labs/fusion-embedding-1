"""qb_norm — QB-Norm/DIS test-time hubness correction (Bogolin et al., CVPR 2022)."""
import pytest
import torch

from fusion_embedding.train_stage1 import qb_norm


def _hub_world():
    """Gallery item 0 is a hub: every bank query loves it. Test query's TRUE item is 1,
    but the hub edges it out in raw similarity."""
    bank = torch.tensor([[0.9, 0.1, 0.0],
                         [0.8, 0.0, 0.2],
                         [0.9, 0.2, 0.1]])                     # [B=3, G=3] all top-1 -> item 0
    q = torch.tensor([[0.7, 0.65, 0.1]])                       # raw top-1 = hub item 0
    return q, bank


def test_dis_demotes_hub_and_recovers_true_item():
    q, bank = _hub_world()
    assert q.argmax(dim=1).item() == 0                          # raw retrieval hits the hub
    out = qb_norm(q, bank, beta=20.0, mode="dis")
    assert out.argmax(dim=1).item() == 1                        # corrected: true item wins


def test_dis_leaves_non_hub_queries_untouched():
    _, bank = _hub_world()
    q = torch.tensor([[0.1, 0.2, 0.9]])                        # top-1 = item 2, never bank top-1
    out = qb_norm(q, bank, beta=20.0, mode="dis")
    assert torch.equal(out, q)                                  # raw scores preserved exactly


def test_is_mode_normalizes_all_queries():
    q, bank = _hub_world()
    out = qb_norm(q, bank, beta=20.0, mode="is")
    assert out.shape == q.shape
    assert not torch.equal(out, q)
    assert out.argmax(dim=1).item() == 1


def test_rank_of_true_item_never_worse_for_single_hub_query():
    q, bank = _hub_world()
    out = qb_norm(q, bank, beta=20.0, mode="dis")
    true_idx = 1
    raw_rank = int((q[0] > q[0, true_idx]).sum()) + 1
    new_rank = int((out[0] > out[0, true_idx]).sum()) + 1
    assert new_rank <= raw_rank


def test_shape_and_mode_validation():
    q, bank = _hub_world()
    with pytest.raises(ValueError, match="shape"):
        qb_norm(q, bank[:, :2])
    with pytest.raises(ValueError, match="mode"):
        qb_norm(q, bank, mode="bogus")
