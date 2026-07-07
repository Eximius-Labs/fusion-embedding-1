import pytest
import torch

from fusion_embedding.gallery import select_gallery


def _sim_diagonal_dominant(n=6):
    # Diagonal is the strongest -> every true pair should be rank 1.
    return torch.eye(n) * 2.0 + torch.rand(n, n) * 0.1


def test_perfect_diagonal_all_rank1_and_hit():
    sim = torch.eye(5) * 10.0
    rows = select_gallery(sim, k=3)
    assert len(rows) == 5
    for r in rows:
        assert r["true_rank"] == 1
        assert r["hit_at_k"] is True
        assert r["topk"][0] == r["true_idx"]


def test_true_rank_counts_images_scoring_higher():
    # Query 0: scores put image 2 highest, then 0 (the true one) second.
    sim = torch.tensor([[0.5, 0.1, 0.9, 0.0]])
    rows = select_gallery(sim, k=2, true_idx=torch.tensor([0]))
    assert rows[0]["true_rank"] == 2          # exactly one image (idx 2) scores higher
    assert rows[0]["topk"] == [2, 0]
    assert rows[0]["hit_at_k"] is True         # true idx 0 is within top-2


def test_miss_when_true_outside_topk():
    sim = torch.tensor([[0.9, 0.8, 0.7, 0.05]])  # true idx 3 is dead last
    rows = select_gallery(sim, k=2, true_idx=torch.tensor([3]))
    assert rows[0]["true_rank"] == 4
    assert rows[0]["hit_at_k"] is False
    assert 3 not in rows[0]["topk"]


def test_n_queries_truncates_in_order():
    sim = _sim_diagonal_dominant(8)
    rows = select_gallery(sim, k=3, n_queries=3)
    assert [r["query"] for r in rows] == [0, 1, 2]


def test_custom_true_idx_non_diagonal():
    sim = torch.zeros(2, 4)
    sim[0, 1] = 5.0   # query 0's true image is 1
    sim[1, 3] = 5.0   # query 1's true image is 3
    rows = select_gallery(sim, k=1, true_idx=torch.tensor([1, 3]))
    assert rows[0]["true_rank"] == 1 and rows[0]["topk"] == [1]
    assert rows[1]["true_rank"] == 1 and rows[1]["topk"] == [3]


def test_rejects_bad_shapes_and_k():
    with pytest.raises(ValueError):
        select_gallery(torch.zeros(3), k=1)              # not 2-D
    with pytest.raises(ValueError):
        select_gallery(torch.zeros(2, 3), k=9)           # k > n images
    with pytest.raises(ValueError):
        select_gallery(torch.zeros(2, 3), k=1, true_idx=torch.tensor([0]))  # len mismatch
