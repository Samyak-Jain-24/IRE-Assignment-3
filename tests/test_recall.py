import faiss
import numpy as np
import pytest

from annbench.metrics import exact_sq_dists, ground_truth, recall_at_k


def gt_single(ids, dists):
    return np.array([ids], dtype=np.int64), np.array([dists], dtype=np.float64)


def test_perfect_result_gives_one():
    gt_ids, gt_d = gt_single([5, 3, 9, 1], [1.0, 2.0, 3.0, 4.0])
    assert recall_at_k([[5, 3, 9]], gt_ids, gt_d, k=3)[0] == 1.0


def test_order_inside_top_k_does_not_matter():
    gt_ids, gt_d = gt_single([5, 3, 9, 1], [1.0, 2.0, 3.0, 4.0])
    assert recall_at_k([[9, 5, 3]], gt_ids, gt_d, k=3)[0] == 1.0


def test_disjoint_result_gives_zero():
    gt_ids, gt_d = gt_single([5, 3, 9, 1], [1.0, 2.0, 3.0, 4.0])
    assert recall_at_k([[100, 101, 102]], gt_ids, gt_d, k=3)[0] == 0.0


def test_partial_result():
    gt_ids, gt_d = gt_single([5, 3, 9, 1], [1.0, 2.0, 3.0, 4.0])
    assert recall_at_k([[5, 1, 77]], gt_ids, gt_d, k=3)[0] == pytest.approx(1 / 3)  # id 1 is 4th, a miss


def test_tie_at_kth_neighbour_is_not_a_mistake():
    # ids 9 and 1 are both at distance 3.0; with k=3 either may legitimately be third.
    gt_ids, gt_d = gt_single([5, 3, 9, 1, 8], [1.0, 2.0, 3.0, 3.0, 4.0])
    assert recall_at_k([[5, 3, 1]], gt_ids, gt_d, k=3)[0] == 1.0
    assert recall_at_k([[5, 3, 9]], gt_ids, gt_d, k=3)[0] == 1.0
    assert recall_at_k([[5, 3, 8]], gt_ids, gt_d, k=3)[0] == pytest.approx(2 / 3)  # 8 is strictly farther


def test_ties_cannot_push_recall_above_one():
    gt_ids, gt_d = gt_single([1, 2, 3, 4, 5], [1.0, 1.0, 1.0, 1.0, 9.0])
    assert recall_at_k([[4, 3]], gt_ids, gt_d, k=2)[0] == 1.0
    assert recall_at_k([[4, 3, 2, 1]], gt_ids, gt_d, k=2)[0] == 1.0


def test_tie_tolerance_absorbs_float_rounding():
    gt_ids, gt_d = gt_single([5, 3, 9, 1, 8], [1.0, 2.0, 3.0, 3.0 * (1 + 1e-9), 4.0])
    assert recall_at_k([[5, 3, 1]], gt_ids, gt_d, k=3)[0] == 1.0


def test_duplicates_counted_once_and_missing_ids_are_misses():
    gt_ids, gt_d = gt_single([5, 3, 9, 1], [1.0, 2.0, 3.0, 4.0])
    assert recall_at_k([[5, 5, 5]], gt_ids, gt_d, k=3)[0] == pytest.approx(1 / 3)
    assert recall_at_k([[5, -1, -1]], gt_ids, gt_d, k=3)[0] == pytest.approx(1 / 3)


def test_only_first_k_retrieved_are_used():
    gt_ids, gt_d = gt_single([5, 3, 9, 1], [1.0, 2.0, 3.0, 4.0])
    assert recall_at_k([[100, 101, 102, 5, 3, 9]], gt_ids, gt_d, k=3)[0] == 0.0


def test_per_query_vector_output():
    gt_ids = np.array([[0, 1, 2], [3, 4, 5]])
    gt_d = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    r = recall_at_k([[0, 1], [3, 9]], gt_ids, gt_d, k=2)
    np.testing.assert_allclose(r, [1.0, 0.5])


def test_ties_overflowing_ground_truth_list_raise():
    gt_ids, gt_d = gt_single([1, 2, 3], [1.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        recall_at_k([[1, 2]], gt_ids, gt_d, k=2)


def test_exact_search_on_integer_data_with_ties_has_recall_one():
    # Integer-valued vectors (like SIFT) create many exact distance ties.
    rng = np.random.default_rng(0)
    base = rng.integers(0, 3, size=(2000, 8)).astype(np.float32)
    queries = rng.integers(0, 3, size=(50, 8)).astype(np.float32)
    flat = faiss.IndexFlatL2(8)
    flat.add(base)
    gt_ids, gt_d = ground_truth(flat, base, queries, k=100, extra=200)
    _, I = flat.search(queries, 10)
    assert np.all(recall_at_k(I, gt_ids, gt_d, k=10) == 1.0)
    # a different, equally valid choice among the tied points also scores 1.0
    reversed_choice = np.empty_like(I)
    for i in range(len(I)):
        tied = gt_ids[i][gt_d[i] <= gt_d[i, 9]]
        strictly_closer = gt_ids[i][gt_d[i] < gt_d[i, 9]]
        rest = [t for t in tied[::-1] if t not in set(strictly_closer)]
        reversed_choice[i] = np.concatenate([strictly_closer, rest])[:10]
    assert np.all(recall_at_k(reversed_choice, gt_ids, gt_d, k=10) == 1.0)


def test_exact_sq_dists_matches_numpy():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(100, 5)).astype(np.float32)
    q = rng.normal(size=(3, 5)).astype(np.float32)
    ids = np.array([[0, 1], [2, -1], [99, 50]])
    d = exact_sq_dists(q, base, ids)
    assert d[1, 1] == np.inf
    assert d[2, 0] == pytest.approx(((base[99].astype(float) - q[2]) ** 2).sum())
