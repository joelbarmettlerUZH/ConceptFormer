"""Offline tests for the untrained injection baseline (model/baselines.py)."""

import torch

from conceptformer.model.baselines import TopKMeanEdgeBaseline


def _features(rows: list[list[float]], d: int = 2) -> torch.Tensor:
    # Edge feature = concat(property_vec, neighbor_vec), both length d, encoded from a scalar
    # marker so tests can identify which edge landed in which slot.
    out = torch.zeros((1, len(rows), 2 * d))
    for i, (p, n) in enumerate(rows):
        out[0, i, :d] = p
        out[0, i, d:] = n
    return out


def test_top_k_slots_are_mean_of_property_and_neighbor_halves():
    feats = _features([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    out = TopKMeanEdgeBaseline(k=2)(feats, mask)
    assert out.shape == (1, 2, 2)
    assert torch.allclose(out[0, 0], torch.full((2,), 3.0))  # (2+4)/2 for edge 0
    assert torch.allclose(out[0, 1], torch.full((2,), 7.0))  # (6+8)/2 for edge 1


def test_short_neighborhoods_fill_remaining_slots_with_valid_centroid():
    feats = _features([[2.0, 4.0], [100.0, 100.0]])  # second row is padding
    mask = torch.tensor([[True, False]])
    out = TopKMeanEdgeBaseline(k=3)(feats, mask)
    # Slot 0 = the single real edge; slots 1-2 = centroid over VALID edges only (= same value
    # here), never the padded row's garbage.
    for slot in range(3):
        assert torch.allclose(out[0, slot], torch.full((2,), 3.0))


def test_padding_beyond_k_never_leaks():
    feats = _features([[2.0, 4.0], [999.0, 999.0], [999.0, 999.0]])
    mask = torch.tensor([[True, False, False]])
    out = TopKMeanEdgeBaseline(k=2)(feats, mask)
    assert out.abs().max() < 10  # padded 999s excluded from both slots and centroid


def test_batch_rows_are_independent():
    a = _features([[2.0, 4.0]])
    b = _features([[10.0, 20.0]])
    feats = torch.cat([a, b], dim=0)
    mask = torch.ones((2, 1), dtype=torch.bool)
    out = TopKMeanEdgeBaseline(k=1)(feats, mask)
    assert torch.allclose(out[0, 0], torch.full((2,), 3.0))
    assert torch.allclose(out[1, 0], torch.full((2,), 15.0))


def test_zero_parameters():
    assert sum(p.numel() for p in TopKMeanEdgeBaseline(k=4).parameters()) == 0
