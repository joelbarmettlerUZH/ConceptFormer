"""Offline tests for the pure backbone/assembly pieces (no model load)."""

import pytest

torch = pytest.importorskip("torch")

from conceptformer.model.backbone import pool_label_embeddings  # noqa: E402
from conceptformer.model.conceptformer import ConceptFormer  # noqa: E402
from conceptformer.train.forcing import gather_path_logits, path_predicting_indices  # noqa: E402


def test_pool_label_embeddings_mean_pools_tokens():
    embedding = torch.tensor([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0], [10.0, 10.0]])
    pooled = pool_label_embeddings([[1, 2], [3], []], embedding)
    assert torch.allclose(pooled[0], torch.tensor([3.0, 6.0]))  # mean of rows 1,2
    assert torch.allclose(pooled[1], torch.tensor([10.0, 10.0]))  # single token
    assert torch.count_nonzero(pooled[2]) == 0  # empty label → zero vector


def test_path_predicting_indices():
    assert path_predicting_indices(context_len=5, path_len=3) == [4, 5, 6]
    assert path_predicting_indices(context_len=1, path_len=1) == [0]
    with pytest.raises(ValueError, match="context_len must be"):
        path_predicting_indices(context_len=0, path_len=2)


def test_gather_path_logits_aligns_different_prefixes():
    # two rows, different context lengths + path lengths, padded to m_max
    logits = torch.arange(2 * 6 * 1, dtype=torch.float).reshape(2, 6, 1)
    out, mask = gather_path_logits(logits, context_lens=[2, 4], path_lens=[3, 2])
    # row 0: indices 1,2,3 → logits values 1,2,3 ; row 1: indices 3,4 → 9,10 (+pad)
    assert out[0, :, 0].tolist() == [1.0, 2.0, 3.0]
    assert out[1, :2, 0].tolist() == [9.0, 10.0]
    assert mask[0].tolist() == [True, True, True]
    assert mask[1].tolist() == [True, True, False]


def test_conceptformer_is_zero_at_init_then_trainable():
    torch.manual_seed(0)
    cf = ConceptFormer(d_in=16, d_llm=24, k=4, d_model=32, n_layers=2, n_heads=4).eval()
    feats = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = cf(feats, mask)
    assert out.shape == (2, 4, 24)
    assert torch.count_nonzero(out) == 0  # gate=0 → concepts==0 → student == frozen model at init
    # after nudging the gate open, concepts become non-zero and depend on the neighborhood
    with torch.no_grad():
        cf.gate.gate.fill_(1.0)
    assert torch.count_nonzero(cf(feats, mask)) > 0
