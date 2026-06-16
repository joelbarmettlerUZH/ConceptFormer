"""gate_mode: 'tanh' (default gate) vs 'none' (no gate, zero-init output projection)."""

import torch

from conceptformer.model.conceptformer import ConceptFormer

D_IN, D_LLM, K = 32, 16, 4


def _batch():
    feats = torch.randn(2, 5, D_IN)
    mask = torch.ones(2, 5, dtype=torch.bool)
    return feats, mask


def test_tanh_mode_has_gate_and_zero_concepts_at_init():
    m = ConceptFormer(D_IN, D_LLM, K, d_model=16, n_layers=1, gate_mode="tanh")
    assert m.gate is not None
    feats, mask = _batch()
    out = m(feats, mask)
    assert out.shape == (2, K, D_LLM)
    # zero-init gate => concepts are exactly 0 at init (capability preserved)
    assert torch.allclose(out, torch.zeros_like(out))


def test_none_mode_drops_gate_but_still_zero_concepts_at_init():
    m = ConceptFormer(D_IN, D_LLM, K, d_model=16, n_layers=1, gate_mode="none")
    assert m.gate is None
    # output projection is zero-init => concepts start at 0 without any gate
    assert torch.allclose(m.encoder.out_proj.weight, torch.zeros_like(m.encoder.out_proj.weight))
    assert torch.allclose(m.encoder.out_proj.bias, torch.zeros_like(m.encoder.out_proj.bias))
    feats, mask = _batch()
    out = m(feats, mask)
    assert out.shape == (2, K, D_LLM)
    assert torch.allclose(out, torch.zeros_like(out))


def test_none_mode_can_learn_nonzero_concepts():
    m = ConceptFormer(D_IN, D_LLM, K, d_model=16, n_layers=1, gate_mode="none")
    torch.nn.init.normal_(m.encoder.out_proj.weight, std=0.1)  # simulate a training step
    feats, mask = _batch()
    out = m(feats, mask)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_unknown_gate_mode_raises():
    try:
        ConceptFormer(D_IN, D_LLM, K, gate_mode="bogus")
    except ValueError as e:
        assert "gate_mode" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown gate_mode")
