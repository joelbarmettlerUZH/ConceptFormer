"""Offline tests for concept-token injection primitives (gate, splice, packing, positions)."""

import pytest

torch = pytest.importorskip("torch")

from conceptformer.model.injection import (  # noqa: E402
    ConceptGate,
    build_position_ids,
    pack_embeddings,
    splice_sequence,
)


def test_gate_is_zero_at_init():
    gate = ConceptGate(k=4)
    concepts = torch.randn(2, 4, 8)
    out = gate(concepts)
    # tanh(0)=0 → injected vectors are zero → student == frozen model at step 0
    assert torch.count_nonzero(out) == 0
    assert gate.gate.shape == (4,)


def test_gate_scales_per_token_and_passes_gradient():
    gate = ConceptGate(k=3)
    with torch.no_grad():
        gate.gate.copy_(torch.tensor([100.0, 0.0, -100.0]))  # saturate +, off, saturate -
    concepts = torch.ones(1, 3, 5)
    out = gate(concepts)
    assert torch.allclose(out[0, 0], torch.ones(5), atol=1e-3)  # tanh(100)≈1
    assert torch.count_nonzero(out[0, 1]) == 0  # tanh(0)=0
    assert torch.allclose(out[0, 2], -torch.ones(5), atol=1e-3)  # tanh(-100)≈-1
    gate2 = ConceptGate(k=2)
    gate2(torch.randn(1, 2, 4)).sum().backward()
    assert gate2.gate.grad is not None and gate2.gate.grad.abs().sum() > 0


def test_splice_orders_prefix_concepts_suffix():
    prefix, concepts, suffix = torch.zeros(3, 8), torch.ones(4, 8), torch.full((2, 8), 2.0)
    seq = splice_sequence(prefix, concepts, suffix)
    assert seq.shape == (9, 8)  # 3 + 4 + 2 — concept block in the facts slot
    assert (seq[:3] == 0).all() and (seq[3:7] == 1).all() and (seq[7:] == 2).all()


def test_pack_right_pads_and_masks():
    seqs = [torch.ones(3, 5), torch.full((6, 5), 2.0)]
    inputs_embeds, attention_mask = pack_embeddings(seqs)
    assert inputs_embeds.shape == (2, 6, 5)
    assert attention_mask[0].tolist() == [1, 1, 1, 0, 0, 0]
    assert attention_mask[1].tolist() == [1] * 6
    assert torch.count_nonzero(inputs_embeds[0, 3:]) == 0  # padding is zero-filled


def test_pack_empty_raises():
    with pytest.raises(ValueError, match="empty batch"):
        pack_embeddings([])


def test_position_ids_are_contiguous_and_padding_safe():
    _, attention_mask = pack_embeddings([torch.ones(3, 4), torch.ones(5, 4)])
    pos = build_position_ids(attention_mask)
    assert pos[1].tolist() == [0, 1, 2, 3, 4]  # full row → plain arange (contiguous, unit-step)
    assert pos[0, :3].tolist() == [0, 1, 2]  # real tokens contiguous from 0
    assert pos[0, 3:].tolist() == [0, 0]  # padded positions parked at 0 (ignored)
