"""Offline tests for subgraph featurization + collation (stub embedder; no model load)."""

import pytest

torch = pytest.importorskip("torch")

from conceptformer.model.featurizer import (  # noqa: E402
    collate_features,
    edge_label_pairs,
    featurize_subgraph,
)
from conceptformer.schemas import Edge, Entity, Subgraph  # noqa: E402

D = 8  # stub embedding dim


class _StubEmbedder:
    """Deterministic per-string vector so tests can assert content, not just shapes."""

    def __call__(self, texts):
        out = torch.zeros(len(texts), D)
        for i, t in enumerate(texts):
            out[i] = float((abs(hash(t)) % 1000) - 500) / 100.0
        return out


def _sg(qid="Q42", label="Douglas Adams", edges=2) -> Subgraph:
    es = [
        Edge(property_id=f"P{i}", property_label=f"rel{i}",
             neighbor=Entity(qid=f"Qn{i}", label=f"nbr{i}"))
        for i in range(edges)
    ]
    return Subgraph(center=Entity(qid=qid, label=label), edges=es, n_edges_total=edges)


def test_edge_label_pairs_fall_back_to_ids():
    sg = Subgraph(
        center=Entity(qid="Q1", label="X"),
        edges=[Edge(property_id="P9", property_label=None,
                    neighbor=Entity(qid="Q9", label=None))],
    )
    assert edge_label_pairs(sg) == [("P9", "Q9")]


def test_featurize_shapes_and_concat_fusion():
    feats = featurize_subgraph(_sg(edges=3), _StubEmbedder())
    assert feats.edge_features.shape == (3, 2 * D)  # concat(property, neighbor)
    assert feats.center.shape == (D,)
    assert feats.n_edges == 3


def test_featurize_raises_on_empty_neighborhood():
    empty = Subgraph(center=Entity(qid="Q1", label="X"), edges=[])
    with pytest.raises(ValueError, match="no edges"):
        featurize_subgraph(empty, _StubEmbedder())


def test_collate_pads_and_masks_ragged_batch():
    emb = _StubEmbedder()
    items = [featurize_subgraph(_sg(qid="Q1", edges=2), emb),
             featurize_subgraph(_sg(qid="Q2", edges=5), emb)]
    edge_features, mask, center = collate_features(items)
    assert edge_features.shape == (2, 5, 2 * D)  # padded to N_max=5
    assert center.shape == (2, D)
    assert mask[0].tolist() == [True, True, False, False, False]  # row 0 has 2 real edges
    assert mask[1].all()
    # padded rows are zero-filled
    assert torch.count_nonzero(edge_features[0, 2:]) == 0


def test_collate_empty_raises():
    with pytest.raises(ValueError, match="empty batch"):
        collate_features([])
