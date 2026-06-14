"""Tests for entity quality filtering (Wikimedia-junk + thinness)."""

from conceptformer.data.quality import is_usable_entity, is_wikimedia_internal
from conceptformer.schemas import Edge, Entity, Subgraph


def _sg(p31_target: str | None, n_extra_edges: int) -> Subgraph:
    edges = []
    if p31_target is not None:
        edges.append(Edge(property_id="P31", neighbor=Entity(qid=p31_target, label="type")))
    for i in range(n_extra_edges):
        edges.append(Edge(property_id="P106", neighbor=Entity(qid=f"Q{i}", label=f"n{i}")))
    return Subgraph(center=Entity(qid="Q1", label="x"), edges=edges, n_edges_total=len(edges))


def test_wikimedia_category_is_internal():
    assert is_wikimedia_internal(_sg("Q4167836", 1)) is True  # Wikimedia category
    assert is_wikimedia_internal(_sg("Q4167410", 1)) is True  # disambiguation page


def test_real_entity_is_not_internal():
    assert is_wikimedia_internal(_sg("Q5", 8)) is False  # instance of human


def test_usable_requires_real_type_and_enough_edges():
    assert is_usable_entity(_sg("Q5", 8)) is True  # human, 9 edges
    assert is_usable_entity(_sg("Q5", 2)) is False  # too thin (3 edges < 5)
    assert is_usable_entity(_sg("Q4167836", 20)) is False  # category, even if many edges


def test_unlabeled_entity_is_unusable():
    sg = _sg("Q5", 8)
    unlabeled = sg.model_copy(update={"center": sg.center.model_copy(update={"label": None})})
    assert is_usable_entity(unlabeled) is False  # no name to ask questions about
