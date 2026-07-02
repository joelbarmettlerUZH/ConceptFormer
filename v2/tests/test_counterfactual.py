"""Pure graph-manipulation helpers behind the graph-faithfulness probes (swap/ablate/pool)."""

from __future__ import annotations

import random

from conceptformer.eval.counterfactual import (
    ablate_neighbor,
    answer_edge_property,
    build_swap_pool,
    pick_swap_target,
    swap_edge_neighbor,
)
from conceptformer.schemas import Edge, Entity, Subgraph


def _sg() -> Subgraph:
    return Subgraph(
        center=Entity(qid="Q42", label="Douglas Adams"),
        edges=[
            Edge(property_id="P106", property_label="occupation",
                 neighbor=Entity(qid="Q1", label="writer")),
            Edge(property_id="P19", property_label="birthplace",
                 neighbor=Entity(qid="Q2", label="Cambridge")),
        ],
        n_edges_total=2,
    )


def test_answer_edge_property_finds_the_relation() -> None:
    assert answer_edge_property(_sg(), "Q1") == "P106"
    assert answer_edge_property(_sg(), "Q2") == "P19"
    assert answer_edge_property(_sg(), "Q999") is None  # not a neighbor


def test_swap_replaces_only_the_answer_edge() -> None:
    new = Entity(qid="Q3", label="musician")
    out = swap_edge_neighbor(_sg(), "Q1", new)
    occ = next(e for e in out.edges if e.property_id == "P106")
    assert occ.neighbor.qid == "Q3" and occ.neighbor.label == "musician"  # swapped
    bp = next(e for e in out.edges if e.property_id == "P19")
    assert bp.neighbor.qid == "Q2"  # untouched
    assert _sg().edges[0].neighbor.qid == "Q1"  # original not mutated (model_copy)


def test_ablate_removes_only_that_neighbor() -> None:
    out = ablate_neighbor(_sg(), "Q1")
    assert [e.neighbor.qid for e in out.edges] == ["Q2"]  # answer edge gone, other kept
    assert len(_sg().edges) == 2  # original intact


def test_swap_pool_and_target_pick_type_plausible_false_neighbor() -> None:
    other = Subgraph(
        center=Entity(qid="Q50", label="Brian May"),
        edges=[Edge(property_id="P106", property_label="occupation",
                    neighbor=Entity(qid="Q3", label="musician"))],
        n_edges_total=1,
    )
    pool = build_swap_pool([_sg(), other])
    assert {n.qid for n in pool["P106"]} == {"Q1", "Q3"}  # both occupations pooled
    # For Q42's occupation edge (answer Q1), the only same-property, non-existing target is Q3.
    tgt = pick_swap_target(pool, "P106", _sg(), "Q1", random.Random(0))
    assert tgt is not None and tgt.qid == "Q3"


def test_swap_target_none_when_no_alternative() -> None:
    pool = build_swap_pool([_sg()])  # only Q1 has P106 → no FALSE alternative
    assert pick_swap_target(pool, "P106", _sg(), "Q1", random.Random(0)) is None
