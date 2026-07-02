"""Offline tests for question-aware retrieval ranking (eval/retrieval.py)."""

import torch

from conceptformer.eval.retrieval import rank_edges_by_question, summary_prompt
from conceptformer.schemas import Edge, Entity, Subgraph


class _StubEmbedder:
    """Maps known strings to fixed unit-ish vectors so cosine ranking is hand-checkable."""

    def __init__(self, table: dict[str, list[float]], default: list[float]) -> None:
        self.table = table
        self.default = default

    def __call__(self, texts) -> torch.Tensor:
        return torch.tensor([self.table.get(t, self.default) for t in texts])


def _sg() -> Subgraph:
    return Subgraph(
        center=Entity(qid="Q1", label="X"),
        edges=[
            Edge(property_id="P19", property_label="place of birth",
                 neighbor=Entity(qid="Q2", label="Cambridge")),
            Edge(property_id="P106", property_label="occupation",
                 neighbor=Entity(qid="Q3", label="writer")),
        ],
    )


def test_question_relevant_edge_ranked_first():
    embedder = _StubEmbedder(
        {
            "what does X do for a living?": [1.0, 0.0],
            "occupation: writer": [0.9, 0.1],  # aligned with the question
            "place of birth: Cambridge": [0.0, 1.0],  # orthogonal
        },
        default=[0.5, 0.5],
    )
    ranked = rank_edges_by_question(_sg(), "what does X do for a living?", embedder)
    assert [e.property_id for e in ranked.edges] == ["P106", "P19"]
    # Original subgraph untouched (model_copy, not in-place).
    assert [e.property_id for e in _sg().edges] == ["P19", "P106"]


def test_empty_subgraph_passthrough():
    sg = Subgraph(center=Entity(qid="Q1"))
    embedder = _StubEmbedder({}, default=[1.0, 0.0])
    assert rank_edges_by_question(sg, "q", embedder) is sg


def test_summary_prompt_mentions_budget_and_entity():
    system, user = summary_prompt("Facts about X:\n- a: b", "X", 16)
    assert "16 tokens" in user and "X" in user
    assert system  # non-empty system prompt
