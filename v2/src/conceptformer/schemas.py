"""Typed data schemas (pydantic v2).

These replace v1's dict-juggling + CSV/TAR HF builders. A ``Subgraph`` is the core
artifact the ConceptFormer encoder consumes; a ``QAExample`` is one training/eval row.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

QID = str  # Wikidata entity id, e.g. "Q42"
PID = str  # Wikidata property id, e.g. "P19"


class Entity(BaseModel):
    qid: QID
    label: str | None = None
    description: str | None = None
    # Popularity used for neighbor ranking. Pilot proxy = sitelink count;
    # swappable for PageRank at scale. None when unknown.
    rank: float | None = None


class Edge(BaseModel):
    """A truthy 1-hop edge: center --property--> neighbor."""

    property_id: PID
    property_label: str | None = None
    neighbor: Entity


class Subgraph(BaseModel):
    """An entity's 1-hop neighborhood, rank-sorted and capped."""

    center: Entity
    edges: list[Edge] = Field(default_factory=list)
    n_edges_total: int = 0  # before capping to max_neighbors
    capped: bool = False

    @property
    def neighbor_qids(self) -> list[QID]:
        return [e.neighbor.qid for e in self.edges]


class QAExample(BaseModel):
    """One question→answer row grounded in a subject entity's neighborhood."""

    source: str  # "popqa" | "entityquestions" | "cf-train"
    split: str  # "train" | "validation" | "test"

    subject_qid: QID
    relation: str  # human-readable relation / PopQA "prop"
    relation_id: PID | None = None

    question: str
    answer_qid: QID | None = None
    answer_labels: list[str] = Field(default_factory=list)  # gold + aliases

    popularity: float | None = None  # subject popularity (e.g. PopQA pageviews)
