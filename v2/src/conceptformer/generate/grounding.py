"""Programmatic grounding / quality checks for generated questions.

We don't trust the generator's self-report — we verify against the actual neighborhood:
the answer must match a neighbor label, the question should mention the subject, and we
de-duplicate. Reuses the eval word-boundary matcher.
"""

from __future__ import annotations

from conceptformer.eval.metrics import word_boundary_match
from conceptformer.generate.schema import GeneratedQuestion
from conceptformer.schemas import Entity, Subgraph


def neighbor_labels(sg: Subgraph) -> list[str]:
    return [e.neighbor.label for e in sg.edges if e.neighbor.label]


def grounded_neighbor(q: GeneratedQuestion, sg: Subgraph) -> Entity | None:
    """The neighbor whose label the answer references (the answer's real entity), or None.

    Lets later stages record the answer's qid + canonical label, not just Gemma's verbatim
    string — the basis for alias-aware tiering.
    """
    for edge in sg.edges:
        if edge.neighbor.label and word_boundary_match(q.answer, [edge.neighbor.label]):
            return edge.neighbor
    return None


def answer_is_grounded(q: GeneratedQuestion, sg: Subgraph) -> bool:
    """The answer must reference at least one real neighbor label."""
    return word_boundary_match(q.answer, neighbor_labels(sg))


def mentions_subject(q: GeneratedQuestion, sg: Subgraph) -> bool:
    label = sg.center.label
    return word_boundary_match(q.question, [label]) if label else True


def quality_flags(q: GeneratedQuestion, sg: Subgraph) -> dict[str, bool]:
    return {
        "grounded": answer_is_grounded(q, sg),
        "mentions_subject": mentions_subject(q, sg),
    }


def passes(q: GeneratedQuestion, sg: Subgraph) -> bool:
    flags = quality_flags(q, sg)
    return flags["grounded"] and flags["mentions_subject"]
