"""Verbalize a Subgraph into text — the graph-in-context for the teacher / RAG baseline.

Neighbors are grouped by property (preserving the rank order from the subgraph) so the
context is compact, e.g.::

    Facts about Douglas Adams:
    - occupation: writer, science fiction writer, novelist
    - place of birth: Cambridge
    - country of citizenship: United Kingdom
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

from conceptformer.schemas import Subgraph


def verbalize(sg: Subgraph, *, max_neighbors: int | None = None) -> str:
    """Render a subgraph as grouped fact lines.

    ``max_neighbors`` caps the number of edges rendered (the highest-ranked ones, since
    ``sg.edges`` is rank-sorted). This is a teacher/RAG-side context-budget knob — text in a
    finite context window can't hold an unbounded neighborhood — while the snapshot itself
    stays COMPLETE. (ConceptFormer's soft tokens have no such limit; that asymmetry is a
    feature, not a constraint we put on the data.)
    """
    edges = sg.edges if max_neighbors is None else sg.edges[:max_neighbors]
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for edge in edges:
        prop = edge.property_label or edge.property_id
        value = edge.neighbor.label or edge.neighbor.qid
        groups.setdefault(prop, []).append(value)

    header = f"Facts about {sg.center.label or sg.center.qid}:"
    lines = [f"- {prop}: {', '.join(values)}" for prop, values in groups.items()]
    return "\n".join([header, *lines])


def verbalize_budgeted(sg: Subgraph, count_tokens: Callable[[str], int], budget: int) -> str:
    """Verbalize as many top-ranked neighbors as fit within ``budget`` tokens.

    This is the principled RAG/teacher context bound — a property of the LLM's context window,
    not an arbitrary neighbor count. The common case (neighborhood already fits) costs one
    token count; only overflowing subjects pay a binary search over the rank-ordered edges.
    """
    full = verbalize(sg)
    if not sg.edges or count_tokens(full) <= budget:
        return full
    lo, hi = 0, len(sg.edges)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(verbalize(sg, max_neighbors=mid)) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return verbalize(sg, max_neighbors=lo)
