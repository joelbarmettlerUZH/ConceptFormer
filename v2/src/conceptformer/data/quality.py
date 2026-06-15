"""Entity quality filtering for CF-Train.

The danker pool is sitelinked entities, but its deep tail includes Wikimedia-internal pages
(categories, disambiguation/list pages, templates) — they have sitelinks but aren't real
entities and make useless QA targets. We drop them by their ``instance of`` (P31) type, and
also drop entities too thin to generate diverse questions. Applied at snapshot time, where the
neighborhood (and thus P31) is available.
"""

from __future__ import annotations

from conceptformer.schemas import Subgraph

P_INSTANCE_OF = "P31"

# P31 values marking non-content Wikimedia pages.
WIKIMEDIA_INTERNAL_TYPES = frozenset(
    {
        "Q4167836",  # Wikimedia category
        "Q4167410",  # Wikimedia disambiguation page
        "Q13406463",  # Wikimedia list article
        "Q11266439",  # Wikimedia template
        "Q4663903",  # Wikimedia portal
        "Q15184295",  # Wikimedia module
        "Q11753321",  # Wikimedia navigational template
        "Q22808320",  # Wikimedia human-name disambiguation page
    }
)


def is_wikimedia_internal(sg: Subgraph) -> bool:
    return any(
        e.property_id == P_INSTANCE_OF and e.neighbor.qid in WIKIMEDIA_INTERNAL_TYPES
        for e in sg.edges
    )


def is_usable_entity(sg: Subgraph, *, min_edges: int = 5) -> bool:
    """A CF-Train subject must be a labeled real entity with enough facts to ask about."""
    return (
        bool(sg.center.label)  # need a name to ask questions about
        and len(sg.edges) >= min_edges
        and not is_wikimedia_internal(sg)
    )
