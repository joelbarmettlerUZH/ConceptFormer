"""Graph-faithfulness probes — pure subgraph manipulations behind the "does it learn the graph,
not just compress text?" experiments.

Two interventions on a trained model's INPUT graph (then re-encode + generate):
- **counterfactual swap** — replace the answer edge's neighbor with a type-plausible but FALSE one;
  a graph-faithful model follows the swap to the false answer (a text-memoriser / a model leaning on
  the frozen LLM's parametric knowledge would not).
- **edge ablation** — drop the answer edge; the model should lose THAT answer while keeping others
  (separability), proving edges are encoded as a graph rather than one entangled text blob.

Pure (no model / GPU) so the manipulations are unit-tested in isolation; the CLI command
``cf-graph-faithfulness`` wires them to the encoder + greedy decode.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from conceptformer.schemas import Entity, Subgraph


def answer_edge_property(sg: Subgraph, answer_qid: str) -> str | None:
    """Property id of the edge whose neighbor is ``answer_qid`` (the edge the question targets)."""
    for e in sg.edges:
        if e.neighbor.qid == answer_qid:
            return e.property_id
    return None


def swap_edge_neighbor(sg: Subgraph, answer_qid: str, new_neighbor: Entity) -> Subgraph:
    """Counterfactual: replace the neighbor of every edge pointing at ``answer_qid`` with
    ``new_neighbor``, leaving all other edges (and the property labels) untouched."""
    edges = [
        e.model_copy(update={"neighbor": new_neighbor}) if e.neighbor.qid == answer_qid else e
        for e in sg.edges
    ]
    return sg.model_copy(update={"edges": edges})


def ablate_neighbor(sg: Subgraph, neighbor_qid: str) -> Subgraph:
    """Drop every edge pointing at ``neighbor_qid`` (e.g. the answer edge), keeping the rest."""
    edges = [e for e in sg.edges if e.neighbor.qid != neighbor_qid]
    return sg.model_copy(update={"edges": edges})


def build_swap_pool(subgraphs: Iterable[Subgraph]) -> dict[str, list[Entity]]:
    """``property_id -> distinct neighbor entities`` seen with that property, for type-plausible
    swaps (so a swapped 'occupation' is another real occupation, not a random country)."""
    pool: dict[str, dict[str, Entity]] = {}
    for sg in subgraphs:
        for e in sg.edges:
            if e.neighbor.label:  # only swap in things with a surface form to generate/match
                pool.setdefault(e.property_id, {})[e.neighbor.qid] = e.neighbor
    return {pid: list(by_qid.values()) for pid, by_qid in pool.items()}


def pick_swap_target(
    pool: dict[str, list[Entity]],
    property_id: str,
    sg: Subgraph,
    answer_qid: str,
    rng: random.Random,
) -> Entity | None:
    """A FALSE neighbor for the answer edge: same property (type-plausible), not the true answer,
    and not already a neighbor of this entity (so the swap is genuinely counterfactual). None if no
    candidate exists."""
    existing = set(sg.neighbor_qids)
    candidates = [
        n for n in pool.get(property_id, []) if n.qid != answer_qid and n.qid not in existing
    ]
    return rng.choice(candidates) if candidates else None


def matches(prediction: str, entity: Entity, aliases: Sequence[str] = ()) -> bool:
    """Alias-aware: does the generation name ``entity`` (its label, plus any extra aliases)?"""
    from conceptformer.generate.signal import answer_ok

    forms = [entity.label, *aliases] if entity.label else list(aliases)
    return answer_ok(prediction, [f for f in forms if f])
