"""Answer-in-graph coverage audit.

The core premise of ConceptFormer: the answer to an entity-centric question is reachable
in the subject's 1-hop neighborhood. This quantifies that on a benchmark + snapshot:

- ``subject_resolved``  — subject had a snapshot at all
- ``answer_in_graph``   — gold answer QID is among the (capped) neighbors
- ``answer_via_relation`` — answer reachable via the question's specific relation/property

A high ``answer_in_graph`` rate is the sanity check that the benchmark is in scope; the
gap to ``answer_via_relation`` flags relations our PID mapping or the truthy graph misses.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from conceptformer.data.snapshot import iter_subgraphs
from conceptformer.schemas import QAExample, Subgraph


def audit_coverage(examples: Sequence[QAExample], snapshot_dir: Path) -> dict:
    """Compute answer-in-graph coverage of ``examples`` against a snapshot directory."""
    by_qid: dict[str, Subgraph] = {sg.center.qid: sg for sg in iter_subgraphs(snapshot_dir)}

    total = len(examples)
    resolved = in_graph = via_relation = with_answer_qid = 0
    capped_total = 0
    neighbor_counts: list[int] = []

    for ex in examples:
        sg = by_qid.get(ex.subject_qid)
        if sg is None:
            continue
        resolved += 1
        neighbor_counts.append(len(sg.edges))
        capped_total += int(sg.capped)
        if ex.answer_qid is None:
            continue
        with_answer_qid += 1
        hits = [e for e in sg.edges if e.neighbor.qid == ex.answer_qid]
        if hits:
            in_graph += 1
            if ex.relation_id and any(e.property_id == ex.relation_id for e in hits):
                via_relation += 1

    def pct(n: int, d: int) -> float:
        return round(100.0 * n / d, 2) if d else 0.0

    avg_neighbors = (
        round(sum(neighbor_counts) / len(neighbor_counts), 1) if neighbor_counts else 0.0
    )
    return {
        "examples": total,
        "subject_resolved": resolved,
        "subject_resolved_pct": pct(resolved, total),
        "examples_with_answer_qid": with_answer_qid,
        "answer_in_graph": in_graph,
        # denominator = examples whose subject resolved AND have a gold answer QID
        "answer_in_graph_pct": pct(in_graph, with_answer_qid),
        "answer_via_relation": via_relation,
        "answer_via_relation_pct": pct(via_relation, with_answer_qid),
        "avg_neighbors_per_subject": avg_neighbors,
        "subjects_capped": capped_total,
    }
