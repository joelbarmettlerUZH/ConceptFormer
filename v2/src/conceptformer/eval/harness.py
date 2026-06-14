"""Scoring harness for PopQA.

Three condition-agnostic metrics bracket and refine accuracy:
- ``substring``  — hygiened word-boundary match over the gold aliases (corrected lenient).
- ``strict_em``  — normalized first-line exact match (pessimistic lower bound).
- ``fair``       — word-boundary match over gold aliases PLUS subclass-expanded labels
  (credits a more specific correct answer, e.g. "Minister" for "politician").

Reported on the full snapshot-covered set AND on the graph-supported subset (examples whose
gold answer is reachable via the queried relation in the current snapshot — isolates
"can the model use the knowledge" from "is the knowledge in the KG").
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from conceptformer.eval.metrics import popqa_official, strict_match, word_boundary_match
from conceptformer.schemas import QAExample, Subgraph

_POPULARITY_BUCKETS: list[tuple[float, str]] = [
    (100, "<1e2 (long-tail)"),
    (1_000, "1e2-1e3"),
    (10_000, "1e3-1e4"),
    (100_000, "1e4-1e5"),
]
LONG_TAIL_THRESHOLD = 100.0
_BUCKET_ORDER = [label for _, label in _POPULARITY_BUCKETS] + [">=1e5", "unknown"]
# popqa_official = comparable-to-literature; word_boundary/fair/strict_em = corrected/analysis.
METRIC_NAMES = ("popqa_official", "word_boundary", "strict_em", "fair")


def _popularity_bucket(pop: float | None) -> str:
    if pop is None:
        return "unknown"
    for upper, label in _POPULARITY_BUCKETS:
        if pop < upper:
            return label
    return ">=1e5"


def _pct(hits: int, total: int) -> float:
    return round(100.0 * hits / total, 2) if total else 0.0


def is_graph_supported(example: QAExample, subgraph: Subgraph) -> bool:
    """True if the gold answer is reachable from the subject via the queried relation."""
    if not example.answer_qid:
        return False
    return any(
        e.property_id == example.relation_id and e.neighbor.qid == example.answer_qid
        for e in subgraph.edges
    )


def _expanded(example: QAExample, expanded_aliases: Mapping[str, list[str]]) -> list[str]:
    extra = expanded_aliases.get(example.answer_qid or "", []) if example.answer_qid else []
    return [*example.answer_labels, *extra]


def metric_hits(
    example: QAExample, prediction: str, expanded_aliases: Mapping[str, list[str]]
) -> dict[str, bool]:
    gold = example.answer_labels
    return {
        "popqa_official": popqa_official(prediction, gold),
        "word_boundary": word_boundary_match(prediction, gold),
        "strict_em": strict_match(prediction, gold),
        "fair": word_boundary_match(prediction, _expanded(example, expanded_aliases)),
    }


def score(
    examples: Sequence[QAExample],
    predictions: Sequence[str],
    *,
    expanded_aliases: Mapping[str, list[str]] | None = None,
) -> dict:
    """Score under every metric, with popularity & relation breakdowns."""
    expanded_aliases = expanded_aliases or {}
    n = len(examples)
    bucket_n: dict[str, int] = defaultdict(int)
    rel_n: dict[str, int] = defaultdict(int)
    longtail_n = 0
    hits = dict.fromkeys(METRIC_NAMES, 0)
    lt_hits = dict.fromkeys(METRIC_NAMES, 0)
    bucket_hit = {m: defaultdict(int) for m in METRIC_NAMES}
    rel_hit = {m: defaultdict(int) for m in METRIC_NAMES}

    for ex, pred in zip(examples, predictions, strict=True):
        bucket = _popularity_bucket(ex.popularity)
        bucket_n[bucket] += 1
        rel_n[ex.relation] += 1
        is_longtail = ex.popularity is not None and ex.popularity < LONG_TAIL_THRESHOLD
        longtail_n += is_longtail
        for name, ok in metric_hits(ex, pred, expanded_aliases).items():
            hits[name] += ok
            bucket_hit[name][bucket] += ok
            rel_hit[name][ex.relation] += ok
            if is_longtail:
                lt_hits[name] += ok

    def block(name: str) -> dict:
        return {
            "accuracy_pct": _pct(hits[name], n),
            "longtail_accuracy_pct": _pct(lt_hits[name], longtail_n),
            "by_popularity": {
                label: {
                    "n": bucket_n[label],
                    "acc_pct": _pct(bucket_hit[name][label], bucket_n[label]),
                }
                for label in _BUCKET_ORDER
                if bucket_n[label]
            },
            "by_relation": {
                rel: {"n": rel_n[rel], "acc_pct": _pct(rel_hit[name][rel], rel_n[rel])}
                for rel in sorted(rel_n)
            },
        }

    return {
        "n": n,
        "longtail_n": longtail_n,
        "metrics": {name: block(name) for name in METRIC_NAMES},
    }


def save_report(report: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_model = report["model"].replace("/", "_")
    path = results_dir / f"popqa_{report['condition']}_{safe_model}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def save_predictions(
    examples: Sequence[QAExample],
    predictions: Sequence[str],
    graph_supported: Sequence[bool],
    results_dir: Path,
    *,
    condition: str,
    model: str,
    expanded_aliases: Mapping[str, list[str]] | None = None,
    token_records: Sequence[object] | None = None,
) -> Path:
    """Persist per-example predictions + metric flags + graph-supported flag + token cost.

    ``token_records`` (aligned to ``examples``) carry ``input_tokens``/``knowledge_tokens`` so
    the per-example token cost is recoverable for the token-efficiency analysis.
    """
    expanded_aliases = expanded_aliases or {}
    toks = list(token_records) if token_records is not None else [None] * len(examples)
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    path = results_dir / f"popqa_{condition}_{safe_model}_predictions.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for ex, pred, gs, tok in zip(examples, predictions, graph_supported, toks, strict=True):
            row = {
                "subject_qid": ex.subject_qid,
                "relation": ex.relation,
                "relation_id": ex.relation_id,
                "question": ex.question,
                "gold": ex.answer_labels,
                "answer_qid": ex.answer_qid,
                "prediction": pred,
                "popularity": ex.popularity,
                "graph_supported": gs,
                "hits": metric_hits(ex, pred, expanded_aliases),
                "input_tokens": getattr(tok, "input_tokens", None),
                "knowledge_tokens": getattr(tok, "knowledge_tokens", None),
                "uncapped_knowledge_tokens": getattr(tok, "uncapped_knowledge_tokens", None),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
