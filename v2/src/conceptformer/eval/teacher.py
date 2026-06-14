"""Teacher-validation experiment.

Question: with the subject's neighborhood verbalized in context, does the (frozen) backbone
actually answer entity-centric questions? This is the foundation of the KL-distillation
objective — if the graph-in-context teacher can't answer, distilling from it is pointless.

Reports base (no knowledge) vs RAG (graph-in-context) accuracy on PopQA, and breaks the RAG
accuracy down by trivial (queried property has one value) vs multi-value questions — to see
whether any lift is just property-lookup or genuine disambiguation.
"""

from __future__ import annotations

import random
from pathlib import Path

from conceptformer.config import settings
from conceptformer.data.benchmarks import load_popqa
from conceptformer.data.snapshot import iter_subgraphs
from conceptformer.eval.metrics import word_boundary_match
from conceptformer.eval.predictors import base_prompt, make_rag_prompt
from conceptformer.schemas import Subgraph


def _queried_value_count(sg: Subgraph, relation_id: str | None) -> int:
    if relation_id is None:
        return 0
    return sum(1 for e in sg.edges if e.property_id == relation_id)


def run_teacher_validation(
    *,
    snapshot_name: str = "popqa_sample",
    n: int = 100,
    model_id: str = "Qwen/Qwen3-0.6B",
    device: str = "cuda",
    seed: int = 0,
    max_new_tokens: int = 32,
) -> dict:
    from conceptformer.model.chat import ChatModel  # lazy: needs the `infer` deps

    snapshot_dir: Path = settings.snapshots_dir / snapshot_name
    subgraphs = {sg.center.qid: sg for sg in iter_subgraphs(snapshot_dir)}
    examples = [
        e for e in load_popqa() if e.subject_qid in subgraphs and e.answer_labels
    ]
    random.Random(seed).shuffle(examples)
    items = examples[:n]

    model = ChatModel(model_id, device=device)
    rag_prompt = make_rag_prompt(model.count_tokens, settings.rag_context_tokens)

    base_hits = rag_hits = 0
    trivial_n = trivial_rag = multi_n = multi_rag = 0
    rows: list[dict] = []
    for e in items:
        sg = subgraphs[e.subject_qid]
        base_pred = model.generate(*base_prompt(e, None), max_new_tokens=max_new_tokens)
        rag_pred = model.generate(*rag_prompt(e, sg), max_new_tokens=max_new_tokens)
        base_ok = word_boundary_match(base_pred, e.answer_labels)
        rag_ok = word_boundary_match(rag_pred, e.answer_labels)
        base_hits += base_ok
        rag_hits += rag_ok

        value_count = _queried_value_count(sg, e.relation_id)
        if value_count == 1:
            trivial_n += 1
            trivial_rag += rag_ok
        elif value_count > 1:
            multi_n += 1
            multi_rag += rag_ok

        rows.append({
            "question": e.question,
            "gold": e.answer_labels[0],
            "base": base_pred,
            "rag": rag_pred,
            "base_ok": base_ok,
            "rag_ok": rag_ok,
            "values_under_relation": value_count,
        })

    def pct(a: int, b: int) -> float:
        return round(100.0 * a / b, 1) if b else 0.0

    return {
        "model": model_id,
        "n": len(items),
        "base_acc_pct": pct(base_hits, len(items)),
        "rag_acc_pct": pct(rag_hits, len(items)),
        "lift_pct": round(pct(rag_hits, len(items)) - pct(base_hits, len(items)), 1),
        "rag_acc_trivial_pct": pct(trivial_rag, trivial_n),
        "rag_acc_multi_pct": pct(multi_rag, multi_n),
        "trivial_n": trivial_n,
        "multi_n": multi_n,
        "sample_rows": rows[:10],
    }
