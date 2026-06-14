"""Assemble validated CF-Train QA rows from generator output.

``validate_questions`` is pure (no network) — it pairs each subgraph with its generation
result and keeps only questions that pass the grounding/subject checks, binding them to the
subject. The Gemma call itself lives in the CLI (async); this stays unit-testable.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from conceptformer.generate.grounding import grounded_neighbor, passes
from conceptformer.generate.schema import CFTrainQA, GenerationResult
from conceptformer.schemas import Subgraph


def validate_questions(
    subgraphs: Sequence[Subgraph], results: Sequence[GenerationResult | None]
) -> list[CFTrainQA]:
    """Keep grounded, subject-naming questions; bind each to its subject + answer entity."""
    rows: list[CFTrainQA] = []
    for sg, result in zip(subgraphs, results, strict=True):
        if result is None:
            continue
        for q in result.questions:
            if not passes(q, sg):
                continue
            nbr = grounded_neighbor(q, sg)
            aliases = [nbr.label] if nbr and nbr.label else []
            rows.append(
                CFTrainQA(
                    subject_qid=sg.center.qid,
                    subject_label=sg.center.label or sg.center.qid,
                    question=q.question,
                    answer=q.answer,
                    answer_qid=nbr.qid if nbr else None,
                    answer_aliases=aliases,
                    task_type=q.task_type,
                )
            )
    return rows


def save_cftrain_qa(rows: Sequence[CFTrainQA], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")
    return path


def load_cftrain_qa(path: Path) -> list[CFTrainQA]:
    with path.open(encoding="utf-8") as fh:
        return [CFTrainQA.model_validate_json(line) for line in fh if line.strip()]


def dataset_counts(rows: Sequence[CFTrainQA]) -> dict:
    """Provenance-friendly summary of a CF-Train split: totals by task_type and tier."""
    return {
        "n_rows": len(rows),
        "by_task_type": dict(Counter(r.task_type for r in rows)),
        "by_tier": dict(Counter(r.tier for r in rows if r.tier is not None)),
    }


def update_manifest(dataset_dir: Path, stage: str, payload: dict, *, stamp: str) -> Path:
    """Record one pipeline stage's provenance into the dataset's ``manifest.json``.

    Each stage (generate / tier / teacher) writes the inputs it depended on — crucially the
    backbone **model** and **prompt_version**, since the tier signal and teacher paths are only
    valid for the exact model+prompt that produced them. Stages accumulate under ``stages`` so a
    dataset carries its full lineage. ``stamp`` is passed in (not read from the clock) to keep
    the writer pure/testable.
    """
    manifest_path = dataset_dir / "manifest.json"
    data = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"stages": {}}
    data.setdefault("stages", {})[stage] = {**payload, "at": stamp}
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data, indent=2))
    return manifest_path
