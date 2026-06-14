"""Offline tests for CF-Train QA assembly (validation + save + manifest)."""

import json

from conceptformer.generate.dataset import (
    dataset_counts,
    load_cftrain_qa,
    save_cftrain_qa,
    update_manifest,
    validate_questions,
)
from conceptformer.generate.schema import CFTrainQA, GeneratedQuestion, GenerationResult
from conceptformer.schemas import Edge, Entity, Subgraph


def _sg() -> Subgraph:
    return Subgraph(
        center=Entity(qid="Q42", label="Douglas Adams"),
        edges=[
            Edge(property_id="P106", property_label="occupation",
                 neighbor=Entity(qid="Q1", label="writer")),
        ],
        n_edges_total=1,
    )


def _q(question: str, answer: str) -> GeneratedQuestion:
    return GeneratedQuestion(question=question, answer=answer, task_type="single")


def test_validate_keeps_only_grounded_subject_naming_questions():
    result = GenerationResult(
        questions=[
            _q("What is Douglas Adams's occupation?", "writer"),  # grounded + names → keep
            _q("What is his occupation?", "writer"),  # no subject name → drop
            _q("Where was Douglas Adams born?", "London"),  # answer not a neighbor → drop
        ]
    )
    rows = validate_questions([_sg()], [result])
    assert len(rows) == 1
    assert rows[0].subject_qid == "Q42"
    assert rows[0].question.startswith("What is Douglas Adams")
    assert rows[0].subject_label == "Douglas Adams"


def test_validate_binds_answer_qid_and_aliases():
    rows = validate_questions(
        [_sg()], [GenerationResult(questions=[_q("What is Douglas Adams's occupation?", "writer")])]
    )
    assert rows[0].answer_qid == "Q1"  # the matched neighbor's qid
    assert rows[0].answer_aliases == ["writer"]  # the neighbor's canonical label
    assert rows[0].accepted_answers == ["writer"]  # deduped against `answer`


def test_validate_skips_failed_generations():
    assert validate_questions([_sg()], [None]) == []


def _row(task_type="single", tier=None) -> CFTrainQA:
    return CFTrainQA(subject_qid="Q1", subject_label="X", question="q", answer="a",
                     task_type=task_type, tier=tier)


def test_dataset_counts():
    counts = dataset_counts([_row("single", "needs_graph"), _row("descriptive"), _row("single")])
    assert counts["n_rows"] == 3
    assert counts["by_task_type"] == {"single": 2, "descriptive": 1}
    assert counts["by_tier"] == {"needs_graph": 1}  # untiered rows excluded


def test_update_manifest_accumulates_stages(tmp_path):
    update_manifest(tmp_path, "generate", {"snapshot": "smoke", "n_rows": 3}, stamp="T0")
    path = update_manifest(tmp_path, "tier", {"model": "Qwen/Qwen3-0.6B"}, stamp="T1")
    data = json.loads(path.read_text())
    assert set(data["stages"]) == {"generate", "tier"}
    assert data["stages"]["generate"] == {"snapshot": "smoke", "n_rows": 3, "at": "T0"}
    assert data["stages"]["tier"]["model"] == "Qwen/Qwen3-0.6B"  # records model dependence
    assert data["stages"]["tier"]["at"] == "T1"


def test_save_load_roundtrip(tmp_path):
    rows = validate_questions(
        [_sg()], [GenerationResult(questions=[_q("What is Douglas Adams's occupation?", "writer")])]
    )
    path = save_cftrain_qa(rows, tmp_path / "qa.jsonl")
    assert load_cftrain_qa(path) == rows
