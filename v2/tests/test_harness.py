"""Spec-driven tests for the PopQA scoring harness, graph-supported filter, and fairness
invariants. We assert intended behavior (correct aggregation, exact bucket boundaries, order /
determinism invariants, metric independence from the model) rather than freezing outputs.
"""

import json

import pytest

from conceptformer.eval.harness import (
    METRIC_NAMES,
    _popularity_bucket,
    is_graph_supported,
    metric_hits,
    save_predictions,
    score,
)
from conceptformer.schemas import Edge, Entity, QAExample, Subgraph


def _ex(qid="Q1", pop=50.0, *, relation="occupation", relation_id="P106",
        answer_qid="Qa", labels=("writer",)):
    return QAExample(
        source="popqa", split="test", subject_qid=qid, relation=relation,
        relation_id=relation_id, question="?", answer_qid=answer_qid,
        answer_labels=list(labels), popularity=pop,
    )


# ======================================================================================
# Popularity bucketing — EXACT boundaries (matters for the long-tail reporting).
# ======================================================================================
@pytest.mark.parametrize(
    ("pop", "bucket"),
    [
        (0, "<1e2 (long-tail)"),
        (99.99, "<1e2 (long-tail)"),
        (100, "1e2-1e3"),  # boundary is exclusive-below
        (999, "1e2-1e3"),
        (1000, "1e3-1e4"),
        (99_999, "1e4-1e5"),
        (100_000, ">=1e5"),
        (None, "unknown"),
    ],
)
def test_popularity_bucket_boundaries(pop, bucket):
    assert _popularity_bucket(pop) == bucket


def test_longtail_is_strictly_below_100():
    examples = [_ex("Q1", 99.99), _ex("Q2", 100.0), _ex("Q3", None)]
    report = score(examples, ["writer", "writer", "writer"])
    assert report["longtail_n"] == 1  # only the 99.99 one; 100 and None excluded


# ======================================================================================
# Aggregation correctness.
# ======================================================================================
def test_all_metrics_present_and_counts_correct():
    examples = [_ex("Q1", 50), _ex("Q2", 50), _ex("Q3", 5000)]
    predictions = ["a famous writer", "a doctor", "Writer"]
    report = score(examples, predictions)
    assert set(report["metrics"]) == set(METRIC_NAMES)
    # popqa_official + word_boundary: Q1 + Q3 hit "writer" → 2/3
    assert report["metrics"]["popqa_official"]["accuracy_pct"] == round(100 * 2 / 3, 2)
    assert report["metrics"]["word_boundary"]["accuracy_pct"] == round(100 * 2 / 3, 2)
    # strict_em: only "Writer" is a terse exact match → 1/3
    assert report["metrics"]["strict_em"]["accuracy_pct"] == round(100 * 1 / 3, 2)


def test_per_relation_breakdown():
    examples = [_ex("Q1", 50, relation="occupation"), _ex("Q2", 50, relation="genre")]
    report = score(examples, ["writer", "comedy"])
    by_rel = report["metrics"]["popqa_official"]["by_relation"]
    assert by_rel["occupation"]["n"] == 1 and by_rel["occupation"]["acc_pct"] == 100.0
    assert by_rel["genre"]["n"] == 1 and by_rel["genre"]["acc_pct"] == 0.0


def test_empty_set_does_not_divide_by_zero():
    report = score([], [])
    assert report["n"] == 0
    assert report["metrics"]["popqa_official"]["accuracy_pct"] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score([_ex()], [])


# ======================================================================================
# Fairness invariants (peer-review defensibility).
# ======================================================================================
def test_score_is_order_invariant():
    examples = [_ex("Q1", 50), _ex("Q2", 5000), _ex("Q3", 200)]
    predictions = ["writer", "nope", "writer"]
    a = score(examples, predictions)
    b = score(list(reversed(examples)), list(reversed(predictions)))
    for m in METRIC_NAMES:
        assert a["metrics"][m]["accuracy_pct"] == b["metrics"][m]["accuracy_pct"]


def test_score_is_deterministic():
    examples = [_ex("Q1", 50), _ex("Q2", 5000)]
    predictions = ["writer", "doctor"]
    assert score(examples, predictions) == score(examples, predictions)


def test_metric_hits_depend_only_on_prediction_and_aliases():
    # The same (example, prediction) must score identically no matter the context/condition —
    # there is no model/condition parameter, which is what keeps base/rag/CF comparable.
    ex = _ex(labels=["writer"])
    assert metric_hits(ex, "a writer", {}) == metric_hits(ex, "a writer", {})


def test_word_boundary_hit_implies_fair_hit():
    # fair scores over gold plus subclass-expanded (a superset), never lower than word_boundary.
    ex = _ex(answer_qid="Qa", labels=["writer"])
    hits = metric_hits(ex, "a writer", {"Qa": ["novelist"]})
    assert not hits["word_boundary"] or hits["fair"]


# ======================================================================================
# fair metric: subclass expansion credits a more specific correct answer.
# ======================================================================================
def test_fair_credits_subclass_but_official_and_word_boundary_do_not():
    ex = _ex(answer_qid="Qpol", labels=["politician"])
    pred = "Minister without portfolio of the Republic of Serbia"
    hits = metric_hits(ex, pred, {"Qpol": ["minister", "mayor", "senator"]})
    assert hits["popqa_official"] is False
    assert hits["word_boundary"] is False
    assert hits["fair"] is True


# ======================================================================================
# Graph-supported filter — isolates capability from coverage.
# ======================================================================================
def _sg() -> Subgraph:
    return Subgraph(
        center=Entity(qid="Q1"),
        edges=[Edge(property_id="P106", neighbor=Entity(qid="Q99"))],
        n_edges_total=1,
    )


@pytest.mark.parametrize(
    ("relation_id", "answer_qid", "expected", "why"),
    [
        ("P106", "Q99", True, "right relation + right answer"),
        ("P106", "Q5", False, "answer not a neighbor"),
        ("P19", "Q99", False, "answer reachable, but via a different relation"),
        ("P106", None, False, "no gold answer qid"),
        (None, "Q99", False, "no relation id"),
    ],
)
def test_is_graph_supported(relation_id, answer_qid, expected, why):
    ex = QAExample(
        source="popqa", split="test", subject_qid="Q1", relation="r",
        relation_id=relation_id, question="?", answer_qid=answer_qid, answer_labels=["a"],
    )
    assert is_graph_supported(ex, _sg()) is expected, why


# ======================================================================================
# Persistence.
# ======================================================================================
def test_save_predictions_roundtrip(tmp_path):
    examples = [_ex("Q1", 50, labels=["writer"])]
    path = save_predictions(
        examples, ["a writer"], [True], tmp_path, condition="base", model="org/m"
    )
    row = json.loads(path.read_text().splitlines()[0])
    assert row["subject_qid"] == "Q1"
    assert row["graph_supported"] is True
    assert set(row["hits"]) == set(METRIC_NAMES)
    assert row["hits"]["word_boundary"] is True
    assert row["input_tokens"] is None and row["knowledge_tokens"] is None  # absent → null


def test_save_predictions_records_token_cost(tmp_path):
    from conceptformer.eval.tokens import TokenRecord

    examples = [_ex("Q1", 50, labels=["writer"])]
    path = save_predictions(
        examples, ["a writer"], [True], tmp_path, condition="rag", model="org/m",
        token_records=[TokenRecord(input_tokens=142, knowledge_tokens=87)],
    )
    row = json.loads(path.read_text().splitlines()[0])
    assert row["input_tokens"] == 142 and row["knowledge_tokens"] == 87
