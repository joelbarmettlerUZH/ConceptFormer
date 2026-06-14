"""Offline tests for CF-Train signal tiering + keep policy."""

from conceptformer.generate.schema import CFTrainQA
from conceptformer.generate.signal import answer_ok, apply_keep_policy, assign_tier


def _qa(answer: str | None = "writer") -> CFTrainQA:
    return CFTrainQA(subject_qid="Q1", subject_label="X", question="q", answer=answer,
                     task_type="single")


def test_answer_ok():
    assert answer_ok("He was a writer.", "writer") is True
    assert answer_ok("He was a doctor.", "writer") is False
    assert answer_ok("writer", None) is False  # no gold → not checkable


def test_answer_ok_is_alias_aware():
    # any accepted surface form matching counts as correct (mirrors the eval harness).
    aliases = ["USA", "United States of America"]
    assert answer_ok("It is in the United States of America.", aliases) is True
    assert answer_ok("It is in the USA.", aliases) is True
    assert answer_ok("It is in Canada.", aliases) is False
    assert answer_ok("anything", []) is False  # no accepted answers → not checkable


def test_answer_ok_uses_accepted_answers_property():
    qa = CFTrainQA(subject_qid="Q1", subject_label="X", question="q", answer="US",
                   answer_aliases=["United States"], task_type="single")
    assert qa.accepted_answers == ["US", "United States"]
    assert answer_ok("Located in the United States.", qa.accepted_answers) is True


def test_assign_tier():
    assert assign_tier(base_ok=True, rag_ok=True) == "base_knows"
    assert assign_tier(base_ok=True, rag_ok=False) == "base_knows"
    assert assign_tier(base_ok=False, rag_ok=True) == "needs_graph"
    assert assign_tier(base_ok=False, rag_ok=False) is None  # hard → drop


def test_keep_policy_keeps_needs_graph_and_drops_hard():
    rows = [(_qa(), "needs_graph"), (_qa(), None), (_qa(), "needs_graph")]
    kept = apply_keep_policy(rows, keep_base_known_frac=0.0)
    assert len(kept) == 2
    assert all(r.tier == "needs_graph" for r in kept)


def test_keep_policy_base_known_fraction_bounds():
    rows = [(_qa(), "base_knows") for _ in range(100)]
    assert len(apply_keep_policy(rows, keep_base_known_frac=0.0)) == 0
    kept_all = apply_keep_policy(rows, keep_base_known_frac=1.0)
    assert len(kept_all) == 100
    assert all(r.tier == "base_knows" for r in kept_all)
