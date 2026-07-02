"""Offline tests for the fact-grouped split + the strict (leakage-free) val subset."""

from conceptformer.generate.schema import CFTrainQA
from conceptformer.train.harness import (
    fact_key,
    split_by_held_out_facts,
    split_by_held_out_questions,
    strict_val_subset,
)


def _qa(qid, q, answer="a", answer_qid="Q9", task="single", path=(1, 2)) -> CFTrainQA:
    return CFTrainQA(
        subject_qid=qid, subject_label=qid, question=q, answer=answer, answer_qid=answer_qid,
        task_type=task, teacher_target_ids=list(path) if path else None,
    )


def test_fact_key_prefers_answer_qid_and_falls_back_to_text():
    assert fact_key(_qa("Q1", "q", answer_qid="Q9")) == ("Q1", "Q9")
    assert fact_key(_qa("Q1", "q", answer="Paris ", answer_qid=None)) == ("Q1", "paris")


def test_paraphrases_of_same_fact_never_straddle_the_split():
    # 4 facts x 3 paraphrases per entity; question-level split WOULD leak, fact-level must not.
    rows = [
        _qa("Q1", f"fact{f} phrasing{p}", answer_qid=f"Q{f}")
        for f in range(4)
        for p in range(3)
    ]
    train, val = split_by_held_out_facts(rows, val_frac=0.25, seed=0)
    train_facts = {fact_key(r) for r in train}
    val_facts = {fact_key(r) for r in val}
    assert not train_facts & val_facts
    assert len(val) == 3  # 1 of 4 fact groups held out -> all 3 paraphrases together
    # determinism
    again = split_by_held_out_facts(rows, val_frac=0.25, seed=0)[1]
    assert [r.question for r in again] == [r.question for r in val]


def test_question_level_split_does_leak_the_fixture_facts():
    # Sanity that the fixture actually distinguishes the two splitters: with paraphrases of the
    # same fact, some seed puts two of them on opposite sides under question-level splitting.
    rows = [
        _qa("Q1", f"fact{f} phrasing{p}", answer_qid=f"Q{f}")
        for f in range(4)
        for p in range(3)
    ]
    leaked = False
    for seed in range(5):
        train, val = split_by_held_out_questions(rows, val_frac=0.25, seed=seed)
        if {fact_key(r) for r in train} & {fact_key(r) for r in val}:
            leaked = True
            break
    assert leaked


def test_non_answerable_rows_always_train():
    rows = [
        _qa("Q1", f"q{i}", answer_qid=f"Q{i}") for i in range(4)
    ] + [_qa("Q1", "desc", task="descriptive", answer=None, answer_qid=None, path=(5,))]
    train, val = split_by_held_out_facts(rows, val_frac=0.5, seed=1)
    assert all(r.task_type == "single" for r in val)
    assert "desc" in {r.question for r in train}


def test_strict_val_subset_removes_fact_overlap():
    train = [_qa("Q1", "who is X married to?", answer_qid="Q9")]
    val = [
        _qa("Q1", "name the spouse of X", answer_qid="Q9"),  # same fact -> leaky -> removed
        _qa("Q1", "where was X born?", answer_qid="Q7"),  # different fact -> kept
        _qa("Q2", "who is Y married to?", answer_qid="Q9"),  # different entity -> kept
    ]
    strict = strict_val_subset(train, val)
    assert [r.question for r in strict] == ["where was X born?", "who is Y married to?"]
