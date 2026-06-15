"""Offline tests for the held-out-question train/val split."""

from conceptformer.generate.schema import CFTrainQA
from conceptformer.train.harness import is_answerable, split_by_held_out_questions


def _qa(qid, q, task="single", answer="a", path=(1, 2)) -> CFTrainQA:
    return CFTrainQA(
        subject_qid=qid, subject_label=qid, question=q, answer=answer,
        task_type=task, teacher_target_ids=list(path) if path else None,
    )


def test_is_answerable():
    assert is_answerable(_qa("Q1", "q")) is True
    assert is_answerable(_qa("Q1", "q", task="descriptive", answer=None)) is False
    assert is_answerable(_qa("Q1", "q", path=None)) is False  # no teacher path


def test_split_holds_out_per_entity_and_is_deterministic():
    rows = [_qa("Q1", f"q{i}") for i in range(10)] + [_qa("Q2", f"r{i}") for i in range(10)]
    train, val = split_by_held_out_questions(rows, val_frac=0.3, seed=0)
    assert len(val) == 6 and len(train) == 14  # 3 of each entity's 10 held out
    # both entities represented in val (per-entity holdout, not global)
    assert {r.subject_qid for r in val} == {"Q1", "Q2"}
    # deterministic
    assert [r.question for r in split_by_held_out_questions(rows, seed=0)[1]] == [
        r.question for r in val
    ]
    # disjoint
    assert not ({id(r) for r in train} & {id(r) for r in val})


def test_descriptive_and_control_always_train_never_val():
    rows = [
        _qa("Q1", "q0"), _qa("Q1", "q1"), _qa("Q1", "q2"), _qa("Q1", "q3"),
        _qa("Q1", "desc", task="descriptive", answer=None, path=(5,)),
        _qa("Q1", "ctrl", task="control", answer=None, path=(6,)),
    ]
    train, val = split_by_held_out_questions(rows, val_frac=0.5, seed=1)
    assert all(r.task_type in ("single", "compositional") for r in val)
    assert {"desc", "ctrl"} <= {r.question for r in train}


def test_few_question_entity_contributes_only_to_train():
    rows = [_qa("Q1", "only")]  # 1 question → int(1*0.3)=0 held out
    train, val = split_by_held_out_questions(rows, val_frac=0.3)
    assert len(val) == 0 and len(train) == 1
