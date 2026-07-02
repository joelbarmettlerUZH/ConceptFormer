"""Offline tests for the frozen eval sets (eval/evalsets.py)."""

from conceptformer.eval.evalsets import popqa_eval_items, sample_rows
from conceptformer.generate.schema import CFTrainQA
from conceptformer.schemas import Edge, Entity, QAExample, Subgraph


def _sg(qid: str, n_edges: int = 1) -> Subgraph:
    return Subgraph(
        center=Entity(qid=qid, label=qid),
        edges=[
            Edge(property_id="P1", property_label="p", neighbor=Entity(qid=f"{qid}N{i}"))
            for i in range(n_edges)
        ],
    )


def _ex(qid: str, question: str, relation: str = "occupation") -> QAExample:
    return QAExample(
        source="popqa", split="test", subject_qid=qid, relation=relation,
        question=question, answer_labels=["gold"],
    )


def test_popqa_items_independent_of_input_order():
    # PopQA on disk is relation-grouped; the frozen set must not depend on that order.
    examples = [_ex(f"Q{i}", f"q{i}", relation="occupation" if i < 5 else "director")
                for i in range(10)]
    sgs = {f"Q{i}": _sg(f"Q{i}") for i in range(10)}
    forward = popqa_eval_items(examples, sgs, n=4)
    reversed_in = popqa_eval_items(list(reversed(examples)), sgs, n=4)
    assert [(sg.center.qid, q) for sg, q, _, _ in forward] == [
        (sg.center.qid, q) for sg, q, _, _ in reversed_in
    ]


def test_popqa_items_filters_and_n_zero_returns_all():
    examples = [_ex("Q1", "q1"), _ex("Q2", "q2"), _ex("Q3", "q3"), _ex("Q4", "no-answers")]
    examples[3].answer_labels = []
    sgs = {"Q1": _sg("Q1"), "Q2": _sg("Q2", n_edges=0), "Q4": _sg("Q4")}
    items = popqa_eval_items(examples, sgs, n=0)
    # Q2 dropped (empty neighborhood), Q3 dropped (no snapshot), Q4 dropped (no gold).
    assert [sg.center.qid for sg, _, _, _ in items] == ["Q1"]


def test_sample_rows_fixed_across_caller_order_and_prefix_consistent():
    rows = [
        CFTrainQA(subject_qid=f"Q{i}", subject_label=f"Q{i}", question=f"q{i}",
                  answer="a", task_type="single", teacher_target_ids=[1])
        for i in range(50)
    ]
    small = sample_rows(rows, 10)
    large = sample_rows(list(reversed(rows)), 30)
    # Same frozen order regardless of input order; smaller n is a PREFIX of larger n, so
    # different-size evals still pair on their intersection.
    assert [r.question for r in small] == [r.question for r in large[:10]]
    assert len(sample_rows(rows, 0)) == 50  # 0 = all
