"""Offline tests for the templated descriptive task family."""

from conceptformer.generate.descriptive import DESCRIPTIVE_TEMPLATES, descriptive_tasks
from conceptformer.schemas import Entity, Subgraph


def _sg(label: str = "Douglas Adams", qid: str = "Q42") -> Subgraph:
    return Subgraph(center=Entity(qid=qid, label=label), edges=[], n_edges_total=0)


def test_descriptive_tasks_name_subject_and_have_no_answer():
    tasks = descriptive_tasks(_sg(), n=2)
    assert len(tasks) == 2
    assert all(t.task_type == "descriptive" for t in tasks)
    assert all(t.answer is None for t in tasks)  # target is the teacher's description
    assert all("Douglas Adams" in t.question for t in tasks)  # injection anchor
    assert all(t.subject_qid == "Q42" for t in tasks)


def test_descriptive_is_deterministic_per_subject():
    assert [t.question for t in descriptive_tasks(_sg())] == [
        t.question for t in descriptive_tasks(_sg())
    ]


def test_descriptive_caps_at_template_count():
    assert len(descriptive_tasks(_sg(), n=99)) == len(DESCRIPTIVE_TEMPLATES)


def test_descriptive_falls_back_to_qid_without_label():
    tasks = descriptive_tasks(_sg(label=""), n=1)  # label falsy → use qid
    assert "Q42" in tasks[0].question
