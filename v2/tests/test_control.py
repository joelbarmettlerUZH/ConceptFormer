"""Offline tests for the control (capability-preservation) task family."""

from conceptformer.generate.control import CONTROL_TEMPLATES, control_tasks
from conceptformer.schemas import Entity, Subgraph


def _sg(label: str = "Douglas Adams", qid: str = "Q42") -> Subgraph:
    return Subgraph(center=Entity(qid=qid, label=label), edges=[], n_edges_total=0)


def test_control_tasks_mention_subject_and_have_no_answer():
    tasks = control_tasks(_sg(), n=2)
    assert len(tasks) == 2
    assert all(t.task_type == "control" for t in tasks)
    assert all(t.answer is None for t in tasks)  # target = teacher's normal behavior
    assert all("Douglas Adams" in t.question for t in tasks)  # injection anchor present


def test_control_is_deterministic_and_caps():
    assert [t.question for t in control_tasks(_sg())] == [t.question for t in control_tasks(_sg())]
    assert len(control_tasks(_sg(), n=99)) == len(CONTROL_TEMPLATES)


def test_control_and_descriptive_differ_for_same_subject():
    from conceptformer.generate.descriptive import descriptive_tasks

    # independent template banks + seeds → control prompts are not descriptive prompts
    ctrl = {t.question for t in control_tasks(_sg(), n=2)}
    desc = {t.question for t in descriptive_tasks(_sg(), n=2)}
    assert ctrl.isdisjoint(desc)
