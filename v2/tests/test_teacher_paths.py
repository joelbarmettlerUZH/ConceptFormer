"""Offline tests for teacher-path trimming, prompt convention, and attachment."""

from conceptformer.generate.schema import CFTrainQA
from conceptformer.generate.teacher import (
    TEACHER_SYSTEM,
    attach_teacher_paths,
    cftrain_prompt,
    drop_degenerate_paths,
    is_degenerate_path,
    teacher_prompt,
)
from conceptformer.model.chat import trim_generated_path


def test_trim_generated_path():
    assert trim_generated_path([5, 6, 2, 0, 0], eos=2, pad=0) == [5, 6, 2]  # include first eos
    assert trim_generated_path([5, 6, 0, 0], eos=2, pad=0) == [5, 6]  # no eos → strip trailing pad
    assert trim_generated_path([2, 0, 0], eos=2, pad=0) == [2]  # immediate eos
    assert trim_generated_path([5, 6], eos=2, pad=0) == [5, 6]  # nothing to trim
    assert trim_generated_path([5, 6, 0], eos=None, pad=None) == [5, 6, 0]  # no eos/pad info → keep


def test_teacher_prompt_convention():
    system, user = teacher_prompt("Describe X.", "Facts about X:\n- occupation: writer")
    assert system == TEACHER_SYSTEM
    assert user == "Facts about X:\n- occupation: writer\n\nDescribe X."


def test_cftrain_prompt_single_source_for_base_rag_teacher():
    facts = "Facts about X:\n- occupation: writer"
    # base = task alone (no graph), rag/teacher = facts then task — same system, same builder.
    base_sys, base_user = cftrain_prompt("Describe X.")
    rag_sys, rag_user = cftrain_prompt("Describe X.", facts)
    assert base_sys == rag_sys == TEACHER_SYSTEM
    assert base_user == "Describe X."
    assert rag_user == f"{facts}\n\nDescribe X."
    # teacher_prompt is just the RAG leg of the one convention — no divergence possible.
    assert teacher_prompt("Describe X.", facts) == cftrain_prompt("Describe X.", facts)
    # empty facts collapse to the base form (no stray leading blank lines).
    assert cftrain_prompt("Describe X.", "") == cftrain_prompt("Describe X.")


def _ex(ids, text) -> CFTrainQA:
    return CFTrainQA(
        subject_qid="Q1", subject_label="X", question="q", answer="a",
        task_type="single", teacher_target_ids=ids, teacher_target_text=text,
    )


def test_is_degenerate_path():
    assert is_degenerate_path(_ex([], "")) is True  # empty path
    assert is_degenerate_path(_ex([5], "   ")) is True  # whitespace-only text
    assert is_degenerate_path(_ex(None, None)) is True  # not yet populated
    assert is_degenerate_path(_ex([5, 6], "writer")) is False  # real content


def test_drop_degenerate_paths():
    kept, n = drop_degenerate_paths([_ex([5], "writer"), _ex([], ""), _ex([7], "  ")])
    assert n == 2 and len(kept) == 1
    assert kept[0].teacher_target_text == "writer"


def test_attach_teacher_paths_preserves_other_fields():
    ex = CFTrainQA(
        subject_qid="Q1", subject_label="X", question="q", answer="a",
        task_type="single", tier="needs_graph",
    )
    out = attach_teacher_paths([ex], [[5, 6, 2]], ["hello world"])
    assert out[0].teacher_target_ids == [5, 6, 2]
    assert out[0].teacher_target_text == "hello world"
    assert out[0].tier == "needs_graph" and out[0].answer == "a"  # untouched
