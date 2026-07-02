"""Per-row cache key making `extract-teacher-paths` resumable (deterministic + discriminates)."""

from __future__ import annotations

from conceptformer.generate.teacher import teacher_path_key


def test_key_is_deterministic_for_same_inputs() -> None:
    a = teacher_path_key("Qwen/Qwen3-0.6B", 64, "facts...\n\nQ: who?")
    b = teacher_path_key("Qwen/Qwen3-0.6B", 64, "facts...\n\nQ: who?")
    assert a == b and a.startswith("teacher_path:")  # stable → a re-run hits the cache


def test_key_distinguishes_prompt_model_and_max_new() -> None:
    base = teacher_path_key("Qwen/Qwen3-0.6B", 64, "p1")
    assert base != teacher_path_key("Qwen/Qwen3-0.6B", 64, "p2")  # different prompt
    assert base != teacher_path_key("other-model", 64, "p1")  # different teacher
    assert base != teacher_path_key("Qwen/Qwen3-0.6B", 32, "p1")  # different decode length
