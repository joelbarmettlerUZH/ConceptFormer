"""Structured output schema for generated CF-Train questions (vLLM guided JSON)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GeneratedQuestion(BaseModel):
    question: str = Field(description="A natural question answerable only from the given facts.")
    answer: str = Field(description="The short answer, taken verbatim from the facts.")
    task_type: Literal["single", "compositional"] = Field(
        description="'single' uses one fact; 'compositional' combines two or more."
    )


class GenerationResult(BaseModel):
    questions: list[GeneratedQuestion]


class CFTrainQA(BaseModel):
    """A validated CF-Train task bound to its subject (one row of the dataset).

    ``answer`` is the grounded short answer for QA tasks; it is ``None`` for ``descriptive``
    tasks (their target is the teacher's free-form description, produced later in Stage 5).
    Later stages enrich each row with the teacher's greedy target path + signal tier.
    """

    subject_qid: str
    subject_label: str
    question: str
    answer: str | None = None
    # The grounded answer's neighbor qid + every surface form we accept as correct (Gemma's
    # verbatim answer plus the matched neighbor's full label). Tiering matches a model output
    # against this whole set, not just ``answer``, so a correct-but-reworded answer isn't
    # mis-judged. ``answer_qid`` is the hook for later alias/subclass expansion (the eval
    # harness's ``expanded_aliases`` map is keyed the same way). Empty for descriptive/control.
    answer_qid: str | None = None
    answer_aliases: list[str] = Field(default_factory=list)
    # single/compositional = Gemma QA; descriptive = "essence"; control = capability-preservation
    # (subject mentioned, task is NOT about its facts — target is the model's normal behavior).
    task_type: Literal["single", "compositional", "descriptive", "control"]
    # Signal tier (Stage 4): does the graph demonstrably help on this example?
    tier: Literal["needs_graph", "base_knows", "descriptive", "control"] | None = None
    # Teacher greedy continuation given graph-in-context (Stage 5) — the distillation target's
    # token path; the per-position distributions are recomputed live from the frozen teacher.
    teacher_target_ids: list[int] | None = None
    teacher_target_text: str | None = None

    @property
    def accepted_answers(self) -> list[str]:
        """Unique non-empty surface forms a model output may match to count as correct."""
        seen: dict[str, None] = {}
        for cand in [self.answer, *self.answer_aliases]:
            if cand and cand not in seen:
                seen[cand] = None
        return list(seen)
