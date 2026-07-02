"""Stage 5: teacher greedy-path extraction.

The canonical CF-Train prompt convention (shared by the teacher now and the student at train
time): a neutral system prompt + the verbalized facts + the task. The TEACHER puts the facts as
text; the STUDENT (ConceptFormer) will replace that fact span with its `k` concept tokens. Both
keep the same task text (which names the subject), so they differ only in graph-text vs tokens.

We store the teacher's greedy continuation token-ids as the distillation target's path; the
per-position teacher distributions are recomputed live from the frozen model during training.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from conceptformer.generate.schema import CFTrainQA


def teacher_path_key(model: str, max_new_tokens: int, prompt: str) -> str:
    """Stable per-row cache key for the frozen-teacher greedy path, so `extract-teacher-paths`
    RESUMES across runs (the greedy decode is deterministic in the prompt). Namespaced so it never
    collides with other cache users."""
    h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]  # cache key, not security
    return f"teacher_path:{model}:{max_new_tokens}:{h}"

# Neutral on purpose: works across all task families (QA, descriptive, control) and never tells
# the model to "talk about the subject" — capability preservation starts at the prompt.
TEACHER_SYSTEM = "You are a helpful assistant."

# Bump when the CF-Train prompt convention below changes, so tiered/distilled datasets built
# under different conventions are never silently mixed (recorded in each dataset's manifest).
CFTRAIN_PROMPT_VERSION = "v1"


def cftrain_prompt(question: str, facts: str | None = None) -> tuple[str, str]:
    """The single CF-Train prompt convention, shared by every leg that must agree:

    - **base** (``facts=None``) — the task alone, used by Stage-4 tiering's no-graph leg.
    - **RAG / teacher** (``facts`` given) — verbalized facts then the task, used by Stage-4's
      graph leg, Stage-5 teacher-path extraction, and the student at train time.

    Routing all three through one builder guarantees the tier signal, the stored teacher path,
    and training share an identical prompt — the only thing that differs is graph-text (teacher)
    vs concept tokens (student).
    """
    user = f"{facts}\n\n{question}" if facts else question
    return TEACHER_SYSTEM, user


def teacher_prompt(question: str, facts: str) -> tuple[str, str]:
    """(system, user) for the teacher: verbalized facts followed by the task (RAG leg)."""
    return cftrain_prompt(question, facts)


def attach_teacher_paths(
    examples: Sequence[CFTrainQA], paths: Sequence[list[int]], texts: Sequence[str]
) -> list[CFTrainQA]:
    """Enrich each example with its teacher greedy target path + decoded text."""
    return [
        ex.model_copy(update={"teacher_target_ids": ids, "teacher_target_text": text})
        for ex, ids, text in zip(examples, paths, texts, strict=True)
    ]


def is_degenerate_path(example: CFTrainQA) -> bool:
    """A teacher target is useless to distill against if it has no real tokens.

    The frozen teacher occasionally emits an empty/immediate-eos continuation or pure
    whitespace (e.g. truncated context, odd prompt). Such rows carry no learnable signal —
    the student would be trained toward "say nothing" — so we drop them rather than poison
    the distillation objective.
    """
    ids = example.teacher_target_ids
    text = example.teacher_target_text
    return not ids or not (text or "").strip()


def drop_degenerate_paths(examples: Sequence[CFTrainQA]) -> tuple[list[CFTrainQA], int]:
    """Return (kept, n_dropped) after removing rows with degenerate teacher targets."""
    kept = [ex for ex in examples if not is_degenerate_path(ex)]
    return kept, len(examples) - len(kept)
