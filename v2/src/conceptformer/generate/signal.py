"""Stage 4: signal-tier CF-Train QA by whether the graph demonstrably helps.

For each QA example we run the backbone twice — base (question only) and RAG (question +
verbalized neighborhood) — and compare each answer to the grounded gold:

- ``needs_graph``  base FAILS, RAG SUCCEEDS  → the graph demonstrably supplies the answer:
  pure knowledge-injection signal, the ideal training example.
- ``base_knows``   base SUCCEEDS             → parametric knowledge; low injection signal,
  kept only as a small "easy" fraction for distribution diversity.
- dropped          base fails AND RAG fails  → even the teacher can't produce it from the
  graph, so its distillation target would be wrong/uninformative.

Descriptive tasks aren't tiered here (no single gold) — they're always kept (the teacher
always produces a description).
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from conceptformer.eval.metrics import word_boundary_match
from conceptformer.generate.schema import CFTrainQA


def answer_ok(prediction: str, answers: str | Sequence[str] | None) -> bool:
    """True if the prediction matches ANY accepted answer surface form (alias-aware).

    Accepts a single gold string or a list of surface forms (Gemma's verbatim answer + the
    matched neighbor's canonical label), so a correct-but-reworded model output isn't judged
    wrong the way a single-string match would. Mirrors the eval harness's alias matching.
    """
    if not answers:
        return False
    aliases = [answers] if isinstance(answers, str) else list(answers)
    return bool(aliases) and word_boundary_match(prediction, aliases)


def assign_tier(base_ok: bool, rag_ok: bool) -> str | None:
    """Return the signal tier, or ``None`` to drop (the 'hard' case)."""
    if base_ok:
        return "base_knows"
    if rag_ok:
        return "needs_graph"
    return None


def apply_keep_policy(
    tiered: Sequence[tuple[CFTrainQA, str | None]],
    *,
    keep_base_known_frac: float = 0.3,
    seed: int = 0,
) -> list[CFTrainQA]:
    """Keep all ``needs_graph``, a fraction of ``base_knows``; drop ``None`` (hard)."""
    rng = random.Random(seed)
    kept: list[CFTrainQA] = []
    for row, tier in tiered:
        if tier == "needs_graph" or (tier == "base_knows" and rng.random() < keep_base_known_frac):
            kept.append(row.model_copy(update={"tier": tier}))
    return kept
