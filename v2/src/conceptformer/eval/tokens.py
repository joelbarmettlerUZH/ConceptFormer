"""Token-efficiency accounting for the eval harness.

ConceptFormer's central claim is token efficiency: injecting an entity's knowledge as ``k``
constant concept tokens instead of a variable-length verbalized neighborhood. To back that with
numbers we record, per example and condition, two quantities:

- ``input_tokens``     — the exact tokenized length of the rendered prompt the model processes
  (the true input cost, including chat-template + system + generation-prompt overhead).
- ``knowledge_tokens`` — the marginal cost of injecting the knowledge: ``0`` for base, the
  verbalized-facts length for RAG, the constant ``k`` for ConceptFormer. The headline
  compression ratio is ``RAG knowledge_tokens / k``.

The aggregate (mean/median/p95/max/total) goes in the eval report; the per-example values go in
the predictions JSONL, so the paper can recompute any framing after the fact.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class _Tokenizes(Protocol):
    def prompt_length(self, system: str, user: str) -> int: ...
    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True)
class TokenRecord:
    input_tokens: int  # exact rendered-prompt length actually fed to the model
    knowledge_tokens: int  # marginal knowledge payload (facts for RAG, k for ConceptFormer, 0 base)
    # Token cost of injecting the FULL (un-budgeted) verbalized neighborhood as text — what
    # text-RAG would pay without the context-window cap. The denominator for the headline
    # "k vs full-text RAG" compression ratio, and it quantifies budget truncation when it
    # exceeds ``knowledge_tokens``. None when the subgraph isn't available.
    uncapped_knowledge_tokens: int | None = None


def _uncapped(model: _Tokenizes, facts: str | None) -> int | None:
    return None if facts is None else model.count_tokens(facts)


def measure_text_prompt_tokens(
    model: _Tokenizes,
    active_prompts: Sequence[tuple[str, str]],
    base_prompts: Sequence[tuple[str, str]],
    uncapped_facts: Sequence[str | None] | None = None,
) -> list[TokenRecord]:
    """Token records for a *text* condition (base/RAG), aligned to the example order.

    ``knowledge_tokens`` is isolated as the active user text minus the base user text, so the
    (longer) RAG system prompt doesn't contaminate it — it measures the facts payload alone.
    For base, active == base → 0. ``uncapped_facts`` (the full verbalized neighborhood per
    example) yields ``uncapped_knowledge_tokens``; pass ``None`` to skip. (ConceptFormer reports
    ``k`` directly; see ``concept_token_records``.)
    """
    facts_seq = uncapped_facts if uncapped_facts is not None else [None] * len(active_prompts)
    records: list[TokenRecord] = []
    for (a_sys, a_user), (_b_sys, b_user), facts in zip(
        active_prompts, base_prompts, facts_seq, strict=True
    ):
        knowledge = max(0, model.count_tokens(a_user) - model.count_tokens(b_user))
        records.append(
            TokenRecord(
                input_tokens=model.prompt_length(a_sys, a_user),
                knowledge_tokens=knowledge,
                uncapped_knowledge_tokens=_uncapped(model, facts),
            )
        )
    return records


def concept_token_records(
    model: _Tokenizes,
    base_prompts: Sequence[tuple[str, str]],
    k: int,
    uncapped_facts: Sequence[str | None] | None = None,
) -> list[TokenRecord]:
    """Token records for the ConceptFormer condition: base prompt + ``k`` concept tokens.

    The knowledge enters as ``k`` injected embeddings (not text), so the input is the base
    prompt length plus a constant ``k`` regardless of neighborhood size — the whole point.
    ``uncapped_knowledge_tokens`` (the full-text RAG cost it replaces) makes the saving explicit.
    """
    facts_seq = uncapped_facts if uncapped_facts is not None else [None] * len(base_prompts)
    return [
        TokenRecord(
            input_tokens=model.prompt_length(s, u) + k,
            knowledge_tokens=k,
            uncapped_knowledge_tokens=_uncapped(model, facts),
        )
        for (s, u), facts in zip(base_prompts, facts_seq, strict=True)
    ]


def _summarize(values: Sequence[int]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 1),
        "median": int(statistics.median(ordered)),
        "p95": p95,
        "max": ordered[-1],
        "min": ordered[0],
        "total": sum(values),
    }


def token_report(records: Sequence[TokenRecord]) -> dict:
    """Aggregate per-example token records into the report's ``tokens`` block."""
    report = {
        "input_tokens": _summarize([r.input_tokens for r in records]),
        "knowledge_tokens": _summarize([r.knowledge_tokens for r in records]),
    }
    uncapped = [
        r.uncapped_knowledge_tokens for r in records if r.uncapped_knowledge_tokens is not None
    ]
    if uncapped:
        report["uncapped_knowledge_tokens"] = _summarize(uncapped)
    return report
