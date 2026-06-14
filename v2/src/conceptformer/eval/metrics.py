"""Answer-matching metrics (all operate on (prediction_text, aliases) — condition-agnostic).

- ``popqa_official`` — the EXACT published PopQA metric (Mallen et al. 2023): correct iff any
  gold alias is a case-insensitive substring of the prediction. Reproduced verbatim (including
  its known leniency) so our headline number is directly comparable to prior work.
- ``word_boundary_match`` — corrected version: an alias occurs at word boundaries, after dropping
  junk aliases shorter than ``MIN_ALIAS_LEN``. Removes false positives (gold "W" matching the
  "w" in "was"; "pol" inside "politics").
- ``strict_match`` — the prediction's normalized first line equals a normalized alias
  (pessimistic; penalizes enumeration / rambling).

The harness combines these with different alias sets to produce ``popqa_official``,
``word_boundary``, ``fair`` (word_boundary over gold + subclass-expanded), and ``strict_em``.
All are deterministic and depend ONLY on (prediction, aliases) — never on the model/condition.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence

# Aliases shorter than this are dropped — PopQA's gold lists contain junk like "W", "CA",
# "ON" (matches "London"/"born on") that produce false positives.
MIN_ALIAS_LEN = 3


def popqa_official(prediction: str, aliases: Sequence[str]) -> bool:
    """Exact PopQA reference metric: any alias is a case-insensitive substring of the prediction.

    Faithfully reproduced (including its over-leniency on short aliases) so numbers are directly
    comparable to published PopQA results. For a corrected metric, use ``word_boundary_match``.
    """
    pred = prediction.lower()
    return any(alias.lower() in pred for alias in aliases)


def _candidates(aliases: Sequence[str]) -> Iterator[str]:
    for alias in aliases:
        normalized = alias.strip().lower()
        if len(normalized) >= MIN_ALIAS_LEN:
            yield normalized


def word_boundary_match(prediction: str, aliases: Sequence[str]) -> bool:
    """True if any (hygiened) alias appears in the prediction bounded by non-word chars."""
    pred = prediction.lower()
    return any(
        re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", pred) for alias in _candidates(aliases)
    )


def _normalize(text: str) -> str:
    first_line = text.lower().strip().split("\n", 1)[0]
    no_punct = re.sub(r"[^\w\s]", " ", first_line)
    no_articles = re.sub(r"\b(?:a|an|the)\b", " ", no_punct)
    return re.sub(r"\s+", " ", no_articles).strip()


def strict_match(prediction: str, aliases: Sequence[str]) -> bool:
    """Normalized first-line of the prediction exactly equals a normalized alias."""
    pred = _normalize(prediction)
    return bool(pred) and any(_normalize(alias) == pred for alias in aliases if alias)
