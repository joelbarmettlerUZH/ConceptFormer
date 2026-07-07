"""Report assembly for the causal probes (cf-graph-faithfulness / cf-capability-preservation).

Pure helpers so the report shape is unit-testable without a GPU. The CLI commands gather raw
per-item flags and write summary.json + items.jsonl under data/analysis/probes/, mirroring the
eval-final / eval-transfer conventions -- before this, probe results existed only in terminal
logs, below the project's own durability bar (every number must be re-derivable from an
artifact, not a scrollback).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from conceptformer.eval.stats import summarize_accuracy


def faithfulness_summary(
    baseline_correct: Sequence[bool],
    base_known: Sequence[bool],
    swap_follow: Sequence[bool],
    swap_stick: Sequence[bool],
    swap_base_known: Sequence[bool],
    ablate_answer_correct: Sequence[bool],
    ablate_other_correct: Sequence[bool],
) -> dict:
    """Assemble the graph-faithfulness report from per-item outcomes.

    ``baseline_correct``/``base_known`` cover ALL probes; the swap_* triple is aligned over the
    swap subset (baseline-correct probes with a type-plausible swap target); the ablate_* lists
    are aligned over their own subsets. The base-UNKNOWN split is derived here rather than in the
    CLI so the confound handling (parametric memory vs graph-reading) is pinned by tests.
    """
    if not len(swap_follow) == len(swap_stick) == len(swap_base_known):
        raise ValueError("swap outcome lists must be aligned")
    unknown = [
        (f, s)
        for f, s, known in zip(swap_follow, swap_stick, swap_base_known, strict=True)
        if not known
    ]
    return {
        "n_probes": len(baseline_correct),
        "baseline_correct": summarize_accuracy(baseline_correct),
        "base_knows": summarize_accuracy(base_known),
        "swap_follow": summarize_accuracy(swap_follow),
        "swap_stick": summarize_accuracy(swap_stick),
        "swap_follow_base_unknown": summarize_accuracy([f for f, _ in unknown]),
        "swap_stick_base_unknown": summarize_accuracy([s for _, s in unknown]),
        "ablate_answer_correct": summarize_accuracy(ablate_answer_correct),
        "ablate_other_correct": summarize_accuracy(ablate_other_correct),
    }


def capability_summary(agree: Sequence[bool], kls: Sequence[float]) -> dict:
    """Assemble the capability-preservation report (greedy agreement + next-token KL spread)."""
    if len(agree) != len(kls):
        raise ValueError(f"paired lists must have equal length, got {len(agree)} vs {len(kls)}")
    return {
        "n_controls": len(agree),
        "greedy_agreement": summarize_accuracy(agree),
        "kl": {
            "mean": round(statistics.fmean(kls), 4) if kls else 0.0,
            "median": round(statistics.median(kls), 4) if kls else 0.0,
            "max": round(max(kls), 4) if kls else 0.0,
        },
    }
