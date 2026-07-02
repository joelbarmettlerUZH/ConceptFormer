"""Paired statistics for eval comparisons (pure; no model, no GPU).

Why paired: our eval accuracies at n=200 carry ~3.5-pt binomial noise, which swamped several
effects we tried to measure (see RESEARCH_FINDINGS F8/F12 methodology caveats). When two systems
are scored on the SAME items, the paired tests here (McNemar / paired bootstrap) remove the
between-item variance and are far more powerful than comparing two independent accuracy numbers.
Eval commands therefore dump per-item correctness (JSONL) so any two runs can be compared pairwise
after the fact.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%).

    Preferred over the normal approximation because our accuracies sit anywhere in [0, 1] and
    some eval subsets are small; Wilson stays inside [0, 1] and behaves at the extremes.
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant-pair counts.

    ``b`` = items system A got right and B got wrong; ``c`` = the reverse. Concordant pairs
    carry no information about the difference and are ignored (that is the power gain over
    unpaired tests). Exact binomial (not the chi-square approximation) because discordant
    counts are often small in our comparisons.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Log-space: math.comb(n, i) overflows float conversion for the n~10^3 discordant counts a
    # full-benchmark comparison produces (and 0.5**n underflows), so sum exp(log C(n,i) - n ln 2).
    log_half_n = n * math.log(0.5)
    tail = sum(
        math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n)
        for i in range(k + 1)
    )
    return min(1.0, 2.0 * tail)


def discordant_counts(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int]:
    """(a-right-b-wrong, a-wrong-b-right) over paired per-item outcomes."""
    if len(a) != len(b):
        raise ValueError(f"paired sequences must have equal length, got {len(a)} vs {len(b)}")
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    return a_only, b_only


def paired_bootstrap_diff(
    a: Sequence[bool],
    b: Sequence[bool],
    *,
    iters: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(mean_diff, ci_lo, ci_hi) of accuracy(a) - accuracy(b), resampling ITEMS with replacement.

    Percentile interval; deterministic given ``seed``. Complements McNemar with an effect-size
    interval (McNemar only gives significance).
    """
    if len(a) != len(b):
        raise ValueError(f"paired sequences must have equal length, got {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return (0.0, 0.0, 0.0)
    diffs = [int(x) - int(y) for x, y in zip(a, b, strict=True)]
    mean_diff = sum(diffs) / n
    rng = random.Random(seed)
    samples = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    lo = samples[int((alpha / 2) * iters)]
    hi = samples[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (mean_diff, lo, hi)


def summarize_accuracy(flags: Sequence[bool]) -> dict:
    """Accuracy + Wilson CI + n, in the shape eval reports embed."""
    n = len(flags)
    k = sum(bool(f) for f in flags)
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "correct": k,
        "acc": round(k / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
    }
