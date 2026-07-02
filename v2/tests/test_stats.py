"""Offline tests for the paired-statistics helpers (eval/stats.py)."""

import pytest

from conceptformer.eval.stats import (
    discordant_counts,
    mcnemar_exact_p,
    paired_bootstrap_diff,
    summarize_accuracy,
    wilson_ci,
)


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo0, hi0 = wilson_ci(0, 100)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.05  # stays inside [0, 1] at the extreme


def test_wilson_ci_rejects_bad_counts():
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_mcnemar_exact():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 5) == 1.0  # perfectly balanced discordance
    # b=10, c=0: p = 2 * 0.5^10 ~ 0.00195
    assert mcnemar_exact_p(10, 0) == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact_p(3, 7) == mcnemar_exact_p(7, 3)  # symmetric


def test_mcnemar_large_counts_no_overflow():
    # Full-benchmark comparisons produce ~10^3 discordant pairs; must stay finite + sane.
    p_big_effect = mcnemar_exact_p(576, 340)
    assert 0.0 < p_big_effect < 1e-10
    p_null = mcnemar_exact_p(1500, 1500)
    assert 0.9 < p_null <= 1.0


def test_discordant_counts():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    assert discordant_counts(a, b) == (2, 1)
    with pytest.raises(ValueError):
        discordant_counts([True], [True, False])


def test_paired_bootstrap_diff_deterministic_and_sane():
    a = [True] * 80 + [False] * 20
    b = [True] * 60 + [False] * 40
    mean, lo, hi = paired_bootstrap_diff(a, b, iters=2000, seed=0)
    assert mean == pytest.approx(0.20)
    assert lo <= mean <= hi
    assert (mean, lo, hi) == paired_bootstrap_diff(a, b, iters=2000, seed=0)  # deterministic
    assert paired_bootstrap_diff([], []) == (0.0, 0.0, 0.0)


def test_summarize_accuracy_shape():
    s = summarize_accuracy([True, True, False, False])
    assert s["n"] == 4 and s["correct"] == 2 and s["acc"] == 0.5
    assert s["ci95"][0] < 0.5 < s["ci95"][1]
