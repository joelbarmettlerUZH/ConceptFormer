"""LR-schedule shape (``lr_factor``) — the pure helper behind the trainer's LambdaLR. Tested without
a model so warmup / cosine / constant behaviour is locked independently of training.
"""

from __future__ import annotations

import math

from conceptformer.train.trainer import lr_factor


def test_linear_warmup_ramps_to_peak() -> None:
    warmup, total = 100, 1000
    assert lr_factor(0, warmup, total, "cosine") == 1 / 100  # first step is non-zero
    assert lr_factor(49, warmup, total, "cosine") == 50 / 100  # half-way up the ramp
    assert lr_factor(99, warmup, total, "cosine") == 100 / 100  # peak reached at end of warmup


def test_cosine_decays_to_zero_at_end() -> None:
    warmup, total = 100, 1000
    assert math.isclose(lr_factor(100, warmup, total, "cosine"), 1.0, abs_tol=1e-9)  # peak
    mid = lr_factor(550, warmup, total, "cosine")  # half-way through decay → ~0.5
    assert math.isclose(mid, 0.5, abs_tol=1e-6)
    assert math.isclose(lr_factor(1000, warmup, total, "cosine"), 0.0, abs_tol=1e-9)  # decays to 0


def test_constant_holds_peak_after_warmup() -> None:
    warmup, total = 100, 1000
    assert lr_factor(100, warmup, total, "constant") == 1.0
    assert lr_factor(550, warmup, total, "constant") == 1.0  # no decay
    assert lr_factor(1000, warmup, total, "constant") == 1.0
    # warmup still applies under constant
    assert lr_factor(49, warmup, total, "constant") == 50 / 100


def test_zero_warmup_is_safe() -> None:
    assert lr_factor(0, 0, 1000, "cosine") == 1.0  # no div-by-zero; starts at peak
