"""Offline tests for distillation losses (hand-checked small cases)."""

import math

import pytest

torch = pytest.importorskip("torch")

from conceptformer.train.losses import sequence_cross_entropy, sequence_kl  # noqa: E402


def test_kl_is_zero_when_distributions_match():
    logits = torch.randn(2, 3, 7)
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert torch.allclose(sequence_kl(logits, logits.clone(), mask), torch.tensor(0.0), atol=1e-6)


def test_kl_matches_hand_computed_two_class():
    # teacher p=[0.5,0.5] (logits equal); student p=[0.731,0.269] (logits [1,0]).
    student = torch.tensor([[[1.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 0.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    ps = torch.softmax(student[0, 0], dim=-1)
    expected = 0.5 * math.log(0.5 / ps[0]) + 0.5 * math.log(0.5 / ps[1])
    assert torch.allclose(sequence_kl(student, teacher, mask), torch.tensor(expected), atol=1e-5)


def test_kl_only_counts_masked_positions():
    student = torch.randn(1, 4, 5)
    teacher = torch.randn(1, 4, 5)
    full = torch.ones(1, 4, dtype=torch.bool)
    half = torch.tensor([[True, True, False, False]])
    # corrupting only the unmasked tail must not change a half-masked loss
    # (shift a single class, not all — adding a constant to every logit leaves softmax unchanged)
    s2 = student.clone()
    s2[0, 2:, 0] += 10.0
    assert torch.allclose(sequence_kl(student, teacher, half), sequence_kl(s2, teacher, half))
    assert not torch.allclose(sequence_kl(student, teacher, full), sequence_kl(s2, teacher, full))


def test_kl_teacher_is_detached_gradient_flows_only_to_student():
    student = torch.randn(1, 2, 6, requires_grad=True)
    teacher = torch.randn(1, 2, 6, requires_grad=True)
    sequence_kl(student, teacher, torch.ones(1, 2, dtype=torch.bool)).backward()
    assert student.grad is not None
    assert teacher.grad is None  # teacher detached → fixed target


def test_temperature_scales_by_T_squared_at_zero_kl_stays_zero():
    logits = torch.randn(1, 2, 4)
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert torch.allclose(sequence_kl(logits, logits.clone(), mask, temperature=2.0),
                          torch.tensor(0.0), atol=1e-6)


def test_cross_entropy_perfect_prediction_is_low():
    # near-one-hot logits on the gold id → CE ≈ 0
    logits = torch.full((1, 1, 5), -10.0)
    logits[0, 0, 3] = 10.0
    ce = sequence_cross_entropy(logits, torch.tensor([[3]]), torch.ones(1, 1, dtype=torch.bool))
    assert ce < 1e-3


def test_cross_entropy_respects_mask():
    logits = torch.randn(1, 3, 5)
    ids = torch.tensor([[0, 1, 2]])
    none_mask = torch.zeros(1, 3, dtype=torch.bool)
    # no supervised positions → denom clamps to 1, numerator 0 → exactly 0
    assert sequence_cross_entropy(logits, ids, none_mask) == 0.0
