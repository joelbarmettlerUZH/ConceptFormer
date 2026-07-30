"""Distillation losses for ConceptFormer training.

The primary objective is **soft-target KL** over the full vocabulary distribution, teacher-forced
along the teacher's greedy path (paper, Method: objective). We match the teacher's whole
belief state — its "dark knowledge" — not just the answer token, so the concept tokens must
reproduce *how* the frozen-LLM-with-facts distributes probability, position by position.

``KL(P_teacher || P_student)`` is the forward KL (mean-covering). A temperature ``T`` softens both
distributions and the loss is scaled by ``T^2`` (standard Hinton KD) so gradient magnitude is
comparable across temperatures. The teacher is always detached — it is a fixed target.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def sequence_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    target_mask: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Mean per-token ``KL(P_teacher || P_student)`` over supervised positions.

    ``student_logits`` / ``teacher_logits``: ``(B, T, V)``. ``target_mask``: ``(B, T)``, True where
    a path token is being predicted (positions to supervise). Returns a scalar. Positions outside
    the mask never contribute; the mean is over the number of supervised tokens.
    """
    teacher_logits = teacher_logits.detach()
    student_logp = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    teacher_p = teacher_logp.exp()
    kl_per_pos = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)  # (B, T)
    mask = target_mask.to(kl_per_pos.dtype)
    denom = mask.sum().clamp(min=1.0)
    loss = (kl_per_pos * mask).sum() / denom
    return loss * (temperature**2)


def sequence_cross_entropy(
    student_logits: Tensor,
    target_ids: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Mean hard cross-entropy of the path tokens — the optional factuality anchor (ablate).

    ``student_logits``: ``(B, T, V)``; ``target_ids``: ``(B, T)`` the gold path token at each
    position; ``target_mask``: ``(B, T)``. Secondary to ``sequence_kl``; kept small if used at all.
    """
    student_logp = F.log_softmax(student_logits, dim=-1)
    gold_logp = student_logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
    mask = target_mask.to(gold_logp.dtype)
    denom = mask.sum().clamp(min=1.0)
    return -(gold_logp * mask).sum() / denom
