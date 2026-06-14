"""Splice concept tokens into the frozen LLM's input sequence.

Three primitives the trainer/predictor compose (see ``docs/MODEL_DESIGN.md`` §2-3):

- ``ConceptGate`` — a zero-init, tanh, per-token gate on the concept embeddings. At step 0 the
  gate is 0, so the injected vectors are zero and the student's distribution is *exactly* the
  frozen model's. Training ramps the gate in smoothly. This is the strongest capability-preservation
  / "no-harm" regularizer and stabilizes early KL.
- ``splice_sequence`` — assembles one student sequence as ``[prefix][k concepts][suffix]``: the
  concept block sits in the same slot the verbalized-facts text occupied for the teacher, so teacher
  and student differ only by graph-text vs concept-tokens and their question tokens land at matching
  positions.
- ``pack_embeddings`` / ``build_position_ids`` — right-pad a batch of variable-length embedding
  sequences and assign **contiguous, unit-step** RoPE positions (no gaps, no overlap, no
  position-sharing — all three are empirically harmful for a frozen RoPE model).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class ConceptGate(nn.Module):
    """Per-concept-token tanh gate, initialized to 0 (student == frozen model at init)."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(k))

    def forward(self, concepts: Tensor) -> Tensor:
        """``concepts`` ``(B, k, d)`` -> gated ``(B, k, d)`` = ``tanh(gate) * concepts``."""
        return torch.tanh(self.gate).view(1, -1, 1) * concepts

    def gate_values(self) -> Tensor:
        """Current ``tanh(gate)`` per concept token (for logging how far each has turned on)."""
        return torch.tanh(self.gate).detach()


def splice_sequence(prefix: Tensor, concepts: Tensor, suffix: Tensor) -> Tensor:
    """One sequence ``[prefix][concepts][suffix]`` along the time axis. All are ``(L_i, d)``."""
    return torch.cat([prefix, concepts, suffix], dim=0)


def pack_embeddings(sequences: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Right-pad a batch of ``(L_i, d)`` embedding sequences to ``(B, L_max, d)``.

    Returns ``(inputs_embeds, attention_mask)`` with ``attention_mask`` ``(B, L_max)`` = 1 on real
    tokens. Right-padding is fine for teacher-forced training: pad tokens are attention-masked and
    excluded from the loss.
    """
    if not sequences:
        raise ValueError("cannot pack an empty batch")
    batch = len(sequences)
    lengths = [int(s.shape[0]) for s in sequences]
    l_max = max(lengths)
    d = int(sequences[0].shape[-1])
    ref = sequences[0]
    inputs_embeds = ref.new_zeros((batch, l_max, d))
    attention_mask = torch.zeros((batch, l_max), dtype=torch.long, device=ref.device)
    for i, (seq, n) in enumerate(zip(sequences, lengths, strict=True)):
        inputs_embeds[i, :n] = seq
        attention_mask[i, :n] = 1
    return inputs_embeds, attention_mask


def build_position_ids(attention_mask: Tensor) -> Tensor:
    """Contiguous, unit-step position ids from an attention mask (robust to left/right padding).

    Real tokens get ``0, 1, 2, …`` with no gaps; padding positions are set to 0 (ignored anyway).
    This realizes the design rule for the injected concept block: it occupies a contiguous unit-step
    RoPE span with the suffix continuing monotonically — never a reserved gap, overlap, or shared
    position.
    """
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.clamp(min=0).masked_fill(attention_mask == 0, 0)
