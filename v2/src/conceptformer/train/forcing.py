"""Teacher-forcing alignment: gather the logits that predict each target-path token.

The teacher and student sequences share the same appended target path but have *different* prefixes
(verbalized-facts text vs ``k`` concept tokens), so the path sits at different absolute indices in
each. In a causal LM, the logits at position ``t`` predict token ``t+1``; so the distribution that
predicts path token ``j`` (path starting right after a ``context_len``-token prefix) lives at index
``context_len - 1 + j``.

``gather_path_logits`` compacts those positions into ``(B, m_max, V)`` aligned by path index
``j`` for *either* sequence, so teacher and student become comparable and ``sequence_kl`` applies.
All pure tensor indexing — unit-tested without a model.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def path_predicting_indices(context_len: int, path_len: int) -> list[int]:
    """Indices whose logits predict each of the ``path_len`` path tokens after a prefix.

    The first path token is predicted by the last prefix position (``context_len - 1``).
    """
    if context_len < 1:
        raise ValueError("context_len must be >= 1 (need a position that predicts the first token)")
    return [context_len - 1 + j for j in range(path_len)]


def gather_path_logits(
    logits: Tensor, context_lens: Sequence[int], path_lens: Sequence[int]
) -> tuple[Tensor, Tensor]:
    """Compact path-predicting logits to ``(B, m_max, V)`` + a ``(B, m_max)`` validity mask.

    ``logits``: ``(B, T, V)``. For row ``i`` the path is predicted by indices
    ``context_lens[i]-1 .. context_lens[i]+path_lens[i]-2``; short paths are right-padded + masked.
    """
    batch, _, vocab = logits.shape
    m_max = max(path_lens)
    out = logits.new_zeros((batch, m_max, vocab))
    mask = torch.zeros((batch, m_max), dtype=torch.bool, device=logits.device)
    for i, (c_len, p_len) in enumerate(zip(context_lens, path_lens, strict=True)):
        idx = path_predicting_indices(c_len, p_len)
        out[i, :p_len] = logits[i, torch.tensor(idx, dtype=torch.long, device=logits.device)]
        mask[i, :p_len] = True
    return out, mask
