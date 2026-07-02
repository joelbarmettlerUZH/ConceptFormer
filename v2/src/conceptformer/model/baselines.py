"""Untrained injection baselines — what do concept slots buy WITHOUT a learned encoder?

The trained resampler must beat the cheapest thing that fills the same k slots: the entity's
top-k edges, each embedded with the frozen LLM's own label embeddings and spliced in unchanged.
If that baseline already closes much of the base->teacher gap, the encoder's contribution is
small; if it doesn't, the learned graph->concept mapping is doing real work. Zero parameters,
so it drops into the trainer's eval path in place of a ``ConceptFormer``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TopKMeanEdgeBaseline(nn.Module):
    """Concept token i = mean(property-embedding, neighbor-embedding) of the i-th ranked edge.

    Input matches ``ConceptFormer.forward``: ``edge_features`` ``(B, N, 2d)`` is the featurizer's
    concat(property, neighbor) in the LLM's embedding space, rank-sorted (snapshot PageRank
    order) with padding at the end. Averaging the two halves returns each edge to ``d`` — a
    plain LLM-space vector per fact. Rows with fewer than ``k`` edges repeat the mean over their
    valid edges (an entity-level centroid) rather than injecting zeros.

    NOT permutation-invariant by design: slot i is the i-th most important edge, mirroring what
    budgeted verbalization gives the text teacher.
    """

    gate: nn.Module | None = None  # trainer introspects .gate; an untrained baseline has none

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, edge_features: Tensor, edge_mask: Tensor) -> Tensor:
        batch, n_edges, two_d = edge_features.shape
        d = two_d // 2
        per_edge = (edge_features[..., :d] + edge_features[..., d:]) / 2.0  # (B, N, d)
        mask = edge_mask.to(per_edge.dtype).unsqueeze(-1)
        centroid = (per_edge * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, d)
        out = centroid.unsqueeze(1).expand(batch, self.k, d).clone()
        take = min(self.k, n_edges)
        valid = edge_mask[:, :take].unsqueeze(-1)  # padding is trailing, so [:k] = top-k
        out[:, :take] = torch.where(valid, per_edge[:, :take], out[:, :take])
        return out
