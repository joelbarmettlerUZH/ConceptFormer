"""Turn a ``Subgraph`` into edge-feature tensors for the concept encoder.

Each 1-hop edge ``(property_label, neighbor_label)`` becomes one feature vector by embedding both
label strings with the *frozen LLM's own* embeddings and fusing them (concat). Using the LLM's
embeddings keeps features in a space the LLM already understands and makes the encoder **inductive**
— any new entity/relation with a label works, no per-entity lookup table to cold-start (see
``docs/MODEL_DESIGN.md`` §1). The center entity's label is embedded separately as explicit
conditioning (the concept tokens are *about* that center).

This module is pure tensor assembly: it depends only on an injected ``LabelEmbedder`` callable, so
it unit-tests without loading a multi-GB model. The real Qwen3-backed embedder is wired at assembly
time (it needs the loaded model's embedding matrix + tokenizer).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from conceptformer.schemas import Subgraph


class LabelEmbedder(Protocol):
    """Embed labels into ``(len(texts), d)`` — typically frozen-LLM mean-pooled embeddings."""

    def __call__(self, texts: Sequence[str]) -> Tensor: ...


@dataclass
class SubgraphFeatures:
    """Per-subject features: ``edge_features`` ``(N, d_in)`` and the ``center`` vector ``(d,)``."""

    edge_features: Tensor
    center: Tensor

    @property
    def n_edges(self) -> int:
        return int(self.edge_features.shape[0])


def edge_label_pairs(sg: Subgraph) -> list[tuple[str, str]]:
    """``(property_label, neighbor_label)`` per edge; falls back to ids when a label is missing."""
    return [
        (e.property_label or e.property_id, e.neighbor.label or e.neighbor.qid) for e in sg.edges
    ]


def featurize_subgraph(sg: Subgraph, embedder: LabelEmbedder) -> SubgraphFeatures:
    """Embed every edge's (property, neighbor) labels + the center label.

    Edge feature = ``concat(embed(property), embed(neighbor))`` so a strong relation vector can't
    wash out neighbor identity (concat fusion, not v1's additive). Raises if the subgraph is empty
    (the encoder's cross-attention is undefined with an empty neighbor set; the snapshot's
    min-edges filter guarantees ``N >= 1`` for real data).
    """
    pairs = edge_label_pairs(sg)
    if not pairs:
        raise ValueError(f"subgraph {sg.center.qid} has no edges to featurize")
    props = embedder([p for p, _ in pairs])  # (N, d)
    neighbors = embedder([n for _, n in pairs])  # (N, d)
    edge_features = torch.cat([props, neighbors], dim=-1)  # (N, 2d) concat fusion
    center = embedder([sg.center.label or sg.center.qid])[0]  # (d,)
    return SubgraphFeatures(edge_features=edge_features, center=center)


def featurize_subgraph_recursive(
    sg: Subgraph, embedder: LabelEmbedder, concept_lookup: dict[str, Tensor]
) -> SubgraphFeatures:
    """Like ``featurize_subgraph``, but the neighbor half of each edge is the neighbor's own
    (pooled) concept vector instead of its label embedding.

    This encodes one extra hop *without* flattening: the neighbor's compressed neighborhood stays
    bound to the specific edge, so a 2-hop fact (X --r--> N, N --s--> Z) is represented as
    ``[embed(r); pool(C(N))]`` with Z reachable through ``C(N)`` rather than dumped into X's edge
    bag. Neighbors absent from ``concept_lookup`` (e.g. leaves) fall back to the label embedding,
    so the feature space stays consistent. The relation half and the center are unchanged, so the
    input dimension is still ``2d`` and the encoder needs no resize.
    """
    pairs = edge_label_pairs(sg)
    if not pairs:
        raise ValueError(f"subgraph {sg.center.qid} has no edges to featurize")
    props = embedder([p for p, _ in pairs])  # (N, d)
    neighbor_vecs: list[Tensor | None] = []
    missing_idx, missing_labels = [], []
    for i, (_, neighbor_label) in enumerate(pairs):
        vec = concept_lookup.get(sg.edges[i].neighbor.qid)
        neighbor_vecs.append(vec.to(props.dtype) if vec is not None else None)
        if vec is None:
            missing_idx.append(i)
            missing_labels.append(neighbor_label)
    if missing_labels:  # one batched embed for all label fallbacks
        filled = embedder(missing_labels)
        for j, i in enumerate(missing_idx):
            neighbor_vecs[i] = filled[j]
    neighbors = torch.stack([v for v in neighbor_vecs if v is not None])  # (N, d)
    edge_features = torch.cat([props, neighbors], dim=-1)  # (N, 2d)
    center = embedder([sg.center.label or sg.center.qid])[0]
    return SubgraphFeatures(edge_features=edge_features, center=center)


def collate_features(
    items: Sequence[SubgraphFeatures],
) -> tuple[Tensor, Tensor, Tensor]:
    """Pad a batch of variable-``N`` subgraphs to a dense ``(B, N_max, d_in)`` tensor.

    Returns ``(edge_features, mask, center)`` where ``mask`` ``(B, N_max)`` is True for real edges
    (padding rows are zero-filled and masked out by the encoder). ``center`` is ``(B, d)``.
    """
    if not items:
        raise ValueError("cannot collate an empty batch")
    batch = len(items)
    n_max = max(it.n_edges for it in items)
    d_in = int(items[0].edge_features.shape[-1])
    ref = items[0].edge_features
    edge_features = ref.new_zeros((batch, n_max, d_in))
    mask = torch.zeros((batch, n_max), dtype=torch.bool, device=ref.device)
    for i, it in enumerate(items):
        n = it.n_edges
        edge_features[i, :n] = it.edge_features
        mask[i, :n] = True
    center = torch.stack([it.center for it in items], dim=0)
    return edge_features, mask, center
