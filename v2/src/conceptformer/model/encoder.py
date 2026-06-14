"""ConceptFormer encoder: an entity's 1-hop neighborhood -> ``k`` concept tokens.

A Perceiver/Flamingo-style latent-query resampler (see ``docs/MODEL_DESIGN.md``): ``k`` learned
latent queries cross-attend over the variable-size set of edge features and emit exactly ``k``
output vectors, independent of the neighbor count ``N``. Chosen over a GNN (a depth-1 star makes
message passing redundant) and over an MLP (the input is a variable-size *set*).

Key design properties:
- **Permutation-invariant over neighbors.** No positional encoding is applied to the edge set, and
  cross-attention sums over keys, so the output is invariant to neighbor ordering — neighbors are an
  unordered set. (RoPE positions are assigned later, to the ``k`` *output* slots, not here.)
- **Internal dim decoupled from the LLM.** Edge features are projected ``d_in -> d_model``; a
  single output projection maps ``d_model -> d_llm``. Encoder capacity is tuned independently of
  the 0.6B LLM.
- **Masking.** Padded neighbors (batching ragged neighborhoods) are excluded via a key-padding mask,
  so they cannot influence the output.

The zero-init injection gate that keeps the frozen LLM's behavior intact at step 0 lives in the
injection module, not here — this stays a clean features -> concept-vectors map.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResamplerBlock(nn.Module):
    """One resampler layer: latents cross-attend edges, then self-attend, then FFN (pre-norm).

    Perceiver-style: every block re-attends the *original* edge features, so depth refines the
    summary rather than compressing a compression.
    """

    def __init__(
        self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, latents: Tensor, edges: Tensor, key_padding_mask: Tensor) -> Tensor:
        q = self.q_norm(latents)
        kv = self.kv_norm(edges)
        attn, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )
        latents = latents + attn
        s = self.self_norm(latents)
        self_attn, _ = self.self_attn(s, s, s, need_weights=False)
        latents = latents + self_attn
        return latents + self.ffn(self.ffn_norm(latents))


class ConceptEncoder(nn.Module):
    """Edge features ``(B, N, d_in)`` + mask ``(B, N)`` -> concept tokens ``(B, k, d_llm)``.

    ``d_in`` is the per-edge feature dim (e.g. fused property+neighbor label embeddings); ``d_llm``
    is the frozen LLM's input-embedding dim. ``latent_init`` optionally seeds the ``k`` latent
    queries from real LLM token embeddings (decoupled here so the module stays LLM-agnostic).

    Precondition: every row has at least one valid neighbor (``edge_mask.any(dim=1)`` all True) — a
    fully-masked row would make the cross-attention softmax ill-defined. The snapshot's min-edges
    filter guarantees this for real data.
    """

    def __init__(
        self,
        d_in: int,
        d_llm: int,
        k: int,
        *,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        latent_init: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.k = k
        self.in_proj = nn.Linear(d_in, d_model)
        self.latents = nn.Parameter(torch.empty(k, d_model))
        if latent_init is not None:
            if latent_init.shape != (k, d_model):
                got = tuple(latent_init.shape)
                raise ValueError(f"latent_init must be ({k}, {d_model}), got {got}")
            with torch.no_grad():
                self.latents.copy_(latent_init)
        else:
            nn.init.normal_(self.latents, std=0.02)
        self.blocks = nn.ModuleList(
            ResamplerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_llm)

    def forward(self, edge_features: Tensor, edge_mask: Tensor) -> Tensor:
        """``edge_features``: ``(B, N, d_in)``. ``edge_mask``: ``(B, N)``, True = real neighbor."""
        batch = edge_features.shape[0]
        edges = self.in_proj(edge_features)
        latents = self.latents.unsqueeze(0).expand(batch, self.k, -1).contiguous()
        key_padding_mask = ~edge_mask  # MultiheadAttention masks where True
        for block in self.blocks:
            latents = block(latents, edges, key_padding_mask)
        return self.out_proj(self.out_norm(latents))
