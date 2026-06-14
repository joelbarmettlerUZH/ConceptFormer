"""The trainable ConceptFormer core: neighborhood edge features -> gated concept tokens.

Composes the two trainable modules — the latent-query ``ConceptEncoder`` and the zero-init
``ConceptGate`` — into the single ``nn.Module`` the trainer optimizes (the frozen LLM lives
in ``Backbone``). Splicing the concept tokens into the frozen LLM and running the forward pass is
orchestration done by the trainer/predictor with the ``injection`` helpers; this module only
produces the ``(B, k, d_llm)`` concept embeddings ready to inject.
"""

from __future__ import annotations

from torch import Tensor, nn

from conceptformer.model.encoder import ConceptEncoder
from conceptformer.model.injection import ConceptGate


class ConceptFormer(nn.Module):
    """Edge features ``(B,N,d_in)`` + mask ``(B,N)`` -> gated concept tokens ``(B,k,d_llm)``."""

    def __init__(
        self,
        d_in: int,
        d_llm: int,
        k: int,
        *,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.encoder = ConceptEncoder(
            d_in, d_llm, k, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout
        )
        self.gate = ConceptGate(k)

    def forward(self, edge_features: Tensor, edge_mask: Tensor) -> Tensor:
        """Encode the neighborhood, then apply the zero-init gate (concepts == 0 at step 0)."""
        return self.gate(self.encoder(edge_features, edge_mask))
