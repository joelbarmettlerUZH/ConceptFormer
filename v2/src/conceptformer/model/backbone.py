"""Frozen-LLM adapter for ConceptFormer training.

Wraps an already-loaded ``ChatModel`` and exposes exactly what the encoder/trainer need from the
*frozen* backbone:

- ``embed_labels`` — the real ``LabelEmbedder``: mean-pool the LLM's input embeddings over each
  label's tokens (inductive, LLM-aligned edge features — see ``docs/MODEL_DESIGN.md`` §1).
- ``embed_tokens`` — input embeddings for prompt/path token ids (the frozen context the concept
  tokens are spliced into).
- ``forward_embeds`` — logits from ``inputs_embeds`` (so the student can run on spliced embeddings).

The backbone is frozen (``requires_grad_(False)``, ``eval``): gradients still flow *through* it to
the concept-token embeddings, but no LLM weights update. The label/token-embedding pooling logic is
a pure function so it unit-tests without a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from conceptformer.model.chat import ChatModel


def pool_label_embeddings(token_ids: list[list[int]], embedding: Tensor) -> Tensor:
    """Mean-pool an embedding matrix over each label's token ids -> ``(len(token_ids), d)``.

    ``embedding``: ``(V, d)``. Empty token lists (a label that tokenizes to nothing) yield a zero
    vector. Pure and device/dtype-agnostic so it tests on a tiny CPU matrix.
    """
    out = embedding.new_zeros((len(token_ids), embedding.shape[1]))
    for i, ids in enumerate(token_ids):
        if ids:
            idx = torch.tensor(ids, dtype=torch.long, device=embedding.device)
            out[i] = embedding.index_select(0, idx).mean(dim=0)
    return out


class Backbone:
    """Thin frozen-LLM interface built from a loaded ``ChatModel`` (no second model load)."""

    def __init__(self, chat: ChatModel) -> None:
        self.model = chat.model
        self.tokenizer = chat.tokenizer
        self.device = chat._device
        self.model.eval()
        self.model.requires_grad_(False)
        self._embedding = self.model.get_input_embeddings()
        self.d_model: int = int(self._embedding.embedding_dim)
        self.dtype = self._embedding.weight.dtype

    @torch.no_grad()
    def embed_labels(self, texts: Sequence[str]) -> Tensor:
        """Frozen mean-pooled label embeddings ``(len(texts), d_model)`` (the edge featurizer)."""
        encoded = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return pool_label_embeddings(encoded, self._embedding.weight)

    @torch.no_grad()
    def embed_tokens(self, ids: Tensor) -> Tensor:
        """Frozen input embeddings for token ids (prompt/path context)."""
        return self._embedding(ids.to(self.device))

    def forward_embeds(
        self, inputs_embeds: Tensor, attention_mask: Tensor, position_ids: Tensor | None = None
    ) -> Tensor:
        """Logits ``(B, T, V)`` from spliced ``inputs_embeds``. Grad flows to concept tokens only
        (LLM weights are frozen); run the teacher pass inside ``torch.no_grad`` at the call site."""
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return out.logits
