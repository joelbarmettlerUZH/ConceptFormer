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
from typing import TYPE_CHECKING, Any

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
        # Multimodal wrappers (Qwen3.5 ConditionalGeneration) nest the text stack one level
        # deeper (model.model.language_model); text-only forwards must target it directly so
        # ``forward_hidden`` skips the vision tower.
        base = self.model.model
        self._lm = getattr(base, "language_model", base)
        self.is_multimodal = self._lm is not base

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

    def forward_hidden(
        self, inputs_embeds: Tensor, attention_mask: Tensor, position_ids: Tensor | None = None
    ) -> Tensor:
        """Final hidden states ``(B, T, d)`` — the base transformer *without* the LM head.

        Avoids materializing the ``(B, T, V)`` logits (V≈152k): the caller gathers the few path
        positions first, then applies ``lm_head`` only there. Identical results, far less memory.

        Multimodal stacks use M-RoPE and expect ``(3, B, L)`` position ids. For text-only
        splices our ids are the trivial contiguous ones the stack derives by default, so we
        pass ``None``; explicit 3-D ids (the vision-port path, computed via
        ``mrope_position_ids``) are passed through unchanged.
        """
        if self.is_multimodal:
            position_ids = position_ids if position_ids is not None and position_ids.dim() == 3 \
                else None
        out = self._lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return out.last_hidden_state

    def mrope_position_ids(
        self,
        input_ids: Tensor,
        mm_token_type_ids: Tensor,
        image_grid_thw: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """``(3, B, L)`` M-RoPE ids + per-row continuation deltas, via the model's own indexer.

        The wrapper derives image-grid positions from token IDS, which ``inputs_embeds`` alone
        cannot convey — so the vision-port trainer computes them here explicitly and generation
        continues text positions at ``seq_len + step + delta`` per row.
        """
        if not self.is_multimodal:
            raise RuntimeError("mrope_position_ids requires a multimodal backbone")
        return self.model.model.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

    def forward_cached(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor | None,
        past_key_values: object = None,
    ) -> Any:  # HF ModelOutput (untyped library boundary, like from_pretrained)
        """One KV-cached forward on the text stack — the vision-port incremental decode step.

        The wrapper's ``generate`` cannot be used there (it derives M-RoPE from token ids,
        which spliced ``inputs_embeds`` do not carry), so the trainer drives decoding manually.
        """
        return self._lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

    def lm_head(self, hidden: Tensor) -> Tensor:
        """Project hidden states to vocab logits (position-wise; apply after gathering paths)."""
        return self.model.lm_head(hidden)
