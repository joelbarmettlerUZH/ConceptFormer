"""Vision-port injection: splice concept tokens through the frozen VLM's IMAGE-token interface.

Hypothesis (grant aim O4): natively-multimodal LLMs are pretrained to consume continuous,
out-of-vocabulary token streams between ``<vision_start>``/``<vision_end>``; that interface may
be a better landing pad for soft concept tokens than the text-embedding stream, where every
pretraining token was a row of the embedding matrix. The comparison is only meaningful if the
splice is *faithful* to what a real k-patch image produces, i.e. all three of:

1. ids: ``[head][vision_start][image_token x k][vision_end][tail]``;
2. embeddings: ``embed(ids)`` with the k image-token positions overwritten by concept vectors
   (exactly where the vision tower's projected patches would land);
3. positions: the model's own ``get_rope_index`` over those ids with a ``(1, 1, k)`` pseudo-image
   grid and ``mm_token_type_ids`` = 1 on the image span — M-RoPE grid positions, not text
   positions (the wrapper cannot derive these from ``inputs_embeds`` alone, which is why the
   trainer computes them explicitly and generation uses an incremental loop).

Pure id/mask assembly lives here (unit-testable without a model); the trainer wires it to the
backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class VisionPort:
    """The three special-token ids that delimit an image block in the chat sequence."""

    vision_start_id: int
    image_token_id: int
    vision_end_id: int

    @classmethod
    def from_config(cls, config: object) -> VisionPort:
        """Read the ids from a multimodal model config; raises if any is missing."""
        ids = {}
        for name in ("vision_start_token_id", "image_token_id", "vision_end_token_id"):
            value = getattr(config, name, None)
            if value is None:
                raise ValueError(f"model config has no {name}; not a vision-port backbone")
            ids[name] = int(value)
        return cls(
            vision_start_id=ids["vision_start_token_id"],
            image_token_id=ids["image_token_id"],
            vision_end_id=ids["vision_end_token_id"],
        )


def vision_block_ids(port: VisionPort, k: int) -> list[int]:
    """``[vision_start][image_token x k][vision_end]`` — the id-level image block."""
    return [port.vision_start_id] + [port.image_token_id] * k + [port.vision_end_id]


def student_ids_with_block(
    head_ids: list[int], tail_ids: list[int], port: VisionPort, k: int
) -> tuple[list[int], int]:
    """Full student id sequence and the index of the first concept slot.

    The concept vectors later overwrite positions ``start .. start+k`` (the image tokens);
    the delimiters keep their real embeddings — they are text-side interface, not content.
    """
    ids = head_ids + vision_block_ids(port, k) + tail_ids
    return ids, len(head_ids) + 1


def mm_token_type_ids(input_ids: Tensor, port: VisionPort) -> Tensor:
    """``(B, L)`` int tensor: 1 on image tokens, 0 elsewhere (text) — get_rope_index's contract."""
    return (input_ids == port.image_token_id).int()


def image_grids(
    n_rows: int, k: int, device: torch.device | str, merge: int = 2
) -> Tensor:
    """``(n_rows, 3)`` grid tensor whose LLM-side image span is exactly ``1 x k`` tokens.

    ``image_grid_thw`` is in VISION-PATCH units; the LLM sees ``t * (h/merge) * (w/merge)``
    tokens after spatial merging (``merge`` = the vision config's ``spatial_merge_size``). A
    ``(1, merge, merge*k)`` patch grid therefore lands k merged tokens laid out as one row —
    the M-RoPE geometry of a wide, single-row image.
    """
    return torch.tensor([[1, merge, merge * k]] * n_rows, dtype=torch.long, device=device)


def overwrite_image_slots(
    embeds: Tensor, concepts: Tensor, starts: list[int]
) -> Tensor:
    """Replace each row's k image-token embeddings with its concept vectors (grad-preserving).

    ``embeds``: ``(B, L, d)`` from the embedding table; ``concepts``: ``(B, k, d)`` with grad;
    ``starts``: per-row index of the first image slot. Uses concatenation (not in-place writes)
    so autograd flows to the concept encoder.
    """
    batch, _, _ = embeds.shape
    k = concepts.shape[1]
    rows = []
    for i in range(batch):
        s = starts[i]
        rows.append(torch.cat([embeds[i, :s], concepts[i], embeds[i, s + k :]], dim=0))
    return torch.stack(rows, dim=0)
