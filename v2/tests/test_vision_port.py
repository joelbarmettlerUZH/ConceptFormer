"""Offline tests for the vision-port id/mask assembly (model/vision_port.py)."""

from types import SimpleNamespace

import pytest
import torch

from conceptformer.model.vision_port import (
    VisionPort,
    image_grids,
    mm_token_type_ids,
    overwrite_image_slots,
    student_ids_with_block,
    vision_block_ids,
)

PORT = VisionPort(vision_start_id=100, image_token_id=101, vision_end_id=102)


def test_from_config_reads_ids_and_rejects_text_only():
    cfg = SimpleNamespace(
        vision_start_token_id=100, image_token_id=101, vision_end_token_id=102
    )
    assert VisionPort.from_config(cfg) == PORT
    with pytest.raises(ValueError, match="vision_start_token_id"):
        VisionPort.from_config(SimpleNamespace())


def test_block_and_student_ids():
    assert vision_block_ids(PORT, 3) == [100, 101, 101, 101, 102]
    ids, start = student_ids_with_block([1, 2], [3, 4], PORT, k=3)
    assert ids == [1, 2, 100, 101, 101, 101, 102, 3, 4]
    assert start == 3  # first image slot, right after head + vision_start
    assert ids[start : start + 3] == [101, 101, 101]


def test_mm_token_type_ids_marks_only_image_tokens():
    ids = torch.tensor([[1, 100, 101, 101, 102, 3]])
    types = mm_token_type_ids(ids, PORT)
    assert types.tolist() == [[0, 0, 1, 1, 0, 0]]


def test_image_grids_resolve_to_k_llm_tokens():
    g = image_grids(2, 8, torch.device("cpu"), merge=2)
    assert g.tolist() == [[1, 2, 16], [1, 2, 16]]
    # LLM-side token count after spatial merging must equal k for every row.
    for t, h, w in g.tolist():
        assert t * (h // 2) * (w // 2) == 8


def test_overwrite_image_slots_preserves_grad():
    embeds = torch.zeros((2, 6, 4))
    concepts = torch.ones((2, 2, 4), requires_grad=True)
    out = overwrite_image_slots(embeds, concepts, starts=[1, 3])
    assert torch.equal(out[0, 1:3], torch.ones((2, 4)))
    assert torch.equal(out[1, 3:5], torch.ones((2, 4)))
    assert out[0, 0].abs().sum() == 0 and out[0, 3:].abs().sum() == 0
    out.sum().backward()
    assert concepts.grad is not None and concepts.grad.abs().sum() > 0
