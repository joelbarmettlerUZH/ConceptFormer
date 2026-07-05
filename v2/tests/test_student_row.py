"""Regression tests for student-sequence assembly (context length excludes the path).

The context length feeds ``gather_path_logits``: over-counting it by the path length indexes
past the packed sequence (a CUDA device-side assert in training) — exactly the bug these pin.
"""

from types import SimpleNamespace
from typing import Any, cast

import torch

from conceptformer.model.vision_port import VisionPort
from conceptformer.train.trainer import ConceptTrainer

HEAD, TAIL, PATH = [1, 2, 3], [4, 5], [6, 7, 8]


def _fake(k: int, vport: VisionPort | None) -> Any:
    # A stub `self` for the unbound method: _student_row touches only cfg.k / _vport / _embed,
    # so the test runs without a GPU-backed trainer. Cast keeps the call site type-clean.
    return cast(
        ConceptTrainer,
        SimpleNamespace(
            cfg=SimpleNamespace(k=k),
            _vport=vport,
            _embed=lambda ids: torch.zeros((len(ids), 4)),
        ),
    )


def test_text_port_ctx_excludes_path():
    emb, ctx, ids = ConceptTrainer._student_row(
        _fake(2, None), HEAD, TAIL, PATH, torch.zeros((2, 4))
    )
    assert ctx == len(HEAD) + 2 + len(TAIL)  # NOT + len(PATH)
    assert emb.shape[0] == ctx + len(PATH)
    assert ids == []


def test_vision_port_ctx_counts_delimiters_and_ids_cover_whole_row():
    port = VisionPort(vision_start_id=100, image_token_id=101, vision_end_id=102)
    emb, ctx, ids = ConceptTrainer._student_row(
        _fake(2, port), HEAD, TAIL, PATH, torch.zeros((2, 4))
    )
    assert ctx == len(HEAD) + 2 + 2 + len(TAIL)  # + vision_start/vision_end
    assert emb.shape[0] == ctx + len(PATH)
    # ids must cover the FULL row (M-RoPE positions are derived for path tokens too).
    assert ids == [*HEAD, 100, 101, 101, 102, *TAIL, *PATH]
    assert len(ids) == emb.shape[0]


def test_generation_call_shape_no_path():
    port = VisionPort(vision_start_id=100, image_token_id=101, vision_end_id=102)
    emb, ctx, ids = ConceptTrainer._student_row(
        _fake(2, port), HEAD, TAIL, [], torch.zeros((2, 4))
    )
    assert emb.shape[0] == ctx == len(ids)
