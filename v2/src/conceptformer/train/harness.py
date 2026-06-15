"""Train/val splitting for the generalization test.

The central question (``docs/CONCEPTFORMER_V2_EXPLAINED.md`` §8) is whether concept tokens encode an
entity's neighborhood *generally* or merely memorize the training Q->A pairs. The cheap, decisive
first probe is **held-out questions per entity**: the encoder always reads the full neighborhood
(it comes from the snapshot, not the QA rows), so holding out some of an entity's questions tests
whether its concept tokens answer questions they were never trained on.

A positive result (held-out accuracy >> base, approaching the RAG teacher) is evidence the tokens
carry the neighborhood, not a per-instance codec. This split keeps the encoder's input identical
across the split and only varies which *queries* are supervised.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from conceptformer.generate.schema import CFTrainQA

# Only single/compositional rows with a gold answer + teacher path can score accuracy/KL.
_ANSWERABLE = ("single", "compositional")


def is_answerable(row: CFTrainQA) -> bool:
    return bool(row.task_type in _ANSWERABLE and row.answer and row.teacher_target_ids)


def split_by_held_out_questions(
    rows: Sequence[CFTrainQA], *, val_frac: float = 0.3, seed: int = 0
) -> tuple[list[CFTrainQA], list[CFTrainQA]]:
    """Hold out a per-entity fraction of *answerable* questions for validation.

    Descriptive/control rows and non-held-out QA go to train; the held-out QA go to val. Entities
    with too few questions to hold one out contribute only to train. Deterministic given ``seed``.
    """
    by_entity: dict[str, list[CFTrainQA]] = defaultdict(list)
    for row in rows:
        by_entity[row.subject_qid].append(row)

    rng = random.Random(seed)
    train: list[CFTrainQA] = []
    val: list[CFTrainQA] = []
    for qid in sorted(by_entity):  # sorted → deterministic regardless of input order
        members = by_entity[qid]
        qa = [r for r in members if is_answerable(r)]
        other = [r for r in members if not is_answerable(r)]
        rng.shuffle(qa)
        n_val = int(len(qa) * val_frac)
        val.extend(qa[:n_val])
        train.extend(qa[n_val:])
        train.extend(other)
    return train, val
