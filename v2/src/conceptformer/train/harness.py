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

    CAVEAT (kept for legacy-checkpoint reproduction; prefer ``split_by_held_out_facts``): this
    splits by question ROW, so two paraphrases asking about the same (entity, fact) can land one
    in train and one in val — "held-out" then partly measures paraphrase robustness, not query
    generalization. Use ``strict_val_subset`` to correct old splits at eval time.
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


def fact_key(row: CFTrainQA) -> tuple[str, str]:
    """The (entity, fact) identity a question is about — the unit that must not straddle splits.

    ``answer_qid`` identifies the neighbor of the answering edge; when absent (rare for
    answerable rows) the normalized answer text stands in, so paraphrases with the same gold
    still group together.
    """
    answer = (row.answer or "").strip().lower()
    return (row.subject_qid, row.answer_qid or answer)


def split_by_held_out_facts(
    rows: Sequence[CFTrainQA], *, val_frac: float = 0.3, seed: int = 0
) -> tuple[list[CFTrainQA], list[CFTrainQA]]:
    """Hold out a per-entity fraction of *fact groups* (see ``fact_key``) for validation.

    All paraphrases/compositions sharing a fact move together, so a val question's fact is never
    supervised in train — the split measures query generalization, not paraphrase matching.
    Group counts per entity are small (~9 questions over fewer facts), so ``val_frac`` of groups
    approximates ``val_frac`` of questions. Deterministic given ``seed``.
    """
    by_entity: dict[str, list[CFTrainQA]] = defaultdict(list)
    for row in rows:
        by_entity[row.subject_qid].append(row)

    rng = random.Random(seed)
    train: list[CFTrainQA] = []
    val: list[CFTrainQA] = []
    for qid in sorted(by_entity):
        members = by_entity[qid]
        other = [r for r in members if not is_answerable(r)]
        groups: dict[tuple[str, str], list[CFTrainQA]] = defaultdict(list)
        for r in members:
            if is_answerable(r):
                groups[fact_key(r)].append(r)
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = int(len(keys) * val_frac)
        for i, key in enumerate(keys):
            (val if i < n_val else train).extend(groups[key])
        train.extend(other)
    return train, val


def strict_val_subset(
    train_rows: Sequence[CFTrainQA], val_rows: Sequence[CFTrainQA]
) -> list[CFTrainQA]:
    """Val rows whose (entity, fact) never appears in train — the leakage-free held-out set.

    For re-reporting checkpoints trained under the legacy question-level split without
    retraining: filtering val by fact overlap is equivalent to having grouped by fact upfront,
    just with a smaller resulting set.
    """
    trained_facts = {fact_key(r) for r in train_rows if is_answerable(r)}
    return [r for r in val_rows if is_answerable(r) and fact_key(r) not in trained_facts]
