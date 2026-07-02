"""Frozen evaluation sets — eval sampling decoupled from the training seed (pure; no GPU).

The historical defect this module fixes (RESEARCH_FINDINGS methods note M7): eval subsets were
sampled with the TRAINING seed (and from an RNG whose state depended on unrelated earlier
draws), so every seed/config scored a *different* n=200 question sample. Across-seed spread
therefore conflated model variance with eval-set sampling noise (~3.5 pt binomial at n=200),
and no two runs could be compared with paired tests. A second latent hazard: PopQA is
relation-GROUPED on disk (the first ~1000 rows are only "occupation" and "place of birth"), so
any unshuffled slice — e.g. a ``--limit`` snapshot — silently measures a 2-of-16-relation
subset.

Everything here samples with one fixed, documented seed (``EVAL_SAMPLE_SEED``) after sorting by
a stable key, so the same (dataset, n) always yields the SAME items for every run — making
per-item dumps pairable across checkpoints/configs/seeds (see ``eval/stats.py``).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from conceptformer.schemas import QAExample, Subgraph

# Fixed seed for ALL eval-subset sampling, deliberately independent of the training --seed.
EVAL_SAMPLE_SEED = 20260702

# (subgraph, question, accepted answers, answer qid) — the tuple evaluate_popqa consumes.
PopQAItem = tuple[Subgraph, str, list[str], str | None]


def popqa_eval_items(
    examples: Sequence[QAExample],
    subgraphs: Mapping[str, Subgraph],
    n: int = 0,
) -> list[PopQAItem]:
    """The frozen PopQA eval set: all scoreable rows, fixed-seed shuffled, first ``n`` (0 = all).

    Scoreable = subject has a non-empty snapshot neighborhood and gold aliases exist. Sorting by
    (subject_qid, question) before the shuffle makes the result independent of the benchmark's
    on-disk row order (the source of the relation-grouping bug).
    """
    usable = [
        e
        for e in examples
        if e.answer_labels and e.subject_qid in subgraphs and subgraphs[e.subject_qid].edges
    ]
    usable.sort(key=lambda e: (e.subject_qid, e.question))
    random.Random(EVAL_SAMPLE_SEED).shuffle(usable)
    if n > 0:
        usable = usable[:n]
    return [
        (subgraphs[e.subject_qid], e.question, list(e.answer_labels), e.answer_qid)
        for e in usable
    ]


def sample_rows(rows: Sequence, n: int, key: str = "question") -> list:
    """Fixed-seed sample of ``n`` eval rows (0 or n >= len = all), stable across training seeds.

    ``key`` names a row attribute used (with ``subject_qid``) to sort before shuffling, so the
    sample does not depend on the caller's row order.
    """
    ordered = sorted(rows, key=lambda r: (r.subject_qid, getattr(r, key)))
    random.Random(EVAL_SAMPLE_SEED).shuffle(ordered)
    if 0 < n < len(ordered):
        return ordered[:n]
    return ordered
