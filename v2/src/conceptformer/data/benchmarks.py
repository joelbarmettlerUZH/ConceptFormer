"""Benchmark loaders (eval-only: never used for training).

PopQA (primary) and EntityQuestions (twin) provide gold subject Wikidata QIDs, so no
entity linking is needed. We normalize each into ``QAExample`` rows and expose the set of
subject QIDs whose neighborhoods we must snapshot.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from conceptformer.schemas import QAExample

# PopQA's `prop_id` is an internal 1..16 index, NOT a Wikidata property id. Map the 16
# PopQA relation names to their actual Wikidata PIDs so we can locate the right edge in a
# subject's subgraph during training/eval.
POPQA_RELATION_TO_PID: dict[str, str] = {
    "occupation": "P106",
    "place of birth": "P19",
    "genre": "P136",
    "father": "P22",
    "mother": "P25",
    "capital": "P36",
    "capital of": "P1376",
    "country": "P17",
    "producer": "P162",
    "director": "P57",
    "screenwriter": "P58",
    "author": "P50",
    "composer": "P86",
    "color": "P462",
    "religion": "P140",
    "sport": "P641",
}


def _qid_from_uri(uri: str | None) -> str | None:
    """'http://www.wikidata.org/entity/Q42' or 'Q42' -> 'Q42'."""
    if not uri:
        return None
    return uri.rstrip("/").rsplit("/", 1)[-1]


def load_popqa(split: str = "test") -> list[QAExample]:
    """Load PopQA (akariasai/PopQA) as normalized QAExamples.

    PopQA ships a single split named 'test'. Each row has the gold triple
    (subj/prop/obj) with Wikidata ids and answer aliases in ``possible_answers``.
    """
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split=split)
    return [popqa_row_to_example(row) for row in ds]


def popqa_row_to_example(row: dict) -> QAExample:
    """Convert one raw PopQA row into a normalized ``QAExample`` (pure — unit-testable)."""
    answers = row.get("possible_answers")
    if isinstance(answers, str):  # stored as a JSON-ish string
        try:
            answers = ast.literal_eval(answers)
        except (ValueError, SyntaxError):
            answers = [answers]
    relation = str(row.get("prop"))
    return QAExample(
        source="popqa",
        split="test",
        subject_qid=_qid_from_uri(row.get("s_uri")) or str(row.get("subj_id")),
        relation=relation,
        relation_id=POPQA_RELATION_TO_PID.get(relation),
        question=str(row.get("question")),
        answer_qid=_qid_from_uri(row.get("o_uri")) or None,
        answer_labels=[str(a) for a in (answers or [])] or [str(row.get("obj"))],
        popularity=float(row["s_pop"]) if row.get("s_pop") is not None else None,
    )


def load_entityquestions(split: str = "test") -> list[QAExample]:
    """EntityQuestions loader.

    NOTE: the released EntityQuestions data (princeton-nlp/EntityQuestions) is per-relation
    JSON and does not always carry subject QIDs directly; wiring this up (incl. resolving
    subject entities) is the next data task. Stubbed to keep the pilot focused on PopQA.
    """
    raise NotImplementedError(
        "EntityQuestions loader not implemented yet — pilot validates on PopQA first."
    )


def subject_qids(examples: Iterator[QAExample]) -> list[str]:
    """Unique subject QIDs (order-preserving) needing a neighborhood snapshot."""
    seen: dict[str, None] = {}
    for ex in examples:
        if ex.subject_qid and ex.subject_qid.startswith("Q"):
            seen.setdefault(ex.subject_qid, None)
    return list(seen)
