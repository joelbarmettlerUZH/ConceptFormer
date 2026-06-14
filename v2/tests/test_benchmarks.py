"""Tests for the PopQA loader + the hardcoded relation→PID mapping."""

import httpx
import pytest

from conceptformer.data.benchmarks import POPQA_RELATION_TO_PID, popqa_row_to_example


def _row(**overrides) -> dict:
    row = {
        "s_uri": "http://www.wikidata.org/entity/Q42",
        "prop": "occupation",
        "question": "What is Douglas Adams's occupation?",
        "o_uri": "http://www.wikidata.org/entity/Q36180",
        "possible_answers": "['writer', 'author']",  # PopQA stores this as a string
        "s_pop": 142,
        "obj": "writer",
    }
    row.update(overrides)
    return row


def test_popqa_row_parsing():
    ex = popqa_row_to_example(_row())
    assert ex.subject_qid == "Q42"  # parsed from s_uri
    assert ex.relation == "occupation"
    assert ex.relation_id == "P106"  # mapped, not PopQA's internal prop_id
    assert ex.answer_qid == "Q36180"
    assert ex.answer_labels == ["writer", "author"]  # string list parsed
    assert ex.popularity == 142.0


def test_popqa_row_fallbacks():
    ex = popqa_row_to_example(_row(prop="genre", possible_answers=None, s_pop=None, obj="jazz"))
    assert ex.relation_id == "P136"
    assert ex.answer_labels == ["jazz"]  # falls back to obj when no possible_answers
    assert ex.popularity is None


def test_unknown_relation_maps_to_no_pid():
    assert popqa_row_to_example(_row(prop="not_a_real_relation")).relation_id is None


@pytest.mark.integration
def test_relation_pids_actually_match_their_properties():
    # A typo in any PID would silently corrupt graph_supported / fair / per-relation numbers.
    resp = httpx.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbgetentities",
            "ids": "|".join(POPQA_RELATION_TO_PID.values()),
            "props": "labels",
            "languages": "en",
            "format": "json",
        },
        headers={"User-Agent": "ConceptFormer-test"},
        timeout=30,
    )
    entities = resp.json()["entities"]
    for relation, pid in POPQA_RELATION_TO_PID.items():
        label = entities[pid]["labels"]["en"]["value"].lower()
        assert relation.lower() in label, f"{relation} -> {pid} resolved to '{label}'"
