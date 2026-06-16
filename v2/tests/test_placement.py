"""Unit tests for concept-token placement (where the k soft tokens sit in the user message)."""

import pytest

from conceptformer.train.trainer import _SENTINEL, place_concept_slot

Q = "In which town was Artur Kosicki born?"
E = "Artur Kosicki"


def test_prefix_puts_slot_at_message_start():
    out = place_concept_slot(Q, E, "prefix")
    assert out == f"{_SENTINEL}\n\n{Q}"


def test_before_entity_inserts_slot_just_before_the_mention():
    out = place_concept_slot(Q, E, "before_entity")
    assert _SENTINEL in out
    assert f"{_SENTINEL} {E}" in out  # slot immediately precedes the entity
    assert E in out  # entity surface form retained


def test_after_entity_inserts_slot_just_after_the_mention():
    out = place_concept_slot(Q, E, "after_entity")
    assert f"{E} {_SENTINEL}" in out
    assert E in out


def test_replace_entity_removes_the_surface_form():
    out = place_concept_slot(Q, E, "replace_entity")
    assert _SENTINEL in out
    assert E not in out  # entity name is gone; concepts must stand in for it
    # the rest of the question survives
    assert "In which town was" in out and "born?" in out


def test_case_insensitive_match():
    out = place_concept_slot("who is artur kosicki really", E, "replace_entity")
    assert _SENTINEL in out
    assert "artur kosicki" not in out  # matched despite case difference


def test_missing_entity_falls_back_to_prefix():
    out = place_concept_slot("Who directed this film?", E, "after_entity")
    assert out == f"{_SENTINEL}\n\nWho directed this film?"


def test_no_label_falls_back_to_prefix():
    out = place_concept_slot(Q, None, "replace_entity")
    assert out == f"{_SENTINEL}\n\n{Q}"


def test_unknown_placement_raises():
    with pytest.raises(ValueError, match="unknown placement"):
        place_concept_slot(Q, E, "somewhere")


def test_exactly_one_slot_in_every_mode():
    for mode in ("prefix", "before_entity", "after_entity", "replace_entity"):
        assert place_concept_slot(Q, E, mode).count(_SENTINEL) == 1
