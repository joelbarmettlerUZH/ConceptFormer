"""Tests for SPARQL subclass-binding parsing (offline) + a guarded live integration check."""

import pytest

from conceptformer.eval.subclass import SUBCLASS_RELATIONS, parse_subclass_bindings


def _binding(value: str) -> dict:
    return {"xLabel": {"type": "literal", "value": value}}


def test_parse_keeps_labels_drops_bare_qids_and_blanks():
    payload = {
        "results": {
            "bindings": [
                _binding("minister"),
                _binding("Q6051446"),  # unresolved label → wikibase:label returns the QID
                _binding("member of parliament"),
                _binding(""),  # empty value dropped
                {"other": {"value": "ignored"}},  # no xLabel key → skipped
            ]
        }
    }
    assert parse_subclass_bindings(payload) == ["minister", "member of parliament"]


def test_parse_empty_payload():
    assert parse_subclass_bindings({}) == []
    assert parse_subclass_bindings({"results": {"bindings": []}}) == []


def test_qid_label_that_is_not_bare_is_kept():
    # only EXACT bare-QID strings are dropped; a label containing a QID stays
    assert parse_subclass_bindings({"results": {"bindings": [_binding("Q-pop band")]}}) == [
        "Q-pop band"
    ]


def test_subclass_relations_are_the_class_like_ones():
    assert frozenset({"occupation", "genre", "religion", "sport", "color"}) == SUBCLASS_RELATIONS


@pytest.mark.integration
def test_live_subclass_query_for_politician():
    from conceptformer.eval.subclass import SubclassExpander

    with SubclassExpander() as expander:
        labels = expander.labels("Q82955")  # politician
    blob = " ".join(labels).lower()
    assert len(labels) > 10
    assert any(term in blob for term in ("politician", "senator", "minister"))
