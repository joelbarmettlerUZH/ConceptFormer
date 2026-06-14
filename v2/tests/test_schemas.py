import httpx
import pytest

from conceptformer.data.wikidata import _entities_from_response, _truthy_entity_edges
from conceptformer.schemas import Edge, Entity, Subgraph


def test_response_parsing_distinguishes_error_kinds():
    # normal payload → entities pass through
    assert _entities_from_response({"entities": {"Q1": {"x": 1}}}) == {"Q1": {"x": 1}}
    # permanent + expected (id doesn't exist) → empty, no raise
    assert _entities_from_response({"error": {"code": "no-such-entity"}}) == {}
    # unexpected error code → empty (logged), no raise / no retry spin
    assert _entities_from_response({"error": {"code": "whatever"}}) == {}
    # transient maxlag → raise so the retry wrapper backs off
    with pytest.raises(httpx.HTTPError):
        _entities_from_response({"error": {"code": "maxlag", "info": "lag 7s"}})


def test_subgraph_roundtrip():
    sg = Subgraph(
        center=Entity(qid="Q42", label="Douglas Adams", rank=200.0),
        edges=[
            Edge(
                property_id="P19",
                property_label="place of birth",
                neighbor=Entity(qid="Q350", label="Cambridge", rank=50.0),
            )
        ],
        n_edges_total=61,
        capped=False,
    )
    restored = Subgraph.model_validate_json(sg.model_dump_json())
    assert restored == sg
    assert restored.neighbor_qids == ["Q350"]


def test_truthy_filter_prefers_preferred_and_drops_deprecated():
    claims = {
        "P39": [
            {"rank": "deprecated", "mainsnak": _value("Q1")},
            {"rank": "normal", "mainsnak": _value("Q2")},
            {"rank": "preferred", "mainsnak": _value("Q3")},
        ],
        "P106": [
            {"rank": "normal", "mainsnak": _value("Q4")},
            {"rank": "normal", "mainsnak": _value("Q5")},
        ],
        # non-entity (time) value is ignored
        "P569": [
            {"rank": "normal", "mainsnak": {"snaktype": "value", "datavalue": {"type": "time"}}}
        ],
        # somevalue snak ignored
        "P40": [{"rank": "normal", "mainsnak": {"snaktype": "somevalue"}}],
    }
    edges = _truthy_entity_edges(claims)
    assert ("P39", "Q3") in edges
    assert ("P39", "Q2") not in edges  # normal dropped when preferred exists
    assert ("P39", "Q1") not in edges  # deprecated dropped
    assert ("P106", "Q4") in edges and ("P106", "Q5") in edges
    assert all(pid != "P569" for pid, _ in edges)
    assert all(pid != "P40" for pid, _ in edges)


def test_truthy_filter_drops_somevalue_novalue_and_prefers_all_preferred():
    claims = {
        "P1": [{"rank": "normal", "mainsnak": {"snaktype": "somevalue"}}],  # somevalue → drop
        "P2": [{"rank": "normal", "mainsnak": {"snaktype": "novalue"}}],  # novalue → drop
        "P3": [  # two preferred kept; the normal one dropped because preferred exist
            {"rank": "preferred", "mainsnak": _value("Qa")},
            {"rank": "preferred", "mainsnak": _value("Qb")},
            {"rank": "normal", "mainsnak": _value("Qc")},
        ],
    }
    edges = _truthy_entity_edges(claims)
    assert ("P3", "Qa") in edges and ("P3", "Qb") in edges
    assert ("P3", "Qc") not in edges
    assert all(pid not in ("P1", "P2") for pid, _ in edges)


def _value(qid: str) -> dict:
    return {
        "snaktype": "value",
        "datavalue": {"type": "wikibase-entityid", "value": {"entity-type": "item", "id": qid}},
    }
