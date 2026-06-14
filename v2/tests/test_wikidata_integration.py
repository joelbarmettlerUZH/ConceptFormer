"""Live-API integration tests (deselect with -m 'not integration')."""

import asyncio

import pytest

from conceptformer.data.wikidata import AsyncWikidataClient, WikidataClient


@pytest.mark.integration
def test_fetch_q42_neighborhood_is_complete():
    with WikidataClient() as client:
        sg = client.fetch_neighborhood("Q42")  # no cap → complete neighborhood
    assert sg is not None
    assert sg.center.label == "Douglas Adams"  # label lives under `mul`, must resolve
    assert sg.n_edges_total > 20
    assert len(sg.edges) == sg.n_edges_total  # nothing dropped
    assert sg.capped is False
    assert all(e.property_label and e.neighbor.label for e in sg.edges)


@pytest.mark.integration
def test_missing_entity_returns_none():
    with WikidataClient() as client:
        assert client.fetch_neighborhood("Q0") is None


@pytest.mark.integration
def test_async_fetch_many_matches_sync():
    async def run():
        async with AsyncWikidataClient() as client:
            return await client.fetch_many(["Q42", "Q1"])

    out = asyncio.run(run())
    assert set(out) == {"Q42", "Q1"}
    assert out["Q42"].center.label == "Douglas Adams"
    # complete by default: every kept edge is accounted for in the total
    assert all(len(sg.edges) == sg.n_edges_total for sg in out.values())
