"""Offline test: the async extractor skips a permanently-failing batch AND accounts for it,
so an incomplete snapshot is never silently reported as complete.
"""

import asyncio

import httpx

from conceptformer.config import Settings
from conceptformer.data.wikidata import AsyncWikidataClient


def test_failed_batch_is_skipped_and_counted(tmp_path, monkeypatch):
    cfg = Settings(data_root=tmp_path, max_retries=1, retry_max_wait_s=0.01)

    async def boom(self, ids, props):
        raise httpx.HTTPError("simulated transient failure")

    monkeypatch.setattr(AsyncWikidataClient, "_api_get_entities_once", boom)

    async def run():
        async with AsyncWikidataClient(cfg) as client:
            out = await client.fetch_many(["Q1", "Q2"])
            return out, client.n_failed_batches

    out, n_failed = asyncio.run(run())
    assert out == {}  # every center batch failed → no subgraphs produced
    assert n_failed >= 1  # ...and the failure is recorded (→ manifest 'complete': False)
