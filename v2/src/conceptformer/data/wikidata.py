"""Wikidata client: extract 1-hop *truthy* neighborhoods via the wbgetentities API.

Endpoint-agnostic by design — phase 1 uses the public REST API; swapping to a local
qEndpoint later only changes the ``_api_get_entities*`` methods. Results are cached on disk
(sqlite) so re-runs are deterministic and don't re-hit the API.

Two clients share the same pure subgraph-building logic:
- ``WikidataClient``       — synchronous, one entity at a time (CLI / tests).
- ``AsyncWikidataClient``  — batched + concurrent, dedups neighbor fetches across all
  subjects; this is the path used to snapshot tens of thousands of entities.

"Truthy" semantics (matching Wikidata ``wdt:`` direct claims): for each property keep the
best-rank statements (preferred if any exist, else normal), drop deprecated, and keep only
``value`` snaks whose value is an entity (Q-id).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping, Sequence

import httpx
from tenacity import AsyncRetrying, Retrying, stop_after_attempt, wait_exponential

from conceptformer.cache import KVCache
from conceptformer.config import Settings, settings
from conceptformer.schemas import Edge, Entity, Subgraph

logger = logging.getLogger(__name__)

_FULL_PROPS = "labels|descriptions|sitelinks|claims"  # for centers (needs claims)
_LITE_PROPS = "labels|descriptions|sitelinks"  # for neighbors / properties


def _entities_from_response(data: dict) -> dict[str, dict]:
    """Parse a wbgetentities response into {qid: entity}, distinguishing error kinds.

    - ``maxlag`` is transient → raise so the retry wrapper backs off and retries.
    - ``no-such-entity`` is permanent and expected (the id doesn't exist) → return {}
      so the caller treats it as missing.
    - any other error is unexpected → log and return {} (don't spin on it).
    """
    err = data.get("error")
    if err is not None:
        code = err.get("code")
        if code == "maxlag":
            raise httpx.HTTPError(f"maxlag: {err.get('info')}")
        if code != "no-such-entity":
            logger.warning("Wikidata API error: %s", err)
        return {}
    return data.get("entities", {})


# --------------------------------------------------------------------------------------
# Pure helpers (no I/O) — shared by the sync and async clients.
# --------------------------------------------------------------------------------------
def _truthy_entity_edges(claims: dict) -> list[tuple[str, str]]:
    """Return [(property_id, neighbor_qid)] for truthy item-valued statements."""
    edges: list[tuple[str, str]] = []
    for pid, statements in claims.items():
        ranks = {s.get("rank") for s in statements}
        keep = "preferred" if "preferred" in ranks else "normal"
        for s in statements:
            if s.get("rank") != keep:
                continue
            snak = s.get("mainsnak", {})
            if snak.get("snaktype") != "value":
                continue
            dv = snak.get("datavalue", {})
            if dv.get("type") != "wikibase-entityid":
                continue
            val = dv.get("value", {})
            if val.get("entity-type") != "item":
                continue
            edges.append((pid, val["id"]))
    return edges


def _dedup_edges(edges: Iterable[tuple[str, str]], center_qid: str) -> list[tuple[str, str]]:
    """Drop self-loops and duplicate (property, neighbor) pairs, preserving order."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for pid, nid in edges:
        if nid == center_qid or (pid, nid) in seen:
            continue
        seen.add((pid, nid))
        out.append((pid, nid))
    return out


def _localized(entity: dict, field: str, langs: Sequence[str]) -> str | None:
    """First non-empty value of ``field`` (labels/descriptions) across ``langs``."""
    values = entity.get(field, {})
    for lang in langs:
        v = values.get(lang, {}).get("value")
        if v:
            return v
    return None


def _sitelink_count(entity: dict) -> float:
    return float(len(entity.get("sitelinks", {})))


def _build_subgraph(
    center_qid: str,
    center_raw: dict,
    neighbor_meta: dict[str, dict],
    property_meta: dict[str, dict],
    *,
    cap: int | None = None,
    always_keep: Iterable[str] = (),
    langs: Sequence[str],
    pagerank: Mapping[str, float] | None = None,
) -> Subgraph:
    """Assemble a rank-sorted ``Subgraph`` from already-fetched raw entities.

    Neighbors are ordered by importance: a global ``pagerank`` score when available, else a
    sitelink-count fallback. This ordering decides which neighbors survive a ``cap`` / the
    teacher's token budget, so it should match the (PageRank) measure used for entity selection.
    ``cap`` is optional: ``None`` keeps the COMPLETE neighborhood (the default).
    """
    pagerank = pagerank or {}
    edges_raw = _dedup_edges(_truthy_entity_edges(center_raw.get("claims", {})), center_qid)
    n_total = len(edges_raw)

    def neighbor_rank(nid: str) -> float:
        pr = pagerank.get(nid)
        if pr is not None:
            return pr
        ent = neighbor_meta.get(nid)
        return _sitelink_count(ent) if ent else 0.0

    keep_set = set(always_keep)
    # Deterministic order: popularity desc; always-keep edges float to the front.
    edges_raw.sort(key=lambda e: (e[1] in keep_set, neighbor_rank(e[1])), reverse=True)

    capped = cap is not None and n_total > cap
    kept = edges_raw[:cap] if capped else edges_raw
    if capped:  # guarantee always-keep edges survive the cap
        present = {n for _, n in kept}
        for p, n in edges_raw[cap:]:
            if n in keep_set and n not in present:
                kept[-1] = (p, n)
                present.add(n)

    edges = [
        Edge(
            property_id=pid,
            property_label=_localized(property_meta.get(pid, {}), "labels", langs),
            neighbor=Entity(
                qid=nid,
                label=_localized(neighbor_meta.get(nid, {}), "labels", langs),
                description=_localized(neighbor_meta.get(nid, {}), "descriptions", langs),
                rank=neighbor_rank(nid),
            ),
        )
        for pid, nid in kept
    ]
    center_pr = pagerank.get(center_qid)
    center = Entity(
        qid=center_qid,
        label=_localized(center_raw, "labels", langs),
        description=_localized(center_raw, "descriptions", langs),
        rank=center_pr if center_pr is not None else _sitelink_count(center_raw),
    )
    return Subgraph(center=center, edges=edges, n_edges_total=n_total, capped=capped)


# --------------------------------------------------------------------------------------
# Synchronous client (single entity).
# --------------------------------------------------------------------------------------
class WikidataClient:
    """Fetches entity data and builds rank-sorted, capped 1-hop subgraphs (sync)."""

    def __init__(
        self, cfg: Settings | None = None, *, pagerank: Mapping[str, float] | None = None
    ) -> None:
        self.cfg = cfg or settings
        self._pagerank = pagerank or {}
        self._cache = KVCache(self.cfg.cache_path)
        self._client = httpx.Client(
            timeout=self.cfg.request_timeout_s,
            headers={"User-Agent": self.cfg.user_agent},
            follow_redirects=True,
        )
        self._retryer = Retrying(
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=self.cfg.retry_max_wait_s),
            reraise=True,
        )

    def _params(self, ids: Sequence[str], props: str) -> dict[str, str | int]:
        return {
            "action": "wbgetentities",
            "ids": "|".join(ids),
            "props": props,
            "languages": self.cfg.api_languages,
            "format": "json",
            "maxlag": self.cfg.maxlag,
        }

    def _api_get_entities(self, ids: list[str], props: str) -> dict[str, dict]:
        return self._retryer(self._api_get_entities_once, ids, props)

    def _api_get_entities_once(self, ids: list[str], props: str) -> dict[str, dict]:
        resp = self._client.get(self.cfg.wikidata_api_url, params=self._params(ids, props))
        resp.raise_for_status()
        return _entities_from_response(resp.json())

    def _get_entities_raw(self, ids: Sequence[str], *, full: bool) -> dict[str, dict]:
        """Cached batched fetch. Returns {qid: raw_entity_json}."""
        tier = "full" if full else "lite"
        props = _FULL_PROPS if full else _LITE_PROPS
        out, missing = _split_cached(self._cache, ids, tier)
        for i in range(0, len(missing), self.cfg.api_batch_size):
            batch = missing[i : i + self.cfg.api_batch_size]
            resolved = _resolved(self._api_get_entities(batch, props))
            self._cache.put_many({f"{tier}:{q}": e for q, e in resolved.items()})
            out.update(resolved)
        return out

    def fetch_neighborhood(
        self,
        qid: str,
        *,
        max_neighbors: int | None = None,
        always_keep: Iterable[str] = (),
    ) -> Subgraph | None:
        """Build a rank-sorted, capped 1-hop truthy subgraph for ``qid``."""
        cap = self.cfg.max_neighbors if max_neighbors is None else max_neighbors
        center_raw = self._get_entities_raw([qid], full=True).get(qid)
        if center_raw is None:
            return None
        edges = _dedup_edges(_truthy_entity_edges(center_raw.get("claims", {})), qid)
        neighbor_meta = self._get_entities_raw(sorted({n for _, n in edges}), full=False)
        property_meta = self._get_entities_raw(sorted({p for p, _ in edges}), full=False)
        return _build_subgraph(
            qid, center_raw, neighbor_meta, property_meta,
            cap=cap, always_keep=always_keep, langs=self.cfg.label_languages,
            pagerank=self._pagerank,
        )

    def close(self) -> None:
        self._client.close()
        self._cache.close()

    def __enter__(self) -> WikidataClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------------------
# Asynchronous client (batched + concurrent, for snapshotting many entities).
# --------------------------------------------------------------------------------------
class AsyncWikidataClient:
    """Concurrent extractor: dedups neighbor/property fetches across all subjects."""

    def __init__(
        self, cfg: Settings | None = None, *, pagerank: Mapping[str, float] | None = None
    ) -> None:
        self.cfg = cfg or settings
        self._pagerank = pagerank or {}
        # Count of API batches that exhausted retries and were skipped — a non-zero value
        # means the snapshot is incomplete (some entities/labels missing); re-run to fill.
        self.n_failed_batches = 0
        self._cache = KVCache(self.cfg.cache_path)
        self._client = httpx.AsyncClient(
            timeout=self.cfg.request_timeout_s,
            headers={"User-Agent": self.cfg.user_agent},
            follow_redirects=True,
        )
        self._sem = asyncio.Semaphore(self.cfg.concurrency)
        self._aretryer = AsyncRetrying(
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=self.cfg.retry_max_wait_s),
            reraise=True,
        )

    def _params(self, ids: Sequence[str], props: str) -> dict[str, str | int]:
        return {
            "action": "wbgetentities",
            "ids": "|".join(ids),
            "props": props,
            "languages": self.cfg.api_languages,
            "format": "json",
            "maxlag": self.cfg.maxlag,
        }

    async def _api_get_entities_once(self, ids: list[str], props: str) -> dict[str, dict]:
        async with self._sem:
            resp = await self._client.get(
                self.cfg.wikidata_api_url, params=self._params(ids, props)
            )
        resp.raise_for_status()
        return _entities_from_response(resp.json())

    async def _aget_entities_raw(self, ids: Sequence[str], *, full: bool) -> dict[str, dict]:
        tier = "full" if full else "lite"
        props = _FULL_PROPS if full else _LITE_PROPS
        out, missing = _split_cached(self._cache, ids, tier)
        batches = [
            missing[i : i + self.cfg.api_batch_size]
            for i in range(0, len(missing), self.cfg.api_batch_size)
        ]

        async def run(batch: list[str]) -> dict[str, dict]:
            return _resolved(await self._aretryer(self._api_get_entities_once, batch, props))

        # Resilient: a batch that exhausts retries (e.g. a sustained maxlag spike) is skipped
        # rather than aborting the whole run. Its entities stay missing/uncached and are
        # picked up on a re-run (the cache makes snapshots resumable).
        results = await asyncio.gather(*(run(b) for b in batches), return_exceptions=True)
        n_failed = 0
        for resolved in results:
            if isinstance(resolved, BaseException):
                n_failed += 1
                continue
            self._cache.put_many({f"{tier}:{q}": e for q, e in resolved.items()})
            out.update(resolved)
        if n_failed:
            self.n_failed_batches += n_failed
            logger.warning(
                "%d/%d API batches failed (transient); re-run to fill gaps", n_failed, len(batches)
            )
        return out

    async def fetch_many(
        self,
        qids: Sequence[str],
        *,
        always_keep: dict[str, str] | None = None,
        max_neighbors: int | None = None,
    ) -> dict[str, Subgraph]:
        """Snapshot many subjects, fetching each tier once across the whole batch."""
        cap = self.cfg.max_neighbors if max_neighbors is None else max_neighbors
        keep = always_keep or {}

        centers = await self._aget_entities_raw(qids, full=True)
        edges_by_qid: dict[str, list[tuple[str, str]]] = {}
        neighbor_ids: set[str] = set()
        property_ids: set[str] = set()
        for qid, center_raw in centers.items():
            edges = _dedup_edges(_truthy_entity_edges(center_raw.get("claims", {})), qid)
            edges_by_qid[qid] = edges
            neighbor_ids.update(n for _, n in edges)
            property_ids.update(p for p, _ in edges)

        neighbor_meta = await self._aget_entities_raw(sorted(neighbor_ids), full=False)
        property_meta = await self._aget_entities_raw(sorted(property_ids), full=False)

        return {
            qid: _build_subgraph(
                qid, centers[qid], neighbor_meta, property_meta,
                cap=cap, always_keep=[keep[qid]] if qid in keep else [],
                langs=self.cfg.label_languages, pagerank=self._pagerank,
            )
            for qid in edges_by_qid
        }

    async def aclose(self) -> None:
        await self._client.aclose()
        self._cache.close()

    async def __aenter__(self) -> AsyncWikidataClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def _split_cached(
    cache: KVCache, ids: Sequence[str], tier: str
) -> tuple[dict[str, dict], list[str]]:
    """Partition ``ids`` into {cached qid: json} and a list of missing qids."""
    out: dict[str, dict] = {}
    missing: list[str] = []
    for qid in ids:
        cached = cache.get(f"{tier}:{qid}")
        if cached is not None:
            out[qid] = cached
        else:
            missing.append(qid)
    return out, missing


def _resolved(fetched: dict[str, dict]) -> dict[str, dict]:
    """Keep only entities that actually resolved (skip API 'missing' markers)."""
    return {q: e for q, e in fetched.items() if "missing" not in e}
