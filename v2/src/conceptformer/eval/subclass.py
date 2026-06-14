"""Subclass expansion for the ``fair`` metric.

For a gold answer entity (a *class* like "politician"), fetch the labels of its Wikidata
subclasses up to a bounded depth via ``P279``. Matching the prediction against these (with
word boundaries) credits a *more specific* correct answer — e.g. "Minister"/"Mayor" for gold
"politician" — deterministically and condition-agnostically (no need to link the free-text
answer). Results are cached per gold entity.

Limits (by design): only catches the *is-a-kind-of* relation — not located-in granularity
(Toronto vs Canada) or relation-confusion. Applied only to type-like PopQA relations.
"""

from __future__ import annotations

import re

import httpx
from tenacity import Retrying, stop_after_attempt, wait_exponential

from conceptformer.cache import KVCache
from conceptformer.config import Settings, settings

# PopQA relations whose answers are classes (subclass expansion is meaningful).
SUBCLASS_RELATIONS = frozenset({"occupation", "genre", "religion", "sport", "color"})

_BARE_QID = re.compile(r"^Q\d+$")  # wikibase:label returns the QID when no label exists


def parse_subclass_bindings(payload: dict) -> list[str]:
    """Extract labels from a SPARQL JSON result, dropping unresolved bare-QID 'labels'."""
    bindings = payload.get("results", {}).get("bindings", [])
    labels = [b["xLabel"]["value"] for b in bindings if "xLabel" in b]
    return [v for v in labels if v and not _BARE_QID.match(v)]


class SubclassExpander:
    """Fetches (cached) subclass-descendant labels of a gold entity via SPARQL."""

    def __init__(self, cfg: Settings | None = None, *, depth: int = 2, limit: int = 1000) -> None:
        self.cfg = cfg or settings
        self._cache = KVCache(self.cfg.cache_path)
        self._client = httpx.Client(
            timeout=self.cfg.request_timeout_s,
            headers={
                "User-Agent": self.cfg.user_agent,
                "Accept": "application/sparql-results+json",
            },
        )
        self._retryer = Retrying(
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=self.cfg.retry_max_wait_s),
            reraise=True,
        )
        self._limit = limit
        # bounded P279 property path (depth 2 catches e.g. minister⊂...⊂politician)
        self._path = "wdt:P279|wdt:P279/wdt:P279" if depth == 2 else "wdt:P279*"

    def labels(self, qid: str) -> list[str]:
        key = f"subclass:{qid}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached["labels"]
        labels = self._retryer(self._query, qid)
        self._cache.put_many({key: {"labels": labels}})
        return labels

    def _query(self, qid: str) -> list[str]:
        langs = ",".join(self.cfg.label_languages)
        query = (
            f"SELECT DISTINCT ?xLabel WHERE {{ ?x ({self._path}) wd:{qid}. "
            f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{langs}". }} }} '
            f"LIMIT {self._limit}"
        )
        params = {"query": query, "format": "json"}
        resp = self._client.get(self.cfg.wikidata_sparql_url, params=params)
        resp.raise_for_status()
        return parse_subclass_bindings(resp.json())

    def expand(self, qids: set[str]) -> dict[str, list[str]]:
        """Return {qid: subclass_labels} for each qid (cached)."""
        return {qid: self.labels(qid) for qid in qids}

    def close(self) -> None:
        self._client.close()
        self._cache.close()

    def __enter__(self) -> SubclassExpander:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
