"""Runtime configuration (pydantic-settings).

All settings are overridable via ``CF_``-prefixed environment variables, e.g.
``CF_DATA_ROOT=/data/conceptformer``. Keep large artifacts OFF ``/home`` (96% full
on the dev box) — point ``CF_DATA_ROOT`` at the 3.2 TB ``/`` partition for the full run.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_V2 = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CF_", env_file=".env", extra="ignore")

    # --- storage -----------------------------------------------------------
    # Defaults to v2/data (on /home) for the lite pilot; override for scale.
    data_root: Path = Field(default=_REPO_V2 / "data")

    # --- wikidata source (endpoint-agnostic) -------------------------------
    # Phase 1: the public wbgetentities REST API. To swap to a local qEndpoint
    # later, this is the only thing that changes (see WikidataClient).
    wikidata_api_url: str = "https://www.wikidata.org/w/api.php"
    wikidata_sparql_url: str = "https://query.wikidata.org/sparql"

    # danker Wikipedia-PageRank file = the CF-Train entity pool (~6-7M sitelinked entities,
    # ranked by popularity). Override the date to refresh.
    pagerank_url: str = "https://danker.s3.amazonaws.com/2023-12-04.allwiki.links.rank.bz2"
    # Polite, descriptive UA is required by the Wikimedia API policy.
    user_agent: str = (
        "ConceptFormer/2.0 (https://github.com/joelbarmettler; joel.barmettler@gmail.com)"
    )
    language: str = "en"
    # Wikidata's 2024 "mul" (multilingual) migration moved labels that are identical
    # across languages out of "en" into "mul" (e.g. Q42 "Douglas Adams"). Request and
    # fall back across these, in order, when resolving labels.
    label_languages: list[str] = ["en", "mul"]
    request_timeout_s: float = 30.0
    # Patient retries: Wikidata maxlag spikes (server replication lag) can persist for
    # minutes; with these, backoff waits ~1+2+4+...+60 over the attempts before giving up.
    max_retries: int = 10
    retry_max_wait_s: float = 60.0
    # Tolerate moderate Wikidata replication lag (it routinely sits ~5-12s). Higher = fewer
    # rejections during lag; still polite for a one-time bulk snapshot.
    maxlag: int = 20
    api_batch_size: int = 50  # wbgetentities hard limit for non-bot users
    concurrency: int = 8  # max in-flight API requests for the async extractor

    # RAG/teacher text baseline is bounded by the context window (not an arbitrary neighbor
    # count): we fill up to this many tokens with the top-ranked verbalized neighbors. The
    # snapshot + ConceptFormer encoder stay uncapped.
    rag_context_tokens: int = 2048

    # --- neighborhood extraction ------------------------------------------
    # No cap by default: snapshots store the COMPLETE 1-hop neighborhood (the v2
    # encoder handles variable neighbor counts via masking; capping is a downstream,
    # teacher-side concern). Set an int only to deliberately truncate.
    max_neighbors: int | None = None

    @property
    def api_languages(self) -> str:
        """The 'languages' param for wbgetentities (labels + descriptions)."""
        ordered = dict.fromkeys([*self.label_languages, self.language])
        return "|".join(ordered)

    @property
    def cache_path(self) -> Path:
        return self.data_root / "wikidata_cache.sqlite"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_root / "snapshots"

    @property
    def results_dir(self) -> Path:
        return self.data_root / "results"

    @property
    def generation_cache_path(self) -> Path:
        return self.data_root / "generations.sqlite"

    @property
    def pagerank_path(self) -> Path:
        return self.data_root / "danker" / self.pagerank_url.rsplit("/", 1)[-1]

    def ensure_dirs(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
