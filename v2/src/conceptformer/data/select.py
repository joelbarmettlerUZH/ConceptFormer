"""CF-Train entity selection from the danker Wikipedia-PageRank file.

The danker file ranks the ~6-7M *sitelinked* Wikidata entities by PageRank (descending) — i.e.
exactly the entities people ask about, with popularity built in. We stratify by popularity
(by rank position) and sample **long-tail-heavy**, excluding eval subjects, to produce the
CF-Train entity pool. Long-tail entities are both the highest knowledge-injection signal (the
base model doesn't know them) and the smallest neighborhoods, so this lever does double duty.
"""

from __future__ import annotations

import bz2
import random
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from conceptformer.config import Settings, settings

# Pool is split by rank position: top `high` fraction = most popular, next `mid`, rest = tail.
POOL_HIGH_FRAC = 0.05
POOL_MID_FRAC = 0.15
# Sample composition (low/tail, mid, high) — deliberately tail-heavy, with a thin popular slice.
SAMPLE_FRACS = (0.70, 0.20, 0.10)


class SelectedEntity(BaseModel):
    qid: str
    rank: float
    tier: str  # "high" | "mid" | "low"


def download_pagerank(cfg: Settings | None = None) -> Path:
    """Download the danker PageRank bz2 (cached; ~212 MB)."""
    cfg = cfg or settings
    dest = cfg.pagerank_path
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(cfg.pagerank_url, tmp)
    tmp.rename(dest)
    return dest


def load_pagerank(path: Path) -> list[tuple[str, float]]:
    """Parse the bz2 into [(qid, rank)], preserving the file's rank-descending order."""
    entries: list[tuple[str, float]] = []
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="load pagerank", unit=" lines"):
            qid, _, rank = line.partition("\t")
            if qid and rank:
                entries.append((qid, float(rank)))
    return entries


def load_pagerank_map(path: Path) -> dict[str, float]:
    """Load the danker file as a {qid: rank} map (for PageRank-based neighbor ranking)."""
    return dict(load_pagerank(path))


def _sample_tier(
    entries: Sequence[tuple[str, float]],
    start: int,
    end: int,
    k: int,
    tier: str,
    exclude: set[str],
    rng: random.Random,
) -> list[SelectedEntity]:
    if k <= 0 or end <= start:
        return []
    pool = end - start
    # Oversample by the exclude budget so we still hit k after dropping excluded entities.
    take = min(pool, k + len(exclude) + 100)
    chosen: list[SelectedEntity] = []
    for i in rng.sample(range(start, end), take):
        qid, rank = entries[i]
        if qid in exclude:
            continue
        chosen.append(SelectedEntity(qid=qid, rank=rank, tier=tier))
        if len(chosen) >= k:
            break
    return chosen


# Balanced split: cover the full popularity spectrum (tail/mid/popular) ~evenly. Use this when the
# eval set is popularity-broad (PopQA/EQ) to reduce the tail-train vs popular-eval distribution gap.
BALANCED_FRACS = (0.34, 0.33, 0.33)


def select_entities(
    entries: Sequence[tuple[str, float]],
    *,
    n: int,
    exclude: Iterable[str] = (),
    seed: int = 0,
    sample_fracs: tuple[float, float, float] = SAMPLE_FRACS,
) -> list[SelectedEntity]:
    """Stratified sample of ``n`` entities (minus ``exclude``), per ``(low, mid, high)`` fracs."""
    exclude_set = set(exclude)
    total = len(entries)
    high_end = int(total * POOL_HIGH_FRAC)
    mid_end = int(total * (POOL_HIGH_FRAC + POOL_MID_FRAC))
    low_frac, mid_frac, high_frac = sample_fracs
    rng = random.Random(seed)

    out: list[SelectedEntity] = []
    out += _sample_tier(entries, mid_end, total, round(n * low_frac), "low", exclude_set, rng)
    out += _sample_tier(entries, high_end, mid_end, round(n * mid_frac), "mid", exclude_set, rng)
    out += _sample_tier(entries, 0, high_end, round(n * high_frac), "high", exclude_set, rng)
    rng.shuffle(out)
    return out


def save_entities(entities: Sequence[SelectedEntity], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for e in entities:
            fh.write(e.model_dump_json() + "\n")
    return path


def load_entities(path: Path) -> list[SelectedEntity]:
    with path.open(encoding="utf-8") as fh:
        return [SelectedEntity.model_validate_json(line) for line in fh if line.strip()]


def manifest(entities: Sequence[SelectedEntity], *, n_requested: int, source: str) -> dict:
    counts: dict[str, int] = {"high": 0, "mid": 0, "low": 0}
    for e in entities:
        counts[e.tier] += 1
    return {
        "n_requested": n_requested,
        "n_selected": len(entities),
        "by_tier": counts,
        "source": source,
    }
