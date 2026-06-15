"""Build versioned, content-hashed neighborhood snapshots.

A snapshot = a JSONL of ``Subgraph`` rows + a ``manifest.json`` (version, source, count,
sha256, config). This is the immutable, reproducible graph artifact the rest of the
pipeline consumes — no live API at train/eval time once it exists.

Two builders:
- ``build_snapshot``          — synchronous, one entity at a time (small / simple).
- ``build_snapshot_parallel`` — async + concurrent, fetches each tier once per chunk;
  use this for tens of thousands of subjects.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from conceptformer.config import Settings, settings
from conceptformer.data.quality import is_wikimedia_internal
from conceptformer.data.wikidata import AsyncWikidataClient, WikidataClient
from conceptformer.schemas import Subgraph


def _write_manifest(
    out_dir: Path, *, name: str, n_requested: int, n_ok: int, n_missing: int,
    sha256: str, cfg: Settings, n_failed_batches: int = 0,
) -> dict:
    manifest = {
        "name": name,
        "version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "n_requested": n_requested,
        "n_subgraphs": n_ok,
        "n_missing": n_missing,
        # >0 ⇒ snapshot is INCOMPLETE (skipped API batches → missing entities/labels); re-run.
        "n_failed_batches": n_failed_batches,
        "complete": n_failed_batches == 0,
        "max_neighbors": cfg.max_neighbors,
        "wikidata_api_url": cfg.wikidata_api_url,
        "language": cfg.language,
        "sha256": sha256,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_snapshot(
    name: str,
    subject_qids: Sequence[str],
    *,
    cfg: Settings | None = None,
) -> Path:
    """Synchronous snapshot of ``subject_qids`` (one API round-trip per entity)."""
    cfg = cfg or settings
    cfg.ensure_dirs()
    out_dir = cfg.snapshots_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_missing = 0
    hasher = hashlib.sha256()
    with WikidataClient(cfg) as client, (out_dir / "subgraphs.jsonl").open("w") as fh:
        for qid in tqdm(subject_qids, desc=f"snapshot:{name}", unit="entity"):
            sg = client.fetch_neighborhood(qid)
            if sg is None:
                n_missing += 1
                continue
            line = sg.model_dump_json() + "\n"
            fh.write(line)
            hasher.update(line.encode("utf-8"))
            n_ok += 1

    _write_manifest(
        out_dir, name=name, n_requested=len(subject_qids), n_ok=n_ok,
        n_missing=n_missing, sha256=hasher.hexdigest(), cfg=cfg,
    )
    return out_dir


def build_snapshot_parallel(
    name: str,
    subject_qids: Sequence[str],
    *,
    cfg: Settings | None = None,
    chunk_size: int = 200,
    pagerank: Mapping[str, float] | None = None,
) -> Path:
    """Concurrent snapshot: fetches each tier once per chunk via ``AsyncWikidataClient``."""
    cfg = cfg or settings
    return asyncio.run(_build_parallel(name, subject_qids, cfg, chunk_size, pagerank))


async def _build_parallel(
    name: str,
    subject_qids: Sequence[str],
    cfg: Settings,
    chunk_size: int,
    pagerank: Mapping[str, float] | None,
) -> Path:
    cfg.ensure_dirs()
    out_dir = cfg.snapshots_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_missing = 0
    hasher = hashlib.sha256()
    async with AsyncWikidataClient(cfg, pagerank=pagerank) as client:
        with (out_dir / "subgraphs.jsonl").open("w") as fh:
            bar = tqdm(total=len(subject_qids), desc=f"snapshot:{name}", unit="entity")
            for i in range(0, len(subject_qids), chunk_size):
                chunk = list(subject_qids[i : i + chunk_size])
                subgraphs = await client.fetch_many(chunk)
                for qid in chunk:  # preserve input order; skip unresolved
                    sg = subgraphs.get(qid)
                    if sg is None:
                        n_missing += 1
                        continue
                    line = sg.model_dump_json() + "\n"
                    fh.write(line)
                    hasher.update(line.encode("utf-8"))
                    n_ok += 1
                bar.update(len(chunk))
            bar.close()

    _write_manifest(
        out_dir, name=name, n_requested=len(subject_qids), n_ok=n_ok,
        n_missing=n_missing, sha256=hasher.hexdigest(), cfg=cfg,
        n_failed_batches=client.n_failed_batches,
    )
    return out_dir


def build_cftrain_snapshot(
    name: str,
    candidate_qids: Sequence[str],
    *,
    target: int,
    min_edges: int = 6,
    cfg: Settings | None = None,
    pagerank: Mapping[str, float] | None = None,
    chunk_size: int = 200,
) -> Path:
    """Snapshot CF-Train candidates, keeping only USABLE entities until ``target`` is reached.

    Candidates are oversampled (the deep tail is ~38% usable); each is dropped if it's a
    Wikimedia-internal page or has fewer than ``min_edges`` facts. Stops early once ``target``
    usable subjects are written. Pass the danker ``pagerank`` map for consistent neighbor order.
    """
    cfg = cfg or settings
    return asyncio.run(
        _build_cftrain(name, candidate_qids, target, min_edges, cfg, pagerank, chunk_size)
    )


async def _build_cftrain(
    name: str,
    candidate_qids: Sequence[str],
    target: int,
    min_edges: int,
    cfg: Settings,
    pagerank: Mapping[str, float] | None,
    chunk_size: int,
) -> Path:
    cfg.ensure_dirs()
    out_dir = cfg.snapshots_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_usable = n_junk = n_thin = n_missing = n_unlabeled = n_seen = 0
    hasher = hashlib.sha256()
    async with AsyncWikidataClient(cfg, pagerank=pagerank) as client:
        with (out_dir / "subgraphs.jsonl").open("w") as fh:
            bar = tqdm(total=target, desc=f"cftrain:{name}", unit="usable")
            for i in range(0, len(candidate_qids), chunk_size):
                if n_usable >= target:
                    break
                chunk = list(candidate_qids[i : i + chunk_size])
                subgraphs = await client.fetch_many(chunk)
                for qid in chunk:
                    n_seen += 1
                    sg = subgraphs.get(qid)
                    if sg is None:
                        n_missing += 1
                    elif is_wikimedia_internal(sg):
                        n_junk += 1
                    elif not sg.center.label:
                        n_unlabeled += 1
                    elif len(sg.edges) < min_edges:
                        n_thin += 1
                    else:
                        line = sg.model_dump_json() + "\n"
                        fh.write(line)
                        hasher.update(line.encode("utf-8"))
                        n_usable += 1
                        bar.update(1)
                        if n_usable >= target:
                            break
            bar.close()

    manifest = {
        "name": name,
        "kind": "cftrain",
        "version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "target": target,
        "n_usable": n_usable,
        "n_candidates_seen": n_seen,
        "n_wikimedia_junk": n_junk,
        "n_unlabeled": n_unlabeled,
        "n_thin": n_thin,
        "n_missing": n_missing,
        "min_edges": min_edges,
        "n_failed_batches": client.n_failed_batches,
        "complete": n_usable >= target and client.n_failed_batches == 0,
        "sha256": hasher.hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def iter_subgraphs(snapshot_dir: Path) -> Iterator[Subgraph]:
    """Stream ``Subgraph`` rows back from a snapshot directory."""
    with (snapshot_dir / "subgraphs.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield Subgraph.model_validate_json(line)
