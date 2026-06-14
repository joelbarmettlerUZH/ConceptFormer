"""Tiny persistent sqlite key→json cache, shared across the data / eval / model layers.

Used for: Wikidata entity fetches, subclass expansions, and LLM generations — so repeated
runs never redo network calls or (deterministic, greedy) model forward passes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class KVCache:
    """A single-process sqlite key→json store. Not safe to share across threads."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(path))
        self._con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._con.commit()

    def get(self, key: str) -> dict | None:
        row = self._con.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_many(self, items: dict[str, dict]) -> None:
        self._con.executemany(
            "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
            [(k, json.dumps(v, separators=(",", ":"))) for k, v in items.items()],
        )
        self._con.commit()

    def put(self, key: str, value: dict) -> None:
        self.put_many({key: value})

    def close(self) -> None:
        self._con.close()
