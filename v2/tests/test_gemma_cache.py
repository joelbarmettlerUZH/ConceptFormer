"""Per-entity resume cache in ``GemmaClient`` — the behaviour that makes a long full-corpus
generation crash-resumable. Stubs the *server call* (no vLLM / network) so the real
``_generate_one`` cache logic is exercised, not a copy.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from conceptformer.cache import KVCache
from conceptformer.generate.gemma import GemmaClient
from conceptformer.generate.schema import GeneratedQuestion, GenerationResult
from conceptformer.schemas import Edge, Entity, Subgraph


def _sg(qid: str = "Q42") -> Subgraph:
    return Subgraph(
        center=Entity(qid=qid, label="Douglas Adams"),
        edges=[Edge(property_id="P106", property_label="occupation",
                    neighbor=Entity(qid="Q1", label="writer"))],
        n_edges_total=1,
    )


def _result() -> GenerationResult:
    return GenerationResult(
        questions=[GeneratedQuestion(question="What is Douglas Adams's occupation?",
                                     answer="writer", task_type="single")]
    )


def _stub_server(client: GemmaClient, parsed: GenerationResult | None, counter: list[int]) -> None:
    """Replace only the OpenAI client with a structural stub exposing chat.completions.parse."""

    async def parse(**_kwargs: object) -> object:
        counter[0] += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])

    # Monkeypatch only the network call; the real cache read/write path stays under test.
    stub = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    setattr(client, "_client", stub)  # noqa: B010


def test_cache_miss_then_hit_skips_second_call(tmp_path: Path) -> None:
    cache = KVCache(tmp_path / "gen.sqlite")

    calls = [0]
    client = GemmaClient(cache=cache)
    _stub_server(client, _result(), calls)
    first = asyncio.run(client.generate([_sg()]))
    assert calls[0] == 1 and first[0] is not None and first[0].questions[0].answer == "writer"

    # Fresh client + server, SAME cache file: the entity is served from disk, server NOT called.
    calls2 = [0]
    client2 = GemmaClient(cache=cache)
    _stub_server(client2, _result(), calls2)
    second = asyncio.run(client2.generate([_sg()]))
    assert calls2[0] == 0  # resumed from cache
    assert second[0] is not None and second[0].questions[0].answer == "writer"


def test_failures_are_not_cached(tmp_path: Path) -> None:
    cache = KVCache(tmp_path / "gen.sqlite")
    client = GemmaClient(cache=cache)
    _stub_server(client, None, [0])  # server returns no parsed result → None

    assert asyncio.run(client.generate([_sg()]))[0] is None
    assert cache.get(client._cache_key(_sg())) is None  # not persisted → retried next run


def test_cache_key_distinguishes_entities_and_config(tmp_path: Path) -> None:
    cache = KVCache(tmp_path / "gen.sqlite")
    c8 = GemmaClient(cache=cache, n_questions=8)
    c4 = GemmaClient(cache=cache, n_questions=4)
    assert c8._cache_key(_sg("Q42")) != c8._cache_key(_sg("Q99"))  # entity → distinct key
    assert c8._cache_key(_sg("Q42")) != c4._cache_key(_sg("Q42"))  # config → distinct key
