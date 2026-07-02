"""Gemma question generator: a concurrent OpenAI-SDK client against a vLLM server.

Gemma runs as a vLLM OpenAI-compatible server (``uv run --with vllm vllm serve ...``); we fire
many requests concurrently from here and vLLM's continuous batching packs them into efficient
GPU batches. We use the OpenAI SDK's structured-output ``.parse(response_format=...)`` so every
response is decoded and validated straight into a typed ``GenerationResult`` (vLLM enforces the
schema via guided decoding).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from openai import AsyncOpenAI, OpenAIError

from conceptformer.cache import KVCache
from conceptformer.generate.prompt import GEN_SYSTEM, build_generation_user
from conceptformer.generate.schema import GenerationResult
from conceptformer.schemas import Subgraph

DEFAULT_MODEL = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"


def _estimate_tokens(text: str) -> int:
    """Rough client-side token estimate (no local tokenizer) for the facts budget."""
    return len(text) // 4


class GemmaClient:
    """Concurrent, typed client for a running vLLM OpenAI server."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        model: str = DEFAULT_MODEL,
        concurrency: int = 64,
        n_questions: int = 8,
        facts_budget: int = 1500,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        cache: KVCache | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url, api_key="EMPTY", timeout=600.0, max_retries=2
        )
        self._sem = asyncio.Semaphore(concurrency)
        self.model = model
        self.n_questions = n_questions
        self.facts_budget = facts_budget
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Per-entity persistence so a long full-corpus run RESUMES after a crash instead of redoing
        # all completed entities (the 100k run is ~10h; a single save-at-end would lose everything).
        self._cache = cache

    def _cache_key(self, sg: Subgraph) -> str:
        # Namespaced (won't collide with the shared eval cache) and keyed by the inputs that change
        # the output, so a config change misses rather than returning stale QA. Temperature is left
        # out on purpose: resume only needs "was this entity already generated", not bit-identity.
        return f"gemma_qa:{self.model}:{self.n_questions}:{self.facts_budget}:{sg.center.qid}"

    async def _generate_one(self, sg: Subgraph) -> GenerationResult | None:
        """Generate for one subgraph; returns None on any API/parse/truncation failure.

        On a cache hit, returns the stored result without calling the server. Only *successes* are
        cached, so a failed entity (``None``) is retried on the next run rather than stuck.
        """
        if self._cache is not None and (hit := self._cache.get(self._cache_key(sg))) is not None:
            return GenerationResult.model_validate(hit)
        user = build_generation_user(sg, self.n_questions, _estimate_tokens, self.facts_budget)
        try:
            async with self._sem:
                completion = await self._client.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "user", "content": f"{GEN_SYSTEM}\n\n{user}"}],
                    temperature=self.temperature,
                    top_p=0.95,
                    max_tokens=self.max_tokens,
                    response_format=GenerationResult,
                )
            result = completion.choices[0].message.parsed
        except (OpenAIError, ValueError):
            # truncated JSON (LengthFinishReasonError), connection error, validation error, …
            return None
        if result is not None and self._cache is not None:
            self._cache.put(self._cache_key(sg), result.model_dump())
        return result

    async def generate(self, subgraphs: Sequence[Subgraph]) -> list[GenerationResult | None]:
        """Fire all requests concurrently; vLLM batches them server-side."""
        return await asyncio.gather(*(self._generate_one(sg) for sg in subgraphs))
