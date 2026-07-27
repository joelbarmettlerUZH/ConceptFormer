"""Question translator: a concurrent OpenAI-SDK client against the same vLLM server as generation.

Translates an English QA question into a target language while keeping the entity mention in its
English surface form, so the mention span stays locatable for the mention-anchored splice (the
localized-mention variant is produced afterwards by substituting the Wikidata label). Structured
output (``TranslationResult``) is enforced by vLLM guided decoding, same as the generator.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from openai import AsyncOpenAI, OpenAIError

from conceptformer.cache import KVCache
from conceptformer.data.multilingual import TranslationResult
from conceptformer.generate.gemma import DEFAULT_MODEL

LANG_NAMES = {"de": "German", "fr": "French", "zh": "Chinese", "ja": "Japanese"}

_SYSTEM = (
    "You are a professional translator. Translate the user's question into {lang}. "
    "Keep the named entity written EXACTLY as it appears in the English source (do not "
    "translate or transliterate the entity name); translate everything else naturally. "
    "Return the translated question and the entity's natural {lang} surface form separately."
)


class GemmaTranslator:
    """Concurrent, typed translation client for a running vLLM OpenAI server."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        model: str = DEFAULT_MODEL,
        lang: str = "de",
        concurrency: int = 64,
        temperature: float = 0.2,
        max_tokens: int = 512,
        cache: KVCache | None = None,
    ) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=600.0, max_retries=2)
        self._sem = asyncio.Semaphore(concurrency)
        self.model = model
        self.lang = lang
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._cache = cache
        self._system = _SYSTEM.format(lang=LANG_NAMES.get(lang, lang))

    def _cache_key(self, question: str) -> str:
        return f"translate:{self.model}:{self.lang}:{question}"

    async def _one(self, question: str) -> TranslationResult | None:
        if self._cache is not None:
            hit = self._cache.get(self._cache_key(question))
            if hit is not None:
                return TranslationResult.model_validate(hit)
        try:
            async with self._sem:
                completion = await self._client.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "system", "content": self._system},
                              {"role": "user", "content": question}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=TranslationResult,
                )
            result = completion.choices[0].message.parsed
        except (OpenAIError, ValueError):
            return None
        if result is not None and self._cache is not None:
            self._cache.put(self._cache_key(question), result.model_dump())
        return result

    async def translate(self, questions: Sequence[str]) -> list[TranslationResult | None]:
        """Fire all translation requests concurrently; vLLM batches them server-side."""
        return await asyncio.gather(*(self._one(q) for q in questions))
