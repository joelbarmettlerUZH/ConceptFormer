"""Minimal HF chat-model wrapper (greedy generation from chat messages).

Thin on purpose — the seed of the v2 backbone adapter. Imports torch/transformers lazily so
the data layer doesn't require the heavy ``infer`` deps.

Greedy decoding is deterministic, so generations are cached (keyed by model + prompt +
max_new_tokens): repeated/identical eval runs skip the forward pass, and a run that crashes
midway (e.g. CUDA OOM) resumes from what it already produced.

The HuggingFace auto-classes are decorator-wrapped and effectively untyped, so we treat the
``from_pretrained`` boundary as ``Any`` rather than fighting partial library stubs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from tqdm import tqdm

from conceptformer.cache import KVCache


def trim_generated_path(ids: list[int], eos: int | None, pad: int | None) -> list[int]:
    """Trim a batched-generation row to the real continuation: up to & including the first eos,
    else strip trailing pad tokens (finished sequences are pad-filled to the batch length)."""
    if eos is not None and eos in ids:
        return ids[: ids.index(eos) + 1]
    if pad is not None:
        while ids and ids[-1] == pad:
            ids = ids[:-1]
    return ids


def generation_cache_key(
    model_id: str, system: str, user: str, max_new_tokens: int, decoding: str = "greedy"
) -> str:
    """Stable cache key for a generation.

    ``decoding`` discriminates the sampling regime (e.g. "greedy" vs "sample-t0.8-p0.95-s0");
    without it, switching from greedy to sampling would silently return stale cached outputs.
    """
    raw = "\x00".join([model_id, decoding, str(max_new_tokens), system, user])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ChatModel:
    """Loads a HF causal chat LM and generates short greedy completions."""

    tokenizer: Any
    model: Any

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        enable_thinking: bool = False,
        cache: KVCache | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        auto_tokenizer: Any = AutoTokenizer
        auto_model: Any = AutoModelForCausalLM
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self._cache = cache
        self.tokenizer = auto_tokenizer.from_pretrained(model_id)
        self.model = auto_model.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device)
        self.model.eval()
        self._device = device
        # Left-padding is required for correct batched decoder-only generation.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def prompt_length(self, system: str, user: str) -> int:
        """Exact token count of the rendered prompt the model receives (chat template +
        specials + generation prompt) — the true input length for token-efficiency reporting."""
        return len(self.tokenizer(self._render(system, user))["input_ids"])

    def _render(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        common = {"tokenize": False, "add_generation_prompt": True}
        try:  # Qwen3 templates accept enable_thinking; others don't
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **common
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **common)

    def generate(self, system: str, user: str, *, max_new_tokens: int = 32) -> str:
        import torch

        text = self._render(system, user)
        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_one_batch(self, texts: list[str], max_new_tokens: int) -> list[str]:
        import torch

        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [t.strip() for t in decoded]

    def generate_batch(
        self,
        prompts: Sequence[tuple[str, str]],
        *,
        max_new_tokens: int = 32,
        batch_size: int = 32,
    ) -> list[str]:
        """Greedy-generate for many (system, user) prompts. Cached + batched (left-padded).

        Misses are sorted by rendered length so each batch pads to a similar length — this cuts
        wasted compute and peak memory (long and short prompts don't share a batch).
        """
        results: list[str | None] = [None] * len(prompts)
        misses: list[int] = []
        for i, (system, user) in enumerate(prompts):
            cached = (
                self._cache.get(generation_cache_key(self.model_id, system, user, max_new_tokens))
                if self._cache
                else None
            )
            if cached is not None:
                results[i] = cached["text"]
            else:
                misses.append(i)

        rendered = {i: self._render(*prompts[i]) for i in misses}
        misses.sort(key=lambda i: len(rendered[i]))

        pending: dict[str, dict] = {}
        for b in tqdm(range(0, len(misses), batch_size), desc="generate", unit="batch"):
            idxs = misses[b : b + batch_size]
            decoded = self._generate_one_batch([rendered[i] for i in idxs], max_new_tokens)
            for i, text in zip(idxs, decoded, strict=True):
                results[i] = text
                if self._cache:
                    system, user = prompts[i]
                    pending[generation_cache_key(self.model_id, system, user, max_new_tokens)] = {
                        "text": text
                    }
            if self._cache and len(pending) >= 256:  # persist incrementally → resumable
                self._cache.put_many(pending)
                pending.clear()
        if self._cache and pending:
            self._cache.put_many(pending)
        return [r if r is not None else "" for r in results]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def generate_batch_ids(
        self,
        prompts: Sequence[tuple[str, str]],
        *,
        max_new_tokens: int = 64,
        batch_size: int = 32,
    ) -> list[list[int]]:
        """Greedy-generate and return the trimmed NEW token-ids per prompt (teacher paths)."""
        import torch

        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        order = sorted(range(len(prompts)), key=lambda i: len(self._render(*prompts[i])))
        out: list[list[int]] = [[] for _ in prompts]
        for b in tqdm(range(0, len(order), batch_size), desc="teacher-paths", unit="batch"):
            idxs = order[b : b + batch_size]
            texts = [self._render(*prompts[i]) for i in idxs]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self._device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad
                )
            new_tokens = gen[:, enc["input_ids"].shape[1] :].tolist()
            for i, row in zip(idxs, new_tokens, strict=True):
                out[i] = trim_generated_path(row, eos, pad)
        return out
