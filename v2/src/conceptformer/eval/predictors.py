"""Predictors = systems under test. A predictor maps (examples, subgraphs) → answer strings.

``TextPredictor`` covers the text-prompt conditions (base, RAG). A future
``ConceptFormerPredictor`` will implement the same interface via injected soft tokens, so
the harness scores all conditions identically.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from conceptformer.model.chat import ChatModel
from conceptformer.schemas import QAExample, Subgraph
from conceptformer.verbalize import verbalize_budgeted

SYSTEM_BASE = "You are a helpful assistant. Answer with just the answer, as briefly as possible."
SYSTEM_RAG = (
    "You are a helpful assistant. Use the provided facts to answer the question. "
    "Answer with just the answer, as briefly as possible."
)

# (example, subgraph|None) -> (system, user)
PromptBuilder = Callable[[QAExample, Subgraph | None], tuple[str, str]]


# Bump when any prompt below changes, so reports with different prompts are never conflated.
PROMPT_VERSION = "v1"


def prompt_spec(condition: str) -> dict:
    """Record of the prompt used, stored in eval reports for reproducibility/comparability."""
    system = SYSTEM_RAG if condition == "rag" else SYSTEM_BASE
    return {
        "version": PROMPT_VERSION,
        "system": system,
        "user_format": "Question: {question}\\nAnswer:",
        "system_sha": hashlib.sha256(system.encode("utf-8")).hexdigest()[:12],
    }


def base_prompt(example: QAExample, subgraph: Subgraph | None) -> tuple[str, str]:
    """No-knowledge condition: the question alone."""
    return SYSTEM_BASE, f"Question: {example.question}\nAnswer:"


def make_rag_prompt(count_tokens: Callable[[str], int], budget: int) -> PromptBuilder:
    """Build the graph-in-context prompt builder, bounded by a token ``budget`` (the model's
    context window) — top-ranked neighbors fill the budget; the snapshot itself stays complete.
    """

    def rag_prompt(example: QAExample, subgraph: Subgraph | None) -> tuple[str, str]:
        if subgraph is None:
            return base_prompt(example, subgraph)
        facts = verbalize_budgeted(subgraph, count_tokens, budget)
        return SYSTEM_RAG, f"{facts}\n\nQuestion: {example.question}\nAnswer:"

    return rag_prompt


class Predictor(Protocol):
    name: str

    def predict_batch(
        self, examples: Sequence[QAExample], subgraphs: Mapping[str, Subgraph] | None
    ) -> list[str]: ...


class TextPredictor:
    """Prompt a chat model with a (system, user) built per example."""

    def __init__(
        self,
        model: ChatModel,
        prompt_builder: PromptBuilder,
        name: str,
        *,
        max_new_tokens: int = 32,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.prompt_builder = prompt_builder
        self.name = name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def predict_batch(
        self, examples: Sequence[QAExample], subgraphs: Mapping[str, Subgraph] | None
    ) -> list[str]:
        sgs = subgraphs or {}
        prompts = [self.prompt_builder(e, sgs.get(e.subject_qid)) for e in examples]
        return self.model.generate_batch(
            prompts, max_new_tokens=self.max_new_tokens, batch_size=self.batch_size
        )
