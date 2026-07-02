"""Retrieval modes for the text-RAG budget baseline (pure ranking; no GPU in the helpers).

The token-efficiency figure needs a *defensible* competitor at small budgets. Top-PageRank
truncation (``verbalize_budgeted``) is query-independent — the honest apples-to-apples for
query-independent concept tokens — but a reviewer will ask what a query-AWARE retriever does
with the same budget: it can often place the one relevant fact inside 8-16 tokens. We report
both (plus an LLM-written budgeted summary, the strong query-independent text competitor), so
the figure can no longer be attacked as a strawman.

Ranking uses the frozen LLM's own mean-pooled label embeddings (same ``LabelEmbedder`` the
featurizer uses) — no extra retriever model, and the comparison stays within one embedding
space.
"""

from __future__ import annotations

import torch
from torch import Tensor

from conceptformer.model.featurizer import LabelEmbedder
from conceptformer.schemas import Subgraph


def rank_edges_by_question(sg: Subgraph, question: str, embedder: LabelEmbedder) -> Subgraph:
    """Reorder ``sg.edges`` by cosine similarity between the question and each edge's text.

    Edge text = "property: neighbor" (the same surface the verbalizer renders), embedded with
    the frozen LLM's mean-pooled embeddings. Returns a copy of ``sg`` with edges re-ranked so
    the budgeted verbalizer keeps the most question-relevant facts first.
    """
    if not sg.edges:
        return sg
    edge_texts = [
        f"{e.property_label or e.property_id}: {e.neighbor.label or e.neighbor.qid}"
        for e in sg.edges
    ]
    vectors = embedder([question, *edge_texts]).float()
    q_vec, edge_vecs = vectors[0], vectors[1:]
    scores = _cosine(q_vec, edge_vecs)
    order = torch.argsort(scores, descending=True).tolist()
    return sg.model_copy(update={"edges": [sg.edges[i] for i in order]})


def _cosine(q: Tensor, m: Tensor) -> Tensor:
    q_norm = q / q.norm().clamp(min=1e-8)
    m_norm = m / m.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return m_norm @ q_norm


def summary_prompt(sg_facts: str, entity_label: str, budget: int) -> tuple[str, str]:
    """(system, user) asking the frozen LLM for a <= ``budget``-token summary of the facts.

    The budget is *enforced* by the caller via ``max_new_tokens=budget`` — the instruction only
    steers the model to prioritize; truncation guarantees the accounting is honest.
    """
    system = "You compress facts. Output only the compressed facts, nothing else."
    user = (
        f"Compress the following facts about {entity_label} into at most {budget} tokens, "
        f"keeping the most identifying facts first:\n\n{sg_facts}"
    )
    return system, user
