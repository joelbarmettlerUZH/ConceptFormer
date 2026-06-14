"""Prompts for CF-Train question generation.

The generator is asked for DIVERSE, natural, grounded questions — explicitly NOT templated
("What is X's Y?") forms — so the resulting training data doesn't overfit benchmark phrasing
and isn't reducible to property-lookup.
"""

from __future__ import annotations

from collections.abc import Callable

from conceptformer.schemas import Subgraph
from conceptformer.verbalize import verbalize_budgeted

# Item-valued properties that make low-value QA targets (bibliographic / metadata) — hidden
# from the generator so it asks about substantive facts. Kept small + extensible.
GENERATION_DENY_PROPERTIES = frozenset(
    {
        "P1343",  # described by source
        "P143",  # imported from Wikimedia project
        "P4656",  # Wikimedia import URL
        "P248",  # stated in
    }
)

GEN_SYSTEM = (
    "You write training questions about a single Wikidata entity. You are given the entity's "
    "name and a list of TRUE facts about it. Generate diverse, natural questions a curious "
    "person might genuinely ask.\n"
    "HARD RULES (a violation makes the example unusable):\n"
    "- Name the entity explicitly in EVERY question. NEVER use a pronoun (he/she/they/it) or a "
    "vague descriptor ('this politician', 'the cartoonist'). The entity's exact name must "
    "appear in each question.\n"
    "- Every question MUST be answerable SOLELY from the listed facts. Do not ask about "
    "anything requiring outside knowledge or multi-step inference beyond what is listed.\n"
    "- NO yes/no questions. Each answer must be a specific value or entity copied from the "
    "facts.\n"
    "STYLE:\n"
    "- Vary phrasing and structure. Do NOT use a rigid template like 'What is X's occupation?'. "
    "Write the way real people ask — direct, indirect, conversational.\n"
    "- Include a healthy share of 'compositional' questions that combine two or more facts.\n"
    "- Keep answers short; avoid near-duplicate questions.\n"
    "Return JSON matching the schema."
)


def build_generation_user(
    sg: Subgraph, n_questions: int, count_tokens: Callable[[str], int], budget: int
) -> str:
    substantive = sg.model_copy(
        update={"edges": [e for e in sg.edges if e.property_id not in GENERATION_DENY_PROPERTIES]}
    )
    facts = verbalize_budgeted(substantive, count_tokens, budget)
    name = sg.center.label or sg.center.qid
    return f"{facts}\n\nEntity: {name}\nGenerate {n_questions} questions about {name}."
