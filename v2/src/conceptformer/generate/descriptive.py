"""Descriptive ("templated") CF-Train tasks — the no-Gemma leg of the task mix.

A small bank of fixed instruction templates asks the model to describe the entity holistically.
There's no single gold answer: at distillation time the teacher (backbone + verbalized graph)
generates a free-form description, and the student must reproduce it from the concept tokens.
This is the "essence" signal that makes ConceptFormer more than a property-lookup.
"""

from __future__ import annotations

import random

from conceptformer.generate.schema import CFTrainQA
from conceptformer.schemas import Subgraph

DESCRIPTIVE_TEMPLATES = (
    "Describe {name}.",
    "Tell me about {name}.",
    "What is {name} known for?",
    "Give a brief summary of {name}.",
    "Who or what is {name}?",
    "Provide an overview of {name}.",
)


def descriptive_tasks(sg: Subgraph, *, n: int = 2, seed: int = 0) -> list[CFTrainQA]:
    """Pick ``n`` descriptive templates for the entity (deterministic per subject)."""
    name = sg.center.label or sg.center.qid
    # String seed → reproducible across runs (unlike the salted built-in hash()).
    rng = random.Random(f"{sg.center.qid}|{seed}")
    templates = rng.sample(DESCRIPTIVE_TEMPLATES, min(n, len(DESCRIPTIVE_TEMPLATES)))
    return [
        CFTrainQA(
            subject_qid=sg.center.qid,
            subject_label=name,
            question=template.format(name=name),
            answer=None,
            task_type="descriptive",
        )
        for template in templates
    ]
