"""Control ("capability-preservation") CF-Train tasks.

These mention the subject — so the concept tokens are injected at its span — but the task is
NOT about the subject's facts (translation, continuation, echo, an embedded question, …). The
distillation target is the teacher's *normal* output (frozen model + graph-as-text behaves
normally on these). They teach the encoder that concept tokens supply available knowledge
WITHOUT hijacking the model / forcing it to talk about the subject — guarding against
behavioral capability loss even though the LLM is frozen.
"""

from __future__ import annotations

import random

from conceptformer.generate.schema import CFTrainQA
from conceptformer.schemas import Subgraph

CONTROL_TEMPLATES = (
    'Translate into French: "{name} arrived early today."',
    "Continue in one sentence: {name} opened the door and",
    "Repeat this name back exactly: {name}",
    "{name} asked: what is two plus two?",
    'Rewrite this more formally: "hey, did {name} show up yet?"',
    'How many words are in the name "{name}"?',
)


def control_tasks(sg: Subgraph, *, n: int = 2, seed: int = 0) -> list[CFTrainQA]:
    """Pick ``n`` non-fact control prompts for the entity (deterministic per subject)."""
    name = sg.center.label or sg.center.qid
    rng = random.Random(f"{sg.center.qid}|control|{seed}")
    templates = rng.sample(CONTROL_TEMPLATES, min(n, len(CONTROL_TEMPLATES)))
    return [
        CFTrainQA(
            subject_qid=sg.center.qid,
            subject_label=name,
            question=template.format(name=name),
            answer=None,
            task_type="control",
        )
        for template in templates
    ]
