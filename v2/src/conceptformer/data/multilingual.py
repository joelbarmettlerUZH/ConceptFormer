"""Multilingual transfer eval: translate a QA set, then detect which language the model answers in.

Eval-only, no retraining. ConceptFormer's concept tokens are built purely from the frozen LLM's
embeddings of ENGLISH Wikidata labels. If injecting those tokens still lifts accuracy when the
question is asked in another language, the injected knowledge is language-agnostic on the LLM side
-- a claim no compression system has tested. We translate the question (keeping the entity mention
locatable, since the splice is mention-anchored) and score two mention conditions per language:
the entity kept in its English surface form, and the entity localized to its target-language
Wikidata label. Scoring against BOTH the English and target-language answer-alias sets, per item,
tells us which language the model answered in -- the second research question.

Pure helpers only (schema, mention substitution, answer-language classification); the vLLM
translation and Wikidata alias fetch live in the CLI.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from conceptformer.eval.metrics import word_boundary_match


class TranslatedQA(BaseModel):
    """One QA row translated to a target language, with the entity mention tracked both ways.

    ``question_en_mention`` keeps the entity in its English surface form inside the translated
    sentence; ``question_localized`` substitutes the target-language label. Both are stored so a
    single translation pass feeds two eval conditions.
    """

    subject_qid: str
    lang: str
    question_source: str  # the original English question
    question_en_mention: str
    question_localized: str
    mention_en: str
    mention_localized: str
    answer_labels_en: list[str] = Field(default_factory=list)
    answer_labels_localized: list[str] = Field(default_factory=list)


class TranslationResult(BaseModel):
    """Structured output the translator LLM must return (keeps the entity span recoverable)."""

    question_translated: str  # full question in the target language, entity in ENGLISH form
    mention_translated: str  # the entity's natural target-language surface form


def localize_mention(question_en_mention: str, mention_en: str, mention_localized: str) -> str:
    """Swap the English entity surface form for its localized label in a translated question.

    The translator is instructed to keep the entity in English so its span is findable; this
    produces the fully-localized variant. If the English mention is not present verbatim (the
    translator paraphrased it), we fall back to appending nothing and return the sentence
    unchanged -- the caller drops such rows rather than guess a substitution.
    """
    if not mention_en or mention_en not in question_en_mention:
        return question_en_mention
    return question_en_mention.replace(mention_en, mention_localized)


def mention_preserved(question_en_mention: str, mention_en: str) -> bool:
    """Whether the English entity span survived translation verbatim (required for anchoring)."""
    return bool(mention_en) and mention_en in question_en_mention


def classify_answer_language(
    prediction: str, aliases_en: list[str], aliases_localized: list[str]
) -> str:
    """Which language's answer surface forms the prediction matches: en/localized/both/neither.

    Uses the same word-boundary alias matching as the accuracy scorer. "both" happens when the
    English and localized labels coincide (common for names/numbers) or the model emits both; it
    counts as correct for accuracy but is reported separately so the language signal stays honest.
    """
    hit_en = bool(aliases_en) and word_boundary_match(prediction, aliases_en)
    hit_loc = bool(aliases_localized) and word_boundary_match(prediction, aliases_localized)
    if hit_en and hit_loc:
        return "both"
    if hit_en:
        return "en"
    if hit_loc:
        return "localized"
    return "neither"
