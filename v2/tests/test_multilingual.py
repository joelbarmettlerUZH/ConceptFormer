"""Offline tests for the multilingual-eval helpers (data/multilingual.py)."""

from conceptformer.data.multilingual import (
    classify_answer_language,
    localize_mention,
    mention_preserved,
)


def test_localize_mention_substitutes_when_span_present():
    q = "In welchem Jahr wurde Albert Einstein geboren?"
    assert localize_mention(q, "Albert Einstein", "Albert Einstein") == q  # same in de
    q2 = "Wer hat Cologne gegründet?"
    assert localize_mention(q2, "Cologne", "Köln") == "Wer hat Köln gegründet?"


def test_localize_mention_unchanged_when_span_missing():
    # Translator paraphrased the entity away: no safe substitution, return as-is.
    q = "Wer war der Autor?"
    assert localize_mention(q, "Isaac Asimov", "Isaac Asimov") == q


def test_mention_preserved():
    assert mention_preserved("Wer regierte Prussia?", "Prussia")
    assert not mention_preserved("Wer regierte dort?", "Prussia")
    assert not mention_preserved("anything", "")


def test_classify_answer_language():
    # English-only surface form in the prediction.
    assert classify_answer_language("The answer is Germany.", ["Germany"], ["Deutschland"]) == "en"
    # Localized surface form.
    assert classify_answer_language("Die Antwort ist Deutschland.",
                                    ["Germany"], ["Deutschland"]) == "localized"
    # Coinciding labels (proper name) -> both.
    assert classify_answer_language("Paris", ["Paris"], ["Paris"]) == "both"
    # No match either way.
    assert classify_answer_language("Mars", ["Germany"], ["Deutschland"]) == "neither"
    # Empty localized aliases must not crash and cannot be "localized".
    assert classify_answer_language("Germany", ["Germany"], []) == "en"
