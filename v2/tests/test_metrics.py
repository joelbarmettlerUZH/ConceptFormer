"""Exhaustive, spec-driven tests for the answer-matching metrics.

These assert the INTENDED behavior of each metric (and, for ``popqa_official``, faithful
reproduction of the published reference), with radical edge cases — Unicode, punctuation,
hyphens, regex metacharacters, boundaries, casing, empties — because the metric is the most
peer-review-sensitive part of the codebase.
"""

import pytest

from conceptformer.eval.metrics import (
    MIN_ALIAS_LEN,
    _normalize,
    popqa_official,
    strict_match,
    word_boundary_match,
)


# ======================================================================================
# popqa_official — MUST reproduce Mallen et al. (2023) exactly: case-insensitive substring.
# Locking to the reference here is correct: the spec IS "the published metric", which is
# what makes our headline number comparable to other papers. Its known over-leniency is
# reproduced on purpose (and fixed by word_boundary_match, tested below).
# ======================================================================================
@pytest.mark.parametrize(
    ("prediction", "aliases", "expected"),
    [
        ("He was a writer.", ["writer"], True),  # plain substring
        ("He was a WRITER.", ["writer"], True),  # case-insensitive (pred)
        ("he was a writer", ["Writer"], True),  # case-insensitive (alias)
        ("born in New York City", ["New York City"], True),  # multi-word
        ("a novelist", ["writer"], False),  # genuinely absent
        ("anything", [], False),  # no aliases → never correct
        # --- the documented reference flaws we faithfully reproduce ---
        ("Joseph was born in Budapest", ["W"], True),  # "w" ∈ "was"  (false positive!)
        ("a career in politics", ["pol"], True),  # "pol" ∈ "politics" (false positive!)
        ("born in London", ["ON"], True),  # "on" ∈ "London" (false positive!)
    ],
)
def test_popqa_official_reproduces_reference(prediction, aliases, expected):
    assert popqa_official(prediction, aliases) is expected


# ======================================================================================
# word_boundary_match — corrected metric: word boundaries + drop junk aliases (< MIN len).
# ======================================================================================
@pytest.mark.parametrize(
    ("prediction", "aliases", "expected", "why"),
    [
        ("Joseph was born in Budapest", ["W"], False, "1-char alias dropped (the smoking gun)"),
        ("born on a farm", ["ON"], False, "2-char alias dropped"),
        ("a career in politics", ["pol"], False, "boundary stops in-word match"),
        ("study of pol", ["pol"], True, "standalone token matches"),
        ("Bart Simpson", ["art"], False, "no match inside a word"),
        ("started early", ["art"], False, "no match mid-word"),
        ("He was a Writer.", ["writer"], True, "trailing punctuation is a boundary"),
        ("writers and poets", ["writer"], False, "no stemming: 'writer' != 'writers'"),
        ("born in New York City.", ["New York City", "NYC"], True, "multi-word phrase"),
        ("lives in the U.S.", ["U.S."], True, "dotted alias, trailing-period boundary"),
        ("from the U.S.A. originally", ["U.S."], False, "U.S. must not match U.S.A."),
        ("a comedy-drama film", ["drama"], True, "hyphen acts as a word boundary"),
        ("axb", ["a.b"], False, "alias dot is literal (no regex injection)"),
        ("born in Zürich", ["Zürich"], True, "unicode alias matches"),
        ("born in Zürich", ["rich"], False, "no match inside a unicode word"),
        ("", ["writer"], False, "empty prediction"),
        ("writer", [""], False, "empty alias is ignored"),
    ],
)
def test_word_boundary_match(prediction, aliases, expected, why):
    assert word_boundary_match(prediction, aliases) is expected, why


def test_min_alias_len_constant_is_sane():
    # Guards the documented contract: 1- and 2-char junk aliases are dropped, 3+ kept.
    assert MIN_ALIAS_LEN == 3


# ======================================================================================
# strict_match — pessimistic: normalized first-line exact match.
# ======================================================================================
@pytest.mark.parametrize(
    ("prediction", "aliases", "expected", "why"),
    [
        ("Writer.", ["writer"], True, "terse answer + trailing punctuation"),
        ("He was a writer.", ["writer"], False, "wrapped in a sentence → reject"),
        ("writer, doctor, lawyer", ["writer"], False, "enumeration → reject"),
        ("a politician", ["politician"], True, "leading article stripped"),
        ("The Beatles", ["Beatles"], True, "article stripped from prediction"),
        ("writer\nand a poet too", ["writer"], True, "first line only"),
        ("J-pop", ["J-pop"], True, "punctuation normalized on both sides"),
        ("", ["writer"], False, "empty prediction"),
        ("writer", [""], False, "empty alias ignored"),
    ],
)
def test_strict_match(prediction, aliases, expected, why):
    assert strict_match(prediction, aliases) is expected, why


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The A an apple!", "apple"),  # articles + punctuation removed
        ("  Multiple   Spaces ", "multiple spaces"),  # whitespace collapsed
        ("line1\nline2", "line1"),  # first line only
        ("UPPER", "upper"),  # lowercased
    ],
)
def test_normalize(text, expected):
    assert _normalize(text) == expected


# ======================================================================================
# Cross-metric invariants (defensibility): a word-boundary hit on gold is, by construction,
# also an official substring hit. (The reverse is not true — that's the false-positive gap.)
# ======================================================================================
@pytest.mark.parametrize(
    ("prediction", "aliases"),
    [
        ("He is a writer", ["writer"]),
        ("Was born", ["W"]),
        ("in politics", ["politics"]),
        ("nothing here", ["xyz"]),
        ("born in Zürich", ["Zürich"]),
        ("a comedy-drama", ["drama"]),
    ],
)
def test_word_boundary_implies_official(prediction, aliases):
    if word_boundary_match(prediction, aliases):
        assert popqa_official(prediction, aliases), "word-boundary hit must be a substring hit"


def test_metrics_are_deterministic():
    args = ("He was a Writer and humorist", ["writer", "author"])
    assert popqa_official(*args) == popqa_official(*args)
    assert word_boundary_match(*args) == word_boundary_match(*args)
    assert strict_match(*args) == strict_match(*args)
