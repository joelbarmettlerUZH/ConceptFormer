"""WorldCup2014 (WC-P) loader: a third cross-graph transfer testbed for ConceptFormer.

WC-P (Zhou et al., COLING 2018; the IRN repo) is a small football knowledge graph (World Cup
2014 squads: players, clubs, countries, positions) with a native 1-hop QA split whose questions
name the subject and whose answers are entity surface forms -- PopQA-shaped, on a graph whose
relation vocabulary (plays in club, plays for country, plays position) is disjoint from both
Wikidata and MetaQA. It is our sports-domain transfer graph after the movies-domain MetaQA.

Format (both files tab-separated):
- KB ``subject\trelation\tobject``. The KB already ships ``_inverse`` relations (e.g.
  ``club plays_in_club_inverse player``), so grouping rows by their subject column yields each
  entity's complete 1-hop neighborhood -- no hand-written reverse labels (unlike MetaQA).
- QA ``question\tanswer\tpath\tanswer_set\tneighborhood``. The subject is the path's first
  ``#``-separated token (machine-extractable, no NER), the asked relation is the second, and the
  accepted answers are the ``/``-separated ``answer_set``.

Entity identity is the raw underscore token (``qid``); labels shown to the featurizer replace
underscores with spaces so the frozen LLM reads natural strings.

Not vendored: fetched from the upstream repo at build time (the repo declares no license; we
redistribute nothing and cite Zhou et al., COLING 2018).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator

from conceptformer.schemas import Edge, Entity, QAExample, Subgraph

# Readable relation labels (forward + the KB's own inverse edges). Anything unmapped falls back
# to underscores-to-spaces, so a new relation still reads sensibly rather than crashing.
REL_LABELS: dict[str, str] = {
    "plays_in_club": "plays in club",
    "plays_in_club_inverse": "has club player",
    "plays_position": "plays position",
    "plays_position_inverse": "is position of",
    "plays_for_country": "plays for country",
    "plays_for_country_inverse": "has national player",
    "is_in_country": "is in country",
    "is_in_country_inverse": "is country of",
    "is_aged": "is aged",
    "is_aged_inverse": "is age of",
    "wears_number": "wears number",
    "wears_number_inverse": "is number of",
}

# Relations whose object is a numeric literal, not an entity: dropped from the entity-answer eval.
NUMERIC_RELATIONS = frozenset({"is_aged", "wears_number"})


def _pretty(token: str) -> str:
    """Raw underscore token -> natural surface string (``Alan_PULIDO`` -> ``Alan PULIDO``)."""
    return token.replace("_", " ").strip()


def parse_kb_line(line: str) -> tuple[str, str, str] | None:
    """``subject\\trelation\\tobject`` -> (s, r, o); None for malformed/empty lines."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3 or not all(p.strip() for p in parts):
        return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def build_subgraphs(kb_lines: Iterable[str]) -> Iterator[Subgraph]:
    """One complete 1-hop neighborhood per entity (the KB already carries inverse edges)."""
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for line in kb_lines:
        triple = parse_kb_line(line)
        if triple is None:
            continue
        s, r, o = triple
        outgoing[s].append((r, o))
        labels.setdefault(s, _pretty(s))
        labels.setdefault(o, _pretty(o))
    for name in sorted(outgoing):
        edges = [
            Edge(property_id=r, property_label=REL_LABELS.get(r, _pretty(r)),
                 neighbor=Entity(qid=o, label=labels[o]))
            for r, o in outgoing[name]
        ]
        yield Subgraph(center=Entity(qid=name, label=labels[name]), edges=edges,
                       n_edges_total=len(edges))


def parse_qa_line(line: str) -> tuple[str, str, str, list[str]] | None:
    """QA row -> (subject_token, relation, natural question, answer surface forms).

    Subject and relation come from the path column (``subj#rel#obj#<end>#obj``); answers from the
    ``/``-separated answer-set column. Returns None for malformed lines.
    """
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4:
        return None
    question, path, answer_set = parts[0], parts[2], parts[3]
    path_parts = path.split("#")
    if len(path_parts) < 3:
        return None
    subject, relation = path_parts[0], path_parts[1]
    answers = [a.strip() for a in answer_set.split("/") if a.strip()]
    if not subject or not relation or not answers:
        return None
    return subject, relation, _pretty(question), answers


def load_worldcup_qa(
    lines: Iterable[str], split: str = "test", drop_numeric: bool = True
) -> list[QAExample]:
    """WC-P QA lines -> normalized ``QAExample`` rows (qid = subject token).

    ``drop_numeric`` removes the two literal-valued relations (age, shirt number) so every kept
    answer is an entity, matching the PopQA / MetaQA entity-answer contract. Answer surface forms
    are stored both prettified and raw so word-boundary scoring matches either.
    """
    out: list[QAExample] = []
    for line in lines:
        parsed = parse_qa_line(line)
        if parsed is None:
            continue
        subject, relation, question, answers = parsed
        if drop_numeric and relation in NUMERIC_RELATIONS:
            continue
        surface = list(dict.fromkeys([_pretty(a) for a in answers] + answers))
        out.append(QAExample(
            source="worldcup", split=split, subject_qid=subject, relation=relation,
            question=question, answer_labels=surface,
        ))
    return out
