"""MetaQA (WikiMovies) loader: cross-graph transfer data for ConceptFormer.

MetaQA (Zhang et al., AAAI 2018; CC BY 3.0) is a movie knowledge graph (~43k entities, 9
relations, ~134k triples in ``subject|relation|object`` lines) with a 1-hop QA split whose
questions name the subject entity in [brackets] and whose answers are pipe-separated surface
forms -- i.e. exactly PopQA-shaped, on a genuinely different graph. That makes it the cheapest
credible test of the encoder's inductive claim: features are built purely from label strings,
so a Wikidata-trained encoder should transfer zero-shot if it learned graph->concept rather
than Wikidata idioms.

Design notes:
- MetaQA has no entity ids; the label IS the identity (questions bracket the exact KB string),
  so ``qid = label``.
- KB triples are movie-centric, but 1-hop questions also ask from the object side ("what films
  did [actor] appear in"), so each entity's neighborhood contains its outgoing edges AND
  incoming edges under hand-written reverse labels (9 relations -- auditable by inspection).
  Reverse labels are phrased so the (property, neighbor) pair reads naturally for the frozen
  LLM's label embeddings, mirroring how Wikidata property labels read.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator

from conceptformer.schemas import Edge, Entity, QAExample, Subgraph

# Movie-side property label per KB relation (underscores -> spaces reads like a Wikidata label).
FORWARD_LABELS: dict[str, str] = {
    "directed_by": "directed by",
    "starred_actors": "cast member",
    "written_by": "written by",
    "has_genre": "genre",
    "release_year": "publication year",
    "in_language": "language",
    "has_tags": "tag",
    "has_imdb_rating": "IMDb rating",
    "has_imdb_votes": "IMDb votes",
}

# Object-side (reverse) property label: how the edge reads from the neighbor's perspective.
REVERSE_LABELS: dict[str, str] = {
    "directed_by": "director of",
    "starred_actors": "actor in",
    "written_by": "writer of",
    "has_genre": "genre of",
    "release_year": "publication year of",
    "in_language": "language of",
    "has_tags": "tag of",
    "has_imdb_rating": "IMDb rating of",
    "has_imdb_votes": "IMDb votes of",
}


def parse_kb_line(line: str) -> tuple[str, str, str] | None:
    """``subject|relation|object`` -> (s, r, o); None for malformed/empty lines."""
    parts = line.rstrip("\n").split("|")
    if len(parts) != 3 or not all(p.strip() for p in parts):
        return None
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def build_subgraphs(kb_lines: Iterable[str]) -> Iterator[Subgraph]:
    """One complete 1-hop neighborhood per entity (outgoing + reverse-labeled incoming edges)."""
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for line in kb_lines:
        triple = parse_kb_line(line)
        if triple is None:
            continue
        s, r, o = triple
        outgoing[s].append((r, o))
        incoming[o].append((r, s))
    for name in sorted(set(outgoing) | set(incoming)):
        edges = [
            Edge(property_id=r, property_label=FORWARD_LABELS.get(r, r.replace("_", " ")),
                 neighbor=Entity(qid=o, label=o))
            for r, o in outgoing.get(name, [])
        ] + [
            Edge(property_id=f"{r}~rev",
                 property_label=REVERSE_LABELS.get(r, f"{r.replace('_', ' ')} of"),
                 neighbor=Entity(qid=s, label=s))
            for r, s in incoming.get(name, [])
        ]
        yield Subgraph(center=Entity(qid=name, label=name), edges=edges,
                       n_edges_total=len(edges))


def parse_qa_line(line: str) -> tuple[str, str, list[str]] | None:
    """``what films did [subject] star in\\tA|B`` -> (subject, question sans brackets, answers).

    The bracketed span is the KB entity string verbatim -- MetaQA's stand-in for an entity id.
    Returns None for malformed lines (no brackets / no answers).
    """
    if "\t" not in line:
        return None
    question, _, answer_part = line.rstrip("\n").partition("\t")
    lo, hi = question.find("["), question.find("]")
    if lo < 0 or hi <= lo:
        return None
    subject = question[lo + 1 : hi]
    clean = (question[:lo] + subject + question[hi + 1 :]).strip()
    answers = [a.strip() for a in answer_part.split("|") if a.strip()]
    if not subject or not answers:
        return None
    return subject, clean, answers


def load_metaqa_qa(lines: Iterable[str], split: str = "test") -> list[QAExample]:
    """1-hop vanilla QA lines -> normalized ``QAExample`` rows (qid = subject label)."""
    out: list[QAExample] = []
    for line in lines:
        parsed = parse_qa_line(line)
        if parsed is None:
            continue
        subject, question, answers = parsed
        out.append(QAExample(
            source="metaqa", split=split, subject_qid=subject, relation="metaqa-1hop",
            question=question, answer_labels=answers,
        ))
    return out
