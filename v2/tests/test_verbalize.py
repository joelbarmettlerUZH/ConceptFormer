"""Offline tests for verbalization + answer matching."""

from conceptformer.schemas import Edge, Entity, Subgraph


def _sg() -> Subgraph:
    return Subgraph(
        center=Entity(qid="Q42", label="Douglas Adams"),
        edges=[
            Edge(property_id="P106", property_label="occupation",
                 neighbor=Entity(qid="Q1", label="writer")),
            Edge(property_id="P106", property_label="occupation",
                 neighbor=Entity(qid="Q2", label="novelist")),
            Edge(property_id="P19", property_label="place of birth",
                 neighbor=Entity(qid="Q350", label="Cambridge")),
        ],
        n_edges_total=3,
    )


def test_verbalize_groups_by_property_in_order():
    from conceptformer.verbalize import verbalize

    text = verbalize(_sg())
    assert text.splitlines() == [
        "Facts about Douglas Adams:",
        "- occupation: writer, novelist",  # same property grouped
        "- place of birth: Cambridge",
    ]


def test_verbalize_max_neighbors_caps_edges_by_rank():
    from conceptformer.verbalize import verbalize

    # caps to the first 2 (rank-ordered) edges → both occupations, drops place of birth
    text = verbalize(_sg(), max_neighbors=2)
    assert text.splitlines() == ["Facts about Douglas Adams:", "- occupation: writer, novelist"]


def test_verbalize_budgeted_keeps_top_ranked_within_budget():
    from conceptformer.verbalize import verbalize, verbalize_budgeted

    words = lambda text: len(text.split())  # noqa: E731 — a simple token-count stand-in

    # generous budget → complete verbalization
    assert verbalize_budgeted(_sg(), words, budget=1000) == verbalize(_sg())

    # tight budget → drops the lowest-ranked edges (place of birth), keeps top occupations
    budget = words("Facts about Douglas Adams:\n- occupation: writer, novelist")
    tight = verbalize_budgeted(_sg(), words, budget=budget)
    assert "place of birth" not in tight
    assert "occupation: writer, novelist" in tight


def test_verbalize_budgeted_returns_largest_fitting_prefix():
    from conceptformer.verbalize import verbalize, verbalize_budgeted

    words = lambda text: len(text.split())  # noqa: E731
    sg = _sg()  # 3 rank-ordered edges
    # budget that exactly fits the top-2 edges → binary search must return precisely that prefix
    budget = words(verbalize(sg, max_neighbors=2))
    assert verbalize_budgeted(sg, words, budget=budget) == verbalize(sg, max_neighbors=2)




def _big_sg(n=30):
    # rank-sorted edges; the answer is a LOW-rank (tail) neighbor that a top-k budget would cut.
    edges = [Edge(property_id=f"P{i}", property_label=f"rel{i}",
                  neighbor=Entity(qid=f"Q{i}", label=f"neighbor_{i}")) for i in range(n)]
    return Subgraph(center=Entity(qid="QX", label="X"), edges=edges, n_edges_total=n)


def test_verbalize_with_answer_guarantees_the_answer_edge():
    from conceptformer.verbalize import verbalize_budgeted, verbalize_with_answer

    sg = _big_sg(30)
    count = lambda t: len(t.split())  # noqa: E731 — crude word tokenizer for the test
    budget = 12  # only fits a handful of edges
    tail_answer = "Q27"  # a deep-tail neighbor (rank 27 of 30)
    plain = verbalize_budgeted(sg, count, budget)
    guaranteed = verbalize_with_answer(sg, tail_answer, count, budget)
    assert "neighbor_27" not in plain  # top-PageRank budget cuts the tail answer
    assert "neighbor_27" in guaranteed  # guaranteed version always includes it


def test_verbalize_with_answer_rng_shuffles_distractors_deterministically():
    import random

    from conceptformer.verbalize import verbalize_with_answer

    sg = _big_sg(30)
    count = lambda t: len(t.split())  # noqa: E731
    a = verbalize_with_answer(sg, "Q5", count, 20, rng=random.Random(0))
    b = verbalize_with_answer(sg, "Q5", count, 20, rng=random.Random(0))
    c = verbalize_with_answer(sg, "Q5", count, 20, rng=random.Random(1))
    assert "neighbor_5" in a  # answer always present
    assert a == b  # same seed → same subset
    assert a != c  # different seed → different distractors (subsampling)


def test_verbalize_with_answer_none_falls_back():
    from conceptformer.verbalize import verbalize_budgeted, verbalize_with_answer

    sg = _big_sg(10)
    count = lambda t: len(t.split())  # noqa: E731
    assert verbalize_with_answer(sg, None, count, 8) == verbalize_budgeted(sg, count, 8)
