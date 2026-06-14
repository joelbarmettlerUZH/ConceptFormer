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


