"""Offline tests for tail-heavy entity selection from a PageRank-ranked pool."""

import bz2

from conceptformer.data.select import (
    SAMPLE_FRACS,
    SelectedEntity,
    load_entities,
    load_pagerank,
    load_pagerank_map,
    save_entities,
    select_entities,
)


def test_load_pagerank_parses_and_skips_garbage(tmp_path):
    path = tmp_path / "pr.bz2"
    with bz2.open(path, "wt", encoding="utf-8") as fh:
        fh.write("Q1\t100.5\nQ2\t50.25\n\nbadline_no_tab\nQ3\t1.0\n")
    assert load_pagerank(path) == [("Q1", 100.5), ("Q2", 50.25), ("Q3", 1.0)]
    assert load_pagerank_map(path) == {"Q1": 100.5, "Q2": 50.25, "Q3": 1.0}

# 1000 entities, rank descending (index 0 = most popular). Pool tiers: high=[0,50),
# mid=[50,200), low=[200,1000).
ENTRIES = [(f"Q{i}", float(1000 - i)) for i in range(1000)]


def test_sample_composition_is_tail_heavy():
    sel = select_entities(ENTRIES, n=100, seed=1)
    tiers = {t: sum(e.tier == t for e in sel) for t in ("low", "mid", "high")}
    assert tiers == {"low": 70, "mid": 20, "high": 10}  # matches SAMPLE_FRACS
    assert SAMPLE_FRACS == (0.70, 0.20, 0.10)


def test_tiers_map_to_the_right_pool_regions():
    sel = select_entities(ENTRIES, n=100, seed=2)
    for e in sel:
        idx = int(e.qid[1:])
        if e.tier == "high":
            assert idx < 50
        elif e.tier == "mid":
            assert 50 <= idx < 200
        else:
            assert idx >= 200


def test_excluded_qids_never_selected():
    exclude = {f"Q{i}" for i in range(0, 1000, 2)}  # all even ids
    sel = select_entities(ENTRIES, n=100, exclude=exclude, seed=3)
    assert all(e.qid not in exclude for e in sel)
    assert all(int(e.qid[1:]) % 2 == 1 for e in sel)


def test_selection_is_deterministic_under_seed():
    a = select_entities(ENTRIES, n=50, seed=7)
    b = select_entities(ENTRIES, n=50, seed=7)
    assert [e.qid for e in a] == [e.qid for e in b]


def test_save_load_roundtrip(tmp_path):
    sel = [SelectedEntity(qid="Q1", rank=5.0, tier="low")]
    path = save_entities(sel, tmp_path / "e.jsonl")
    assert load_entities(path) == sel
