"""Offline spec tests for neighborhood extraction.

These assert the *intended* behavior of the extractor (see ``_build_subgraph``):
1. complete by default (no cap),
2. no self-loops,
3. (property, neighbor) dedup — same neighbor via different properties is kept,
4. deterministic popularity ordering,
5. optional cap keeps the top-ranked + always-keep,
6. label resolution prefers en then falls back to mul.
"""

from conceptformer.data.coverage import audit_coverage
from conceptformer.data.wikidata import _build_subgraph, _localized
from conceptformer.schemas import QAExample

LANGS = ["en", "mul"]


def _entity_value(qid: str) -> dict:
    return {
        "snaktype": "value",
        "datavalue": {"type": "wikibase-entityid", "value": {"entity-type": "item", "id": qid}},
    }


def _center(edges: list[tuple[str, str]], qid: str = "C") -> dict:
    """Build a raw center entity from [(property_id, neighbor_qid)] statements."""
    claims: dict[str, list[dict]] = {}
    for pid, nid in edges:
        claims.setdefault(pid, []).append({"rank": "normal", "mainsnak": _entity_value(nid)})
    return {"qid": qid, "labels": {"en": {"value": "Center"}}, "claims": claims}


def _meta(sitelinks: dict[str, int]) -> dict[str, dict]:
    return {
        qid: {
            "labels": {"en": {"value": f"label-{qid}"}},
            "sitelinks": {f"s{i}": {} for i in range(n)},
        }
        for qid, n in sitelinks.items()
    }


def _props(pids: list[str]) -> dict[str, dict]:
    return {p: {"labels": {"en": {"value": p.lower()}}} for p in pids}


def _build(center: dict, neighbor_meta: dict, property_meta: dict, **kw):
    return _build_subgraph("C", center, neighbor_meta, property_meta, langs=LANGS, **kw)


def test_complete_neighborhood_by_default():
    center = _center([("P1", "Q1"), ("P2", "Q2"), ("P3", "Q3")])
    sg = _build(center, _meta({"Q1": 1, "Q2": 1, "Q3": 1}), _props(["P1", "P2", "P3"]))
    assert sg.n_edges_total == 3
    assert len(sg.edges) == 3  # nothing dropped
    assert sg.capped is False


def test_self_loops_excluded():
    center = _center([("P1", "C"), ("P2", "Q2")])  # P1 points back at the center
    sg = _build(center, _meta({"Q2": 1}), _props(["P1", "P2"]))
    assert sg.neighbor_qids == ["Q2"]
    assert sg.n_edges_total == 1


def test_same_neighbor_via_different_properties_kept_separately():
    center = _center([("P1", "Q9"), ("P2", "Q9")])  # Q9 reached two distinct ways
    sg = _build(center, _meta({"Q9": 1}), _props(["P1", "P2"]))
    assert sg.n_edges_total == 2
    assert {e.property_id for e in sg.edges} == {"P1", "P2"}
    assert sg.neighbor_qids == ["Q9", "Q9"]


def test_duplicate_property_neighbor_pair_collapsed():
    center = _center([("P1", "Q9"), ("P1", "Q9")])  # exact duplicate statement
    sg = _build(center, _meta({"Q9": 1}), _props(["P1"]))
    assert sg.n_edges_total == 1


def test_neighbors_ordered_by_popularity_desc():
    center = _center([("P1", "Qlow"), ("P2", "Qhigh"), ("P3", "Qmid")])
    sg = _build(center, _meta({"Qlow": 0, "Qmid": 5, "Qhigh": 10}), _props(["P1", "P2", "P3"]))
    assert sg.neighbor_qids == ["Qhigh", "Qmid", "Qlow"]


def test_pagerank_overrides_sitelink_ranking_when_provided():
    center = _center([("P1", "Qa"), ("P2", "Qb"), ("P3", "Qc")])
    meta = _meta({"Qa": 1, "Qb": 1, "Qc": 1})  # equal sitelinks
    pagerank = {"Qa": 10.0, "Qb": 30.0, "Qc": 20.0}  # but distinct pageranks
    sg = _build(center, meta, _props(["P1", "P2", "P3"]), pagerank=pagerank)
    assert sg.neighbor_qids == ["Qb", "Qc", "Qa"]  # ordered by pagerank desc


def test_falls_back_to_sitelinks_without_pagerank():
    center = _center([("P1", "Qa"), ("P2", "Qb")])
    sg = _build(center, _meta({"Qa": 5, "Qb": 0}), _props(["P1", "P2"]), pagerank={})
    assert sg.neighbor_qids == ["Qa", "Qb"]  # empty pagerank → sitelink ordering preserved


def test_explicit_cap_keeps_top_ranked():
    center = _center([("P1", "Qlow"), ("P2", "Qhigh"), ("P3", "Qmid")])
    meta = _meta({"Qlow": 0, "Qmid": 5, "Qhigh": 10})
    sg = _build(center, meta, _props(["P1", "P2", "P3"]), cap=2)
    assert sg.capped is True
    assert sg.n_edges_total == 3
    assert sg.neighbor_qids == ["Qhigh", "Qmid"]


def test_explicit_cap_retains_always_keep_below_cut():
    center = _center([("P1", "Qlow"), ("P2", "Qhigh"), ("P3", "Qmid")])
    meta = _meta({"Qlow": 0, "Qmid": 5, "Qhigh": 10})
    sg = _build(center, meta, _props(["P1", "P2", "P3"]), cap=2, always_keep=["Qlow"])
    assert sg.capped is True
    assert "Qlow" in sg.neighbor_qids  # survives despite lowest rank
    assert len(sg.edges) == 2


def test_label_prefers_en_then_mul():
    both = {"labels": {"en": {"value": "English"}, "mul": {"value": "Multi"}}}
    only_mul = {"labels": {"mul": {"value": "Multi"}}}
    other = {"labels": {"fr": {"value": "Français"}}}
    assert _localized(both, "labels", LANGS) == "English"
    assert _localized(only_mul, "labels", LANGS) == "Multi"
    assert _localized(other, "labels", LANGS) is None


def test_audit_coverage(tmp_path):
    sg = _build_subgraph(
        "Q1", _center([("P19", "Q99")], qid="Q1"),
        _meta({"Q99": 3}), _props(["P19"]), langs=LANGS,
    )
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "subgraphs.jsonl").write_text(sg.model_dump_json() + "\n")

    examples = [
        QAExample(
            source="popqa", split="test", subject_qid="Q1", relation="place of birth",
            relation_id="P19", question="?", answer_qid="Q99", answer_labels=["Answer"],
        ),
        QAExample(
            source="popqa", split="test", subject_qid="Q2", relation="x",
            relation_id="P1", question="?", answer_qid="Q5", answer_labels=["x"],
        ),
    ]
    report = audit_coverage(examples, snap)
    assert report["subject_resolved"] == 1  # only Q1 is in the snapshot
    assert report["answer_in_graph"] == 1
    assert report["answer_via_relation"] == 1
    assert report["answer_in_graph_pct"] == 100.0
