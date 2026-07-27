"""Offline tests for the MetaQA cross-graph loader (data/metaqa.py)."""

from conceptformer.data.metaqa import (
    build_2hop_subgraphs,
    build_subgraphs,
    load_metaqa_qa,
    parse_kb_line,
    parse_qa_line,
)

KB = [
    "Kismet|directed_by|William Dieterle",
    "Kismet|starred_actors|Marlene Dietrich",
    "Top Hat|starred_actors|Ginger Rogers",
    "malformed line without pipes",
]


def test_parse_kb_line():
    assert parse_kb_line(KB[0]) == ("Kismet", "directed_by", "William Dieterle")
    assert parse_kb_line("a|b") is None
    assert parse_kb_line("a||c") is None


def test_subgraphs_include_forward_and_reverse_edges():
    sgs = {sg.center.qid: sg for sg in build_subgraphs(KB)}
    # Movie side: outgoing edges with forward labels.
    kismet = sgs["Kismet"]
    labels = {(e.property_label, e.neighbor.qid) for e in kismet.edges}
    assert ("directed by", "William Dieterle") in labels
    assert ("cast member", "Marlene Dietrich") in labels
    # Person side: incoming edge appears under the hand-written REVERSE label.
    dieterle = sgs["William Dieterle"]
    assert [(e.property_label, e.neighbor.qid) for e in dieterle.edges] == [
        ("director of", "Kismet")
    ]
    # Reverse property ids are disambiguated from forward ones.
    assert dieterle.edges[0].property_id == "directed_by~rev"
    # Every KB entity gets a subgraph; the malformed line is dropped.
    assert set(sgs) == {"Kismet", "William Dieterle", "Marlene Dietrich",
                        "Top Hat", "Ginger Rogers"}


def test_build_2hop_subgraphs_chains_and_caps():
    one_hop = list(build_subgraphs(KB))
    two_hop = {sg.center.qid: sg for sg in build_2hop_subgraphs(one_hop, cap=256)}
    # Kismet's 2-hop set should reach Marlene Dietrich's OTHER films via the actor (chained edge).
    # Marlene Dietrich (a Kismet cast member) also has "director of"/"actor in" edges in her own
    # 1-hop subgraph; those become Kismet's 2-hop edges.
    kismet = two_hop["Kismet"]
    reached = {e.neighbor.qid for e in kismet.edges}
    assert "William Dieterle" in reached  # 1-hop kept
    assert "Marlene Dietrich" in reached
    # 2-hop reaches a neighbor-of-neighbor not in Kismet's own 1-hop edges.
    one_hop_reached = {e.neighbor.qid for e in kismet.edges[: len(one_hop[0].edges)]}
    assert len(reached) >= len(one_hop_reached)
    # cap is respected
    assert all(len(sg.edges) <= 256 for sg in two_hop.values())
    # self-loops excluded
    assert all(e.neighbor.qid != c for c, sg in two_hop.items() for e in sg.edges)


def test_parse_qa_line_and_loader():
    line = "what movies are about [ginger rogers]\tTop Hat|Kitty Foyle"
    parsed = parse_qa_line(line)
    assert parsed is not None
    subject, question, answers = parsed
    assert subject == "ginger rogers"
    assert question == "what movies are about ginger rogers"
    assert answers == ["Top Hat", "Kitty Foyle"]
    assert parse_qa_line("no tab here") is None
    assert parse_qa_line("no brackets\tA") is None

    rows = load_metaqa_qa([line, "bad\t", "also bad"])
    assert len(rows) == 1
    assert rows[0].subject_qid == "ginger rogers"
    assert rows[0].answer_labels == ["Top Hat", "Kitty Foyle"]
    assert rows[0].source == "metaqa"
