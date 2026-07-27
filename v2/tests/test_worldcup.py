"""Offline tests for the WorldCup2014 (WC-P) cross-graph loader (data/worldcup.py)."""

from conceptformer.data.worldcup import (
    build_subgraphs,
    load_worldcup_qa,
    parse_kb_line,
    parse_qa_line,
)

KB = [
    "JOAO_MOUTINHO\tplays_in_club\tAS_Monaco",
    "AS_Monaco\tplays_in_club_inverse\tJOAO_MOUTINHO",
    "JOAO_MOUTINHO\tplays_for_country\tPortugal",
    "malformed line",
]

# QA columns: question, answer, path, answer_set, neighborhood
QA_ENTITY = ("which club does Alan_PULIDO play for ?\tTigres_UANL\t"
             "Alan_PULIDO#plays_in_club#Tigres_UANL#<end>#Tigres_UANL\tTigres_UANL/\tnbhd")
QA_NUMERIC = ("how old is Alan_PULIDO ?\t23\tAlan_PULIDO#is_aged#23#<end>#23\t23/\tnbhd")


def test_parse_kb_line():
    assert parse_kb_line(KB[0]) == ("JOAO_MOUTINHO", "plays_in_club", "AS_Monaco")
    assert parse_kb_line("malformed line") is None
    assert parse_kb_line("a\tb") is None


def test_build_subgraphs_uses_native_inverse_edges_and_pretty_labels():
    sgs = {sg.center.qid: sg for sg in build_subgraphs(KB)}
    # Player side: outgoing edges, readable relation + neighbor labels.
    player = sgs["JOAO_MOUTINHO"]
    assert player.center.label == "JOAO MOUTINHO"
    labels = {(e.property_label, e.neighbor.label) for e in player.edges}
    assert ("plays in club", "AS Monaco") in labels
    assert ("plays for country", "Portugal") in labels
    # Club side: the KB's own inverse edge becomes the club's 1-hop neighborhood (no synthesis).
    club = sgs["AS_Monaco"]
    assert [(e.property_label, e.neighbor.qid) for e in club.edges] == [
        ("has club player", "JOAO_MOUTINHO")
    ]


def test_parse_qa_line_entity_and_numeric():
    subj, rel, q, ans = parse_qa_line(QA_ENTITY)
    assert subj == "Alan_PULIDO"
    assert rel == "plays_in_club"
    assert q == "which club does Alan PULIDO play for ?"  # underscores prettified
    assert ans == ["Tigres_UANL"]
    assert parse_qa_line("no tabs at all") is None


def test_load_drops_numeric_relations_and_keeps_surface_aliases():
    rows = load_worldcup_qa([QA_ENTITY, QA_NUMERIC], drop_numeric=True)
    assert len(rows) == 1  # numeric (is_aged) dropped
    r = rows[0]
    assert r.source == "worldcup"
    assert r.subject_qid == "Alan_PULIDO"
    # both prettified and raw answer forms retained for word-boundary scoring
    assert "Tigres UANL" in r.answer_labels
    assert "Tigres_UANL" in r.answer_labels
    # keeping numerics is opt-in
    assert len(load_worldcup_qa([QA_ENTITY, QA_NUMERIC], drop_numeric=False)) == 2
