"""Offline tests for token-efficiency accounting (no model load — a stub tokenizer)."""

from conceptformer.eval.tokens import (
    TokenRecord,
    concept_token_records,
    measure_text_prompt_tokens,
    token_report,
)


class _StubModel:
    """Whitespace tokenizer; prompt_length adds a fixed chat-template overhead."""

    OVERHEAD = 5

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def prompt_length(self, system: str, user: str) -> int:
        return self.OVERHEAD + self.count_tokens(system) + self.count_tokens(user)


def test_base_condition_has_zero_knowledge_tokens():
    base = [("sys", "Question: who")]
    recs = measure_text_prompt_tokens(_StubModel(), base, base)  # active == base
    assert recs[0].knowledge_tokens == 0
    assert recs[0].input_tokens == 5 + 1 + 2  # overhead + "sys" + "Question: who"


def test_rag_knowledge_tokens_isolate_the_facts_payload():
    model = _StubModel()
    base = [("sys", "Question: who")]
    active = [("sys use the facts", "a b c d facts Question: who")]  # +5 facts words in user
    recs = measure_text_prompt_tokens(model, active, base)
    # knowledge = active_user words - base_user words = 7 - 2 = 5 (system delta excluded)
    assert recs[0].knowledge_tokens == 5
    # input_tokens uses the full rendered length (system delta DOES count toward input cost)
    assert recs[0].input_tokens == model.prompt_length(*active[0])
    assert recs[0].uncapped_knowledge_tokens is None  # not provided → omitted


def test_uncapped_knowledge_tokens_counts_full_neighborhood():
    model = _StubModel()
    base = [("sys", "Question: who")]
    active = [("sys", "two facts Question: who")]  # budgeted to 2 fact words
    uncapped = ["a b c d e f"]  # full neighborhood = 6 words
    recs = measure_text_prompt_tokens(model, active, base, uncapped)
    assert recs[0].knowledge_tokens == 2  # capped (what RAG actually fed)
    assert recs[0].uncapped_knowledge_tokens == 6  # full text cost RAG would pay un-budgeted
    # base example (no subgraph) → None passes through
    recs_none = measure_text_prompt_tokens(model, base, base, [None])
    assert recs_none[0].uncapped_knowledge_tokens is None


def test_concept_token_records_are_constant_k():
    base = [("sys", "Question: a b c"), ("sys", "Question: a b c d e f g h")]
    recs = concept_token_records(_StubModel(), base, k=4, uncapped_facts=["a b c", "a b c d e"])
    assert [r.knowledge_tokens for r in recs] == [4, 4]  # constant regardless of prompt size
    # input = base prompt length + k
    assert recs[0].input_tokens == _StubModel().prompt_length(*base[0]) + 4
    # the full-text RAG cost CF replaces — makes the k-vs-uncapped saving explicit
    assert [r.uncapped_knowledge_tokens for r in recs] == [3, 5]


def test_token_report_aggregates():
    recs = [TokenRecord(input_tokens=t, knowledge_tokens=t - 10) for t in (20, 30, 40, 50)]
    rep = token_report(recs)
    assert rep["input_tokens"]["mean"] == 35.0
    assert rep["input_tokens"]["max"] == 50
    assert rep["input_tokens"]["total"] == 140
    assert rep["knowledge_tokens"]["mean"] == 25.0
    assert "uncapped_knowledge_tokens" not in rep  # absent when no record carries it


def test_token_report_includes_uncapped_when_present():
    recs = [TokenRecord(input_tokens=10, knowledge_tokens=5, uncapped_knowledge_tokens=200)]
    rep = token_report(recs)
    assert rep["uncapped_knowledge_tokens"]["mean"] == 200.0


def test_token_report_empty():
    assert token_report([]) == {"input_tokens": {"n": 0}, "knowledge_tokens": {"n": 0}}
