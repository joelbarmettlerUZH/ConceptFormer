"""Tests for the eval prompt-spec recording (reproducibility)."""

from conceptformer.eval.predictors import PROMPT_VERSION, SYSTEM_BASE, SYSTEM_RAG, prompt_spec


def test_prompt_spec_records_system_and_version():
    base, rag = prompt_spec("base"), prompt_spec("rag")
    assert base["system"] == SYSTEM_BASE
    assert rag["system"] == SYSTEM_RAG
    assert base["version"] == rag["version"] == PROMPT_VERSION
    # different prompts → different content hash (so reports are never silently conflated)
    assert base["system_sha"] != rag["system_sha"]
