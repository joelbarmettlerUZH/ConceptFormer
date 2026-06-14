"""Tests for the shared KV cache and the generation cache key."""

from conceptformer.cache import KVCache
from conceptformer.model.chat import generation_cache_key


def test_kvcache_roundtrip_and_persistence(tmp_path):
    path = tmp_path / "c.sqlite"
    cache = KVCache(path)
    assert cache.get("missing") is None
    cache.put("a", {"text": "hello"})
    cache.put_many({"b": {"text": "world"}})
    assert cache.get("a") == {"text": "hello"}
    cache.close()

    # a fresh handle on the same file sees the persisted values
    reopened = KVCache(path)
    assert reopened.get("a") == {"text": "hello"}
    assert reopened.get("b") == {"text": "world"}
    reopened.close()


def test_generation_cache_key_is_stable_and_discriminates():
    base = generation_cache_key("Qwen/Qwen3-0.6B", "sys", "user", 32)
    assert base == generation_cache_key("Qwen/Qwen3-0.6B", "sys", "user", 32)  # stable
    assert base != generation_cache_key("Qwen/Qwen3-0.6B", "sys", "user", 64)  # tokens matter
    assert base != generation_cache_key("Qwen/Qwen3-0.6B", "SYS", "user", 32)  # system matters
    assert base != generation_cache_key("Qwen/Qwen3-0.6B", "sys", "USER", 32)  # user matters
    assert base != generation_cache_key("other/model", "sys", "user", 32)  # model matters
    # decoding regime matters — greedy default must not collide with sampling
    m = "Qwen/Qwen3-0.6B"
    assert base == generation_cache_key(m, "sys", "user", 32, decoding="greedy")
    assert base != generation_cache_key(m, "sys", "user", 32, decoding="sample-t0.8")
