"""Cache-key discipline and the dedup guarantee the free tier depends on."""

from __future__ import annotations

from filing.config import Settings
from filing.llm.cache import CallCache, make_key
from filing.llm.nvidia import NvidiaBackend

BASE = dict(backend="nvidia", model="m", kind="chat")
MSG = [{"role": "user", "content": "x"}]


def test_key_is_stable_across_dict_ordering():
    a = make_key(**BASE, payload={"temperature": 0.0, "messages": MSG})
    b = make_key(**BASE, payload={"messages": MSG, "temperature": 0.0})
    assert a == b


def test_key_changes_with_anything_that_changes_the_answer():
    base = make_key(**BASE, payload={"messages": MSG})
    variants = [
        make_key(**{**BASE, "model": "m2"}, payload={"messages": MSG}),
        make_key(**{**BASE, "kind": "embed"}, payload={"messages": MSG}),
        make_key(**{**BASE, "backend": "ollama"}, payload={"messages": MSG}),
        make_key(**BASE, payload={"messages": [{"role": "user", "content": "y"}]}),
    ]
    assert len(set(variants) | {base}) == len(variants) + 1


def test_roundtrip_and_clear(tmp_path):
    cache = CallCache(tmp_path / "c", enabled=True)
    assert cache.get("missing") is None
    cache.set("k", [1.0, 2.0])
    assert cache.get("k") == [1.0, 2.0]
    assert cache.hits == 1 and cache.misses == 1
    assert cache.clear() == 1
    assert cache.get("k") is None
    cache.close()


def test_disabled_cache_is_a_noop(tmp_path):
    cache = CallCache(tmp_path / "c", enabled=False)
    cache.set("k", 1)
    assert cache.get("k") is None
    assert len(cache) == 0


def test_repeated_prompt_issues_one_http_request(tmp_path):
    """The M0 definition of done, asserted offline.

    The network layer is stubbed so this is a statement about the cache, not
    about NVIDIA -- and so CI can run it with no API key.
    """
    cfg = Settings(
        nvidia_api_key="test-key",
        cache_dir=tmp_path / "llm",
        tracing_enabled=False,
    )
    backend = NvidiaBackend(cfg)

    def fake_call(model_id, role, alternates, payload):
        backend.http_calls += 1
        return "a 10-K is an annual report"

    backend._chat_call = fake_call  # noqa: SLF001 - deliberate seam

    prompt = [{"role": "user", "content": "What is a 10-K?"}]
    first = backend.chat(prompt, role="chat_fast")
    second = backend.chat(prompt, role="chat_fast")

    assert first == second
    assert backend.http_calls == 1
    assert backend.cache.hits == 1
    backend.close()


def test_embed_only_requests_the_texts_it_lacks(tmp_path):
    cfg = Settings(nvidia_api_key="test-key", cache_dir=tmp_path / "llm", tracing_enabled=False)
    backend = NvidiaBackend(cfg)
    requested: list[list[str]] = []

    def fake_embed(model_id, alternates, texts, input_type):
        requested.append(list(texts))
        backend.http_calls += 1
        return [[float(len(t))] * 4 for t in texts]

    backend._embed_call = fake_embed  # noqa: SLF001

    backend.embed(["alpha", "beta"])
    backend.embed(["beta", "gamma"])

    assert requested == [["alpha", "beta"], ["gamma"]]
    assert backend.http_calls == 2
    backend.close()
