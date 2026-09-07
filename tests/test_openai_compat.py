"""The bake-off backend: credentials, provider quirks, and the citation glyph.

No test here opens a socket. What is being asserted is the wiring -- that a
provider needing no key is distinguishable from one whose key is missing, that
the Cloudflare workaround is actually applied, and that the citation contract
survives a model that writes its brackets in a different alphabet.
"""

from __future__ import annotations

import httpx
import pytest

from filing.config import MODEL_REGISTRY, Settings, model_for
from filing.eval.runner import parse_citations
from filing.llm.errors import MissingCredentials
from filing.llm.openai_compat import (
    _BROWSER_UA,
    OpenAICompatBackend,
    _strip_authorization,
)


class _Chunk:
    def __init__(self, chunk_id: str) -> None:
        self.chunk_id = chunk_id


CHUNKS = [_Chunk("a"), _Chunk("b"), _Chunk("c")]


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------


def test_a_provider_that_needs_no_key_is_not_a_provider_missing_one():
    """OVHcloud answers anonymously, and that is the point of it being here.

    A blank secret has to mean two different things depending on the provider,
    so the third element of the tuple carries the distinction rather than the
    emptiness of the second.
    """
    cfg = Settings(cohere_api_key="", groq_api_key="")
    _, secret, where = cfg.provider_credentials("ovh")
    assert secret == "" and where == ""

    _, secret, where = cfg.provider_credentials("cohere")
    assert secret == "" and where.startswith("https://")


def test_an_unknown_provider_names_the_known_ones():
    with pytest.raises(KeyError) as exc:
        Settings().provider_credentials("openrouter")
    assert "cohere" in str(exc.value)


def test_a_missing_key_says_where_to_get_one(monkeypatch):
    """The failure a new clone hits first, so it has to be a signpost."""
    cfg = Settings(groq_api_key="", tracing_enabled=False)
    with pytest.raises(MissingCredentials) as exc:
        OpenAICompatBackend("groq", cfg)
    assert "console.groq.com" in str(exc.value)


def test_ovh_constructs_without_any_credential(tmp_path):
    """Constructed, not called -- no socket is opened by this test."""
    cfg = Settings(cache_dir=tmp_path / "llm", tracing_enabled=False)
    backend = OpenAICompatBackend("ovh", cfg)
    assert backend.name == "ovh"
    assert backend.usage().http_calls == 0
    backend.close()


# --------------------------------------------------------------------------
# provider quirks
# --------------------------------------------------------------------------


def test_the_cloudflare_user_agent_reaches_the_client(tmp_path):
    """Groq answers a default Python UA with 403 before it ever checks the key.

    The failure mode this guards is not a broken call -- it is a call that fails
    with an authentication-shaped error for a reason that has nothing to do with
    the credential, which cost an afternoon to diagnose once already.
    """
    cfg = Settings(groq_api_key="test-key", cache_dir=tmp_path / "llm", tracing_enabled=False)
    backend = OpenAICompatBackend("groq", cfg)
    assert "Mozilla/5.0" in _BROWSER_UA
    assert backend._client.default_headers["User-Agent"] == _BROWSER_UA  # noqa: SLF001
    backend.close()


def test_the_sdk_does_not_retry_behind_the_limiter(tmp_path):
    """Retries must pass the limiter, so the SDK must not own them.

    On an 8,000-token-per-minute tier an unbudgeted SDK retry is not a rounding
    error, it is the next question's whole allowance.
    """
    cfg = Settings(cohere_api_key="test-key", cache_dir=tmp_path / "llm", tracing_enabled=False)
    backend = OpenAICompatBackend("cohere", cfg)
    assert backend._client.max_retries == 0  # noqa: SLF001
    backend.close()


def test_groq_declares_no_embedding_model():
    """Groq serves none, and the registry says so by omission rather than by zero."""
    assert "embed" not in MODEL_REGISTRY["groq"]
    with pytest.raises(KeyError):
        model_for("embed", "groq")


@pytest.mark.parametrize("provider", ["cohere", "groq", "ovh"])
def test_every_bake_off_provider_reranks_locally(provider):
    """The configs change the generator. A hosted reranker would change two things."""
    spec = model_for("rerank", provider)
    assert spec.local is True


@pytest.mark.parametrize("provider", ["cohere", "groq", "ovh"])
def test_a_throttled_provider_declares_its_ceiling(provider):
    """Every rpm here came from a live x-ratelimit header, not from a docs page."""
    assert model_for("chat", provider).rpm


# --------------------------------------------------------------------------
# the citation glyph
# --------------------------------------------------------------------------


def test_fullwidth_brackets_are_still_citations():
    """gpt-oss-120b cites with U+3010/U+3011, and scored 0% until this landed.

    The system prompt asks for "bracketed numbers" and never promises a
    codepoint, so a model that obeys in CJK lenticular brackets has obeyed. The
    metric was wrong, not the model.
    """
    answer = "The deficit was $16.5 billion 【1】 and R&D was $16.0 billion 【3】."
    assert parse_citations(answer, CHUNKS) == ("a", "c")


def test_ascii_citations_are_unaffected():
    assert parse_citations("see [2] then [1]", CHUNKS) == ("b", "a")


def test_a_marker_past_the_last_excerpt_is_still_dropped():
    """Normalising the glyph must not smuggle in an unresolvable id."""
    assert parse_citations("【99】 and 【2】", CHUNKS) == ("b",)


# --------------------------------------------------------------------------
# the anonymous provider
# --------------------------------------------------------------------------


def test_an_anonymous_request_carries_no_authorization_header():
    """An absent header and an empty one are not the same thing.

    `Bearer no-key-required` made OVHcloud answer 403 "authentication failed"
    on all 150 questions of a run -- it stopped reading the request as
    anonymous and started reading it as a failed credential. The header has to
    be gone, not blank.
    """
    request = httpx.Request(
        "POST",
        "https://example.invalid/v1/chat/completions",
        headers={"Authorization": "Bearer anonymous", "User-Agent": _BROWSER_UA},
    )
    _strip_authorization(request)
    assert "authorization" not in request.headers
    assert request.headers["User-Agent"] == _BROWSER_UA


def test_only_the_keyless_provider_strips_the_header(tmp_path):
    """Cohere and Groq must keep sending theirs -- the hook is not a blanket."""
    cfg = Settings(
        cohere_api_key="test-key",
        groq_api_key="test-key",
        cache_dir=tmp_path / "llm",
        tracing_enabled=False,
    )

    def hooks(provider):
        client = OpenAICompatBackend(provider, cfg)._client._client  # noqa: SLF001
        return client.event_hooks["request"]

    assert _strip_authorization in hooks("ovh")
    assert _strip_authorization not in hooks("cohere")
    assert _strip_authorization not in hooks("groq")
