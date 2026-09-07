"""Invariants on the model registry.

The registry only earns its keep if it is complete and unambiguous. These tests
fail the moment someone hardcodes a role or forgets an endpoint.
"""

from __future__ import annotations

import pytest

from filing.config import MODEL_REGISTRY, Settings, model_for

ROLES = ("chat", "chat_fast", "embed", "rerank")

# The one backend that is deliberately not a chat backend. It exists because
# Gemini's free embedding tier serves 1,000 documents a day and this corpus has
# 32,218 chunks, so the vector space runs on this machine; generation still does
# not. A missing chat role here is the design, and ``LocalBackend.chat`` says so
# rather than returning something.
RETRIEVAL_ONLY = frozenset({"local"})


@pytest.mark.parametrize("backend", sorted(MODEL_REGISTRY))
@pytest.mark.parametrize("role", ("embed", "rerank"))
def test_every_backend_can_retrieve(backend, role):
    assert model_for(role, backend).id


@pytest.mark.parametrize("backend", sorted(set(MODEL_REGISTRY) - RETRIEVAL_ONLY))
@pytest.mark.parametrize("role", ("chat", "chat_fast"))
def test_every_chat_backend_covers_both_chat_roles(backend, role):
    assert model_for(role, backend).id


@pytest.mark.parametrize("backend", sorted(RETRIEVAL_ONLY))
def test_a_retrieval_only_backend_names_the_roles_it_does_have(backend):
    """Asking it for chat is a configuration error, and the error has to say so."""
    with pytest.raises(KeyError, match="embed, rerank"):
        model_for("chat", backend)


@pytest.mark.parametrize("backend", sorted(MODEL_REGISTRY))
def test_no_duplicate_candidates(backend):
    for role, spec in MODEL_REGISTRY[backend].items():
        assert len(set(spec.candidates)) == len(spec.candidates), f"{backend}/{role}"
        assert spec.candidates[0] == spec.id



def test_unknown_role_names_the_known_ones():
    with pytest.raises(KeyError) as exc:
        model_for("summariser", "gemini")
    assert "chat" in str(exc.value)


@pytest.mark.parametrize("backend", sorted(MODEL_REGISTRY))
def test_declared_rate_limits_are_sane(backend):
    """A zero or negative rpm would deadlock the limiter rather than slow it."""
    for role, spec in MODEL_REGISTRY[backend].items():
        if spec.rpm is not None:
            assert spec.rpm > 0, f"{backend}/{role} rpm={spec.rpm}"



def test_embed_specs_declare_their_width():
    """`filing smoke` asserts the vector width, so an undeclared dim is untestable."""
    for backend in MODEL_REGISTRY:
        assert MODEL_REGISTRY[backend]["embed"].dim, backend


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------

SECRET_FIELDS = ("gemini_api_key",)


@pytest.mark.parametrize("field", SECRET_FIELDS)
def test_a_key_is_never_rendered_by_accident(field):
    """The key must survive being printed by code that never meant to print it.

    This is not hypothetical. A pytest failure once rendered the settings
    object and put a live Gemini key in the run log -- not through a logging
    call, but through the default repr of an object that happened to be an
    argument to a failing test. Every path below is one something else calls
    on your behalf: repr in a traceback, str in an f-string, format in a log
    line, and the span attributes tracing writes. Only .get_secret_value()
    returns the real thing, which makes reading the key a visible act.
    """
    marker = "sk-live-DO-NOT-LEAK-4d3f2a"
    cfg = Settings(**{field: marker})
    rendered = [repr(cfg), str(cfg), f"{cfg}", format(cfg), repr(getattr(cfg, field))]
    for text in rendered:
        assert marker not in text
    assert getattr(cfg, field).get_secret_value() == marker


@pytest.mark.parametrize("field", SECRET_FIELDS)
def test_an_absent_key_still_reads_as_absent(field):
    """The backends guard on `if not cfg.<key>`, and SecretStr must not break it.

    A secret wrapper that were always truthy would turn "no credentials
    configured" into an authenticated request carrying an empty key -- a 401
    from the provider instead of the local error that names what to set.
    """
    assert not getattr(Settings(**{field: ""}), field)
    assert getattr(Settings(**{field: "x"}), field)
