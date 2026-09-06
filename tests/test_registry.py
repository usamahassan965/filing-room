"""Invariants on the model registry.

The registry only earns its keep if it is complete and unambiguous. These tests
fail the moment someone hardcodes a role or forgets an endpoint.
"""

from __future__ import annotations

import pytest

from filing.config import MODEL_REGISTRY, model_for

ROLES = ("chat", "chat_fast", "embed", "rerank")


@pytest.mark.parametrize("backend", sorted(MODEL_REGISTRY))
@pytest.mark.parametrize("role", ROLES)
def test_every_backend_covers_every_role(backend, role):
    spec = model_for(role, backend)
    assert spec.id


@pytest.mark.parametrize("backend", sorted(MODEL_REGISTRY))
def test_no_duplicate_candidates(backend):
    for role, spec in MODEL_REGISTRY[backend].items():
        assert len(set(spec.candidates)) == len(spec.candidates), f"{backend}/{role}"
        assert spec.candidates[0] == spec.id


def test_every_nvidia_rerank_candidate_has_an_endpoint():
    """Reranking URLs are per-model, so an alternate without one is a landmine."""
    spec = MODEL_REGISTRY["nvidia"]["rerank"]
    for candidate in spec.candidates:
        assert spec.endpoint_for(candidate), candidate


def test_unknown_role_names_the_known_ones():
    with pytest.raises(KeyError) as exc:
        model_for("summariser", "nvidia")
    assert "chat" in str(exc.value)


@pytest.mark.parametrize("backend", sorted(MODEL_REGISTRY))
def test_declared_rate_limits_are_sane(backend):
    """A zero or negative rpm would deadlock the limiter rather than slow it."""
    for role, spec in MODEL_REGISTRY[backend].items():
        if spec.rpm is not None:
            assert spec.rpm > 0, f"{backend}/{role} rpm={spec.rpm}"


@pytest.mark.parametrize("backend", sorted(MODEL_REGISTRY))
def test_local_models_are_not_given_endpoints(backend):
    """`local` and `endpoint` are contradictory claims about where a model runs."""
    for role, spec in MODEL_REGISTRY[backend].items():
        if spec.local:
            assert spec.endpoint is None and not spec.endpoints, f"{backend}/{role}"


def test_embed_specs_declare_their_width():
    """`filing smoke` asserts the vector width, so an undeclared dim is untestable."""
    for backend in MODEL_REGISTRY:
        assert MODEL_REGISTRY[backend]["embed"].dim, backend
