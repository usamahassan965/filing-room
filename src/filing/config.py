"""Settings and the model registry.

Two rules this module exists to enforce:

1. Model IDs live here and nowhere else, so a 404 is a one-line change in this
   file and never a grep across the codebase. Not a hypothetical: every Gemini
   2.x chat ID in this registry was retired out from under a working key
   ("no longer available to new users") while embeddings kept serving. Two
   lines here, no code touched.
2. Nothing reads ``os.environ`` directly. Everything goes through ``settings()``,
   so a config value can be overridden in a test without monkeypatching the world.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Backend = Literal["gemini", "nvidia", "ollama"]
Role = Literal["chat", "chat_fast", "embed", "rerank"]


@dataclass(frozen=True)
class ModelSpec:
    """One model, one place.

    ``alternates`` are not a fallback chain the code walks silently -- silent
    fallback would let a quality regression hide behind a working smoke test.
    They are the candidates ``filing probe`` checks, so when an ID dies you are
    told which replacement is live and you edit ``id`` by hand.
    """

    id: str
    alternates: tuple[str, ...] = ()
    dim: int | None = None
    endpoint: str | None = None  # set only when the call is not OpenAI-shaped
    # Reranking URLs are per-model, so an alternate needs its own.
    endpoints: Mapping[str, str] = field(default_factory=dict)
    asymmetric: bool = False  # needs input_type=query|passage
    # Rate limits are per-model, not per-provider: on Gemini's free tier the
    # chat model allows ~10 rpm while embeddings allow ~100. One shared limiter
    # would throttle indexing to the speed of the slowest model in the registry.
    rpm: int | None = None
    local: bool = False  # runs on this machine; no key, no quota, no network
    note: str = ""

    @property
    def candidates(self) -> tuple[str, ...]:
        return (self.id, *self.alternates)

    def endpoint_for(self, model_id: str | None = None) -> str | None:
        return self.endpoints.get(model_id or self.id, self.endpoint)


# NVIDIA's reranker is not an OpenAI-shaped call -- it has its own host and its
# own request body -- so its URL travels with the spec rather than being derived
# from the base URL.
_NIM_RERANK_URLS = {
    "nvidia/llama-3.2-nv-rerankqa-1b-v2": (
        "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
    ),
    "nvidia/nv-rerankqa-mistral-4b-v3": ("https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"),
}

MODEL_REGISTRY: dict[Backend, dict[str, ModelSpec]] = {
    # Gemini free tier. The rpm figures are the documented free-tier limits at
    # the time of writing; they move, so they are overridable per-model here and
    # globally via RATE_LIMIT_RPM. Daily caps (~250 requests for the big chat
    # model) bind harder than rpm on an evaluation sweep -- budget accordingly.
    "gemini": {
        # Verified live against a fresh key on 2026-09-05. The 2.x IDs that were
        # here first are still returned by ListModels but 404 for new keys with
        # "no longer available to new users" -- so a listing is not a probe, and
        # `filing probe` calls each ID for real rather than trusting the catalog.
        # No *-latest alias as a primary: those float under you, and the one time
        # it mattered `gemini-flash-latest` answered 503 while the pinned ID was
        # fine. Aliases are fine as fallbacks, where floating is the point.
        # No Pro anywhere: every Pro ID answers 429 on the free tier.
        "chat": ModelSpec(
            id="gemini-3.5-flash",
            alternates=("gemini-3.6-flash", "gemini-3-flash-preview"),
            rpm=10,
            note="synthesis and grading -- the calls whose quality shows up in eval",
        ),
        "chat_fast": ModelSpec(
            id="gemini-3.5-flash-lite",
            alternates=("gemini-3.1-flash-lite", "gemini-flash-lite-latest"),
            rpm=15,
            note="routing, planning, cheap classification",
        ),
        "embed": ModelSpec(
            id="gemini-embedding-001",
            alternates=("gemini-embedding-2", "gemini-embedding-2-preview"),
            dim=1536,
            asymmetric=True,
            rpm=100,
            note="taskType is Gemini's spelling of input_type; native REST, not the OpenAI shim",
        ),
        # Gemini serves no reranker, so this one runs here. That is an upgrade
        # over the cosine stand-in, not a compromise: a cross-encoder reads the
        # query and the passage together, which is the whole point of reranking.
        "rerank": ModelSpec(
            id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            alternates=(
                "cross-encoder/ms-marco-MiniLM-L-12-v2",
                "BAAI/bge-reranker-base",
            ),
            local=True,
            note="local cross-encoder; no key, no quota, no rate limit",
        ),
    },
    "nvidia": {
        "chat": ModelSpec(
            id="meta/llama-3.3-70b-instruct",
            alternates=(
                "nvidia/llama-3.3-nemotron-super-49b-v1.5",
                "meta/llama-3.1-70b-instruct",
            ),
            note="synthesis and grading -- the calls whose quality shows up in eval",
        ),
        "chat_fast": ModelSpec(
            id="meta/llama-3.1-8b-instruct",
            alternates=("nvidia/nemotron-mini-4b-instruct",),
            note="routing, planning, cheap classification",
        ),
        "embed": ModelSpec(
            id="nvidia/llama-3.2-nv-embedqa-1b-v2",
            alternates=(
                "nvidia/llama-nemotron-embed-1b-v2",  # the rename target
                "nvidia/nv-embedqa-e5-v5",  # 1024-dim, different family
            ),
            dim=2048,
            asymmetric=True,
            note="asymmetric: queries and passages are embedded differently",
        ),
        "rerank": ModelSpec(
            id="nvidia/llama-3.2-nv-rerankqa-1b-v2",
            alternates=("nvidia/nv-rerankqa-mistral-4b-v3",),
            endpoint=_NIM_RERANK_URLS["nvidia/llama-3.2-nv-rerankqa-1b-v2"],
            endpoints=_NIM_RERANK_URLS,
            note="cross-encoder; the single biggest retrieval quality lever",
        ),
    },
    "ollama": {
        "chat": ModelSpec(id="llama3.1:8b", note="local escape hatch"),
        "chat_fast": ModelSpec(id="llama3.2:3b"),
        "embed": ModelSpec(id="nomic-embed-text", dim=768),
        # Ollama serves no cross-encoder. The Ollama backend reranks by
        # embedding cosine, which is a degraded stand-in and says so at runtime.
        "rerank": ModelSpec(
            id="nomic-embed-text", dim=768, note="cosine stand-in, not a cross-encoder"
        ),
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- providers ---
    llm_backend: Backend = "gemini"

    gemini_api_key: str = ""
    # Two base URLs on purpose. Chat goes through Google's OpenAI-compatible
    # shim so the SDK code is shared with NIM; embeddings go through the native
    # API, because taskType (query vs passage) is not an OpenAI parameter and
    # the shim drops what it does not recognise -- silently, which would leave
    # asymmetric retrieval quietly broken.
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_openai_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    ollama_base_url: str = "http://localhost:11434"

    # --- local models ---
    rerank_device: str = "cpu"
    rerank_batch_size: int = 16

    # --- budget ---
    # Fallback only: a model whose registry entry sets ``rpm`` uses that
    # instead. Kept below the provider's real ceiling so tenacity's retries have
    # somewhere to go without breaching it.
    rate_limit_rpm: int = 35
    request_timeout_s: float = 90.0
    max_attempts: int = 5

    # --- cache ---
    cache_enabled: bool = True
    cache_dir: Path = PROJECT_ROOT / ".cache" / "llm"

    # --- observability ---
    tracing_enabled: bool = True
    phoenix_endpoint: str = "http://localhost:6006"
    phoenix_project: str = "filing-room"

    # --- data sources ---
    # EDGAR rejects an anonymous client with 403, so this is not optional. SEC
    # asks for a real contact address; a fake one is how a project gets its IP
    # blocked rather than throttled.
    sec_user_agent: str = ""
    sec_base_url: str = "https://www.sec.gov"
    sec_data_url: str = "https://data.sec.gov"
    # SEC's fair-access ceiling is 10 requests per *second*, not per minute --
    # a different shape of limit from every model provider here, and the one
    # place in this project where the window is a second wide. Kept under the
    # ceiling because EDGAR answers a breach with a timed IP block, and there is
    # no retry policy that recovers from being blocked.
    sec_rps: int = 8

    # --- corpus ---
    data_dir: Path = PROJECT_ROOT / "data"
    # Committed, so it lives outside data/ -- which is gitignored, because a
    # corpus is reproducible and a declaration of scope is not.
    universe_path: Path = PROJECT_ROOT / "universe.yaml"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def manifest_path(self) -> Path:
        """The manifest is the source of truth about the corpus, not the disk.

        Downloads resume from this file. Scanning the filesystem instead would
        make a half-written document look like a finished one.
        """
        return self.data_dir / "manifest.duckdb"

    @property
    def facts_path(self) -> Path:
        """The structured store: derived from the manifest, and disposable.

        A separate file from the manifest on purpose. Losing the manifest costs
        a 1.12 GB re-download from a rate-limited host; losing this costs a
        minute of parsing the JSON already on disk. Keeping them apart means a
        rebuild can drop everything it owns without ever holding a write handle
        on the irreplaceable one.
        """
        return self.data_dir / "facts.duckdb"

    @property
    def registry(self) -> dict[str, ModelSpec]:
        return MODEL_REGISTRY[self.llm_backend]


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()


def model_for(role: str, backend: Backend | None = None) -> ModelSpec:
    """The only supported way to learn a model ID."""
    backend = backend or settings().llm_backend
    try:
        return MODEL_REGISTRY[backend][role]
    except KeyError as exc:  # pragma: no cover - programmer error
        known = ", ".join(sorted(MODEL_REGISTRY[backend]))
        raise KeyError(
            f"no model registered for role {role!r} on {backend!r}; have: {known}"
        ) from exc
