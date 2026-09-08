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

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Backend = Literal["gemini", "cohere", "groq", "ovh", "ollama", "local"]
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
    asymmetric: bool = False  # needs input_type=query|passage
    # Rate limits are per-model, not per-provider: on Gemini's free tier the
    # chat model allows 5 rpm while embeddings allow ~100. One shared limiter
    # would throttle indexing to the speed of the slowest model in the registry.
    rpm: int | None = None
    local: bool = False  # runs on this machine; no key, no quota, no network
    note: str = ""

    @property
    def candidates(self) -> tuple[str, ...]:
        return (self.id, *self.alternates)


MODEL_REGISTRY: dict[Backend, dict[str, ModelSpec]] = {
    # Gemini free tier. The rpm figures are what the endpoint actually enforces,
    # which is not always what the docs say -- the chat model sat at 10 here for
    # a month on the documented figure and 429s citing 5. They move, so they are
    # overridable per-model here and globally via RATE_LIMIT_RPM. Daily caps
    # (~250 requests for the big chat model) bind harder than rpm on an
    # evaluation sweep -- budget accordingly.
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
            # 5, not the documented 10: a live key 429s with "limit: 5" well
            # before the tenth call in a minute. A limiter set above the real
            # ceiling is worse than none -- it converts a wait into an error.
            rpm=5,
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
    # ---------------------------------------------------------------- bake-off
    # Three providers whose free tiers were probed live rather than read off a
    # listicle, all reachable through filing.llm.openai_compat. Every rpm below
    # was taken from the provider's own x-ratelimit headers on a real call.
    #
    # The rerank entry in each is the same local cross-encoder the Gemini
    # registry uses. That is not laziness: these configs exist to change the
    # generator and hold retrieval fixed, so a provider's own reranker would
    # confound exactly the comparison they are for.
    "cohere": {
        "chat": ModelSpec(
            id="command-a-03-2025",
            alternates=("command-a-plus-05-2026", "command-r-plus-08-2024"),
            rpm=18,
            note="trial key: 20 rpm and 1,000 calls a MONTH -- calls, not tokens",
        ),
        "chat_fast": ModelSpec(id="command-r7b-12-2024", alternates=("command-r-08-2024",), rpm=18),
        "embed": ModelSpec(id="embed-v4.0", alternates=("embed-english-v3.0",), dim=1536, rpm=18),
        "rerank": ModelSpec(
            id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            local=True,
            note="local cross-encoder, so the generator is the only variable",
        ),
    },
    # Groq's headers are necessary and not sufficient. They advertise 8,000
    # tokens per MINUTE and 1,000 requests per day, and say nothing at all about
    # the cap that actually stops a run: 200,000 tokens per DAY, per model,
    # which appears in no header and only in the body of the 429 that ends you.
    # At ~4.2k tokens a question that is 47 questions a day, so a 150-question
    # baseline is a three-day run on the free tier. rpm=2 comes from the TPM
    # header and is still right; it is just not the limit you hit second.
    "groq": {
        "chat": ModelSpec(
            id="openai/gpt-oss-120b",
            alternates=("qwen/qwen3.8-27b", "openai/gpt-oss-20b"),
            rpm=2,
            note="200,000 tokens/DAY is the cap that ends a run; no header mentions it",
        ),
        "chat_fast": ModelSpec(id="qwen/qwen3.6-27b", alternates=("openai/gpt-oss-20b",), rpm=2),
        # No embed entry, deliberately. Groq's catalogue is chat, speech and
        # safety classifiers -- there is no embedding model to name, and a
        # placeholder with dim=0 would be a registry that lies. model_for()
        # raising is the correct answer to "embed with Groq".
        "rerank": ModelSpec(id="cross-encoder/ms-marco-MiniLM-L-6-v2", local=True),
    },
    # No key, no account, no country list to be absent from -- which is the
    # entire reason it is here, after build.nvidia.com turned out to be a wall.
    # The price is 2 requests per minute per IP, shared with everyone else on
    # this address, so a full run is over an hour and the limiter is not
    # optional.
    "ovh": {
        "chat": ModelSpec(
            id="Qwen3.5-397B-A17B",
            alternates=("gpt-oss-120b", "Meta-Llama-3_3-70B-Instruct", "Qwen3-32B"),
            rpm=2,
            note="anonymous; 2 rpm per IP is the whole quota",
        ),
        "chat_fast": ModelSpec(id="Qwen3.5-9B", alternates=("Mistral-7B-Instruct-v0.3",), rpm=2),
        "embed": ModelSpec(id="bge-m3", alternates=("Qwen3-Embedding-8B",), dim=1024, rpm=2),
        "rerank": ModelSpec(id="cross-encoder/ms-marco-MiniLM-L-6-v2", local=True),
    },
    # Sized to the machine, not to the leaderboard. llama3.1:8b at q4 wants
    # ~6 GB resident and this box has 15.8 GB total with ~4.8 GB actually free,
    # so it would swap -- and a generator that swaps does not produce a slow
    # eval, it produces a timed-out one. Both entries below are ~2 GB, which is
    # why there are two of them: they are the bake-off, run over the same 150
    # questions by the baseline-local configs rather than argued about.
    "ollama": {
        "chat": ModelSpec(
            id="qwen2.5:3b",
            alternates=("llama3.2:3b", "llama3.1:8b"),
            note="local escape hatch; 8b is listed but does not fit this machine",
        ),
        "chat_fast": ModelSpec(id="llama3.2:3b"),
        "embed": ModelSpec(id="nomic-embed-text", dim=768),
        # Ollama serves no cross-encoder. The Ollama backend reranks by
        # embedding cosine, which is a degraded stand-in and says so at runtime.
        "rerank": ModelSpec(
            id="nomic-embed-text", dim=768, note="cosine stand-in, not a cross-encoder"
        ),
    },
    # Retrieval only, and on purpose -- see filing.llm.local. Gemini's free
    # embedding tier is 1,000 documents a day and this corpus is 32,218
    # narrative chunks, so the vector space is built here and generation stays
    # hosted. No chat entry: the backend raises rather than pretend a 33M
    # parameter encoder can answer a question.
    "local": {
        "embed": ModelSpec(
            id="BAAI/bge-small-en-v1.5",
            alternates=("BAAI/bge-base-en-v1.5", "intfloat/e5-small-v2"),
            dim=384,
            asymmetric=True,
            local=True,
            note="bi-encoder; query side takes BGE's instruction prefix",
        ),
        "rerank": ModelSpec(
            id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            alternates=("cross-encoder/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-base"),
            local=True,
            note="the same cross-encoder the hosted backends borrow",
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
    # Two backends, because generation and retrieval buy different things.
    # ``llm_backend`` answers questions: one hosted call per answer, where a
    # frontier model is worth the quota. ``embed_backend`` owns the vector
    # space: one call per *chunk*, 32,218 of them, which no free tier serves --
    # Gemini's caps embeddings at 1,000 documents a day. Splitting them is what
    # lets the index be rebuilt on a whim; see docs/retrieval.md.
    llm_backend: Backend = "gemini"
    embed_backend: Backend = "local"

    # SecretStr, not str, and not merely ``Field(repr=False)``. The key leaked
    # once already -- through the settings repr in a pytest failure, printed by
    # nothing that meant to print it. repr=False would close that one path;
    # this closes str(), f-strings, logging and span attributes too, because
    # the value is only readable through .get_secret_value(). Empty still reads
    # as falsey, so the "no credentials" guards below are unchanged.
    gemini_api_key: SecretStr = SecretStr("")
    # Two base URLs on purpose. Chat goes through Google's OpenAI-compatible
    # shim so the SDK code is shared with NIM; embeddings go through the native
    # API, because taskType (query vs passage) is not an OpenAI parameter and
    # the shim drops what it does not recognise -- silently, which would leave
    # asymmetric retrieval quietly broken.
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_openai_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    ollama_base_url: str = "http://localhost:11434"

    # --- bake-off providers (see filing.llm.openai_compat) ---
    # Cohere speaks OpenAI at a /compatibility/v1 path rather than at the root,
    # which is the only structural difference between the three.
    cohere_api_key: SecretStr = SecretStr("")
    cohere_base_url: str = "https://api.cohere.ai/compatibility/v1"
    groq_api_key: SecretStr = SecretStr("")
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # No key field: OVHcloud answers unauthenticated requests, which is the
    # property that put it on this list.
    ovh_base_url: str = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"

    # --- local models ---
    rerank_device: str = "cpu"
    rerank_batch_size: int = 16
    embed_device: str = "cpu"
    # bge-small's trained context. Chunks are cut to fit under it, so this is
    # the ceiling that keeps a truncated tail from being embedded as if it were
    # the whole passage.
    embed_max_tokens: int = 512

    # --- text index (M3) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_timeout_s: float = 60.0
    # Empty for a local Qdrant, which listens without credentials; set for a
    # managed one, which does not. SecretStr for the same reason the model keys
    # are -- a Qdrant Cloud key is a bearer token for the whole cluster, and it
    # would otherwise ride along in every settings repr.
    qdrant_api_key: SecretStr = SecretStr("")
    # Set this and the server is not used at all: qdrant-client runs the store
    # in-process against a directory. It is the deployment mode for a single
    # container that has to be self-contained -- a Hugging Face Space cannot
    # run a second service, and a managed free cluster is worse than useless
    # for a portfolio link because it suspends after a week of nobody visiting
    # and is deleted after four.
    #
    # Not a drop-in for the server: the on-disk formats differ, so an embedded
    # store is built from a server one by `filing pack`, not copied.
    qdrant_path: Path | None = None
    # Measured, not chosen. On Gemini's free tier a batch of 32 chunks (~13.5k
    # tokens) is served in about two seconds; a batch of 100 (~42k tokens) is
    # answered 429 every time, and the retry costs more than the batch saved.
    # It is also a sensible forward-pass width on eight CPU cores, so the local
    # backend uses the same number. See docs/retrieval.md.
    embed_batch_size: int = 32
    # Only the hosted backends read this. Gemini meters embeddings in tokens
    # per minute as well as documents per day, so ``TokenPacer`` spends the
    # budget forwards rather than discovering it one 429 at a time -- reactive
    # backoff measured 40 chunks a minute against a ceiling that permits about
    # 60. Set below the published 30k so a bad token estimate has somewhere to
    # be wrong. The daily cap is the one that made the local backend the
    # default; no pacing survives 1,000 documents a day.
    embed_tokens_per_minute: int = 27_000

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
    def chunks_dir(self) -> Path:
        """The chunk cache: parquet, derived, and safe to delete.

        Parquet rather than DuckDB because nothing here is queried while it is
        written -- it is read once per index build, whole -- and because a
        columnar file the size of the corpus text compresses to something a
        person can copy between machines.
        """
        return self.data_dir / "chunks"

    @property
    def index_dir(self) -> Path:
        """Where the BM25 index lives. The dense half lives in Qdrant."""
        return self.data_dir / "index"

    @property
    def graph_dir(self) -> Path:
        """The entity graph, stored as its edge list.

        NetworkX is an in-memory structure with no file format worth committing
        to, so the durable artefact is the edges -- each with the sentence and
        the offsets that justify it -- and the graph is rebuilt from them.
        """
        return self.data_dir / "graph"

    @property
    def registry(self) -> dict[str, ModelSpec]:
        return MODEL_REGISTRY[self.llm_backend]

    def provider_credentials(self, provider: str) -> tuple[str, str, str]:
        """Base URL, secret, and where to get one, for an OpenAI-compatible provider.

        The third element doubles as a flag: an empty string means the provider
        needs no credential, so a blank secret is not an error. That is not a
        hypothetical case -- it is how OVHcloud works, and the reason it is the
        one provider here that no country list can take away.
        """
        table = {
            "cohere": (
                self.cohere_base_url,
                self.cohere_api_key.get_secret_value(),
                "https://dashboard.cohere.com/api-keys",
            ),
            "groq": (
                self.groq_base_url,
                self.groq_api_key.get_secret_value(),
                "https://console.groq.com/keys",
            ),
            "ovh": (self.ovh_base_url, "", ""),
        }
        if provider not in table:
            raise KeyError(f"{provider!r} is not an OpenAI-compatible provider: {sorted(table)}")
        return table[provider]


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
