"""Backend selection. The only place either backend class is constructed."""

from __future__ import annotations

from functools import lru_cache

from filing.config import Backend, Settings, settings
from filing.llm.base import LLMBackend
from filing.tracing import setup_tracing


def build_backend(cfg: Settings | None = None, backend: Backend | None = None) -> LLMBackend:
    cfg = cfg or settings()
    choice = backend or cfg.llm_backend
    setup_tracing(cfg)
    if choice == "gemini":
        from filing.llm.gemini import GeminiBackend

        return GeminiBackend(cfg)
    if choice == "nvidia":
        from filing.llm.nvidia import NvidiaBackend

        return NvidiaBackend(cfg)
    if choice == "ollama":
        from filing.llm.fallback_ollama import OllamaBackend

        return OllamaBackend(cfg)
    raise ValueError(f"unknown backend {choice!r}")


@lru_cache(maxsize=2)
def get_backend(backend: Backend | None = None) -> LLMBackend:
    """Process-wide singleton -- one limiter and one cache handle, shared."""
    return build_backend(backend=backend)
