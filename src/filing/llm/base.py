"""The interface every backend implements.

Three verbs. If a caller anywhere in this project needs a fourth, it belongs
here first -- not as a direct HTTP call from a node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

InputType = Literal["query", "passage"]


@dataclass(frozen=True, slots=True)
class Ranking:
    """One reranked passage: where it was, and how relevant the cross-encoder found it."""

    index: int  # position in the passages list as submitted
    score: float

    def __repr__(self) -> str:  # keeps smoke output readable
        return f"Ranking(index={self.index}, score={self.score:.4f})"


@dataclass(frozen=True, slots=True)
class Usage:
    """Per-process call accounting -- what the free tier actually spent."""

    http_calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        role: str = "chat",
        model_id: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str: ...

    def embed(
        self,
        texts: list[str],
        *,
        input_type: InputType = "passage",
        model_id: str | None = None,
    ) -> list[list[float]]: ...

    def rerank(
        self,
        query: str,
        passages: list[str],
        *,
        top_n: int | None = None,
        model_id: str | None = None,
    ) -> list[Ranking]: ...

    def usage(self) -> Usage: ...
