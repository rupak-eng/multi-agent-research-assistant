"""Provider abstractions. Swap real services without touching agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMRequest(BaseModel):
    system: str
    user: str
    response_model: type[BaseModel] | None = None  # when set, expect JSON
    max_tokens: int = 2048
    temperature: float = 0.2


class LLMResponse(BaseModel):
    text: str
    tokens_in: int
    tokens_out: int
    model: str
    latency_ms: int


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, req: LLMRequest) -> LLMResponse:
        ...


class SearchHit(BaseModel):
    url: str
    title: str
    snippet: str


class SearchProvider(ABC):
    name: str = "base"

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        """Raise SearchProviderError on failure; return [] when no results."""
        ...


class SearchProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable
