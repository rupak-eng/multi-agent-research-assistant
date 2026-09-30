"""Provider factory: choose implementations from Settings."""

from __future__ import annotations

from ..config import Settings
from .base import LLMProvider, SearchProvider
from .llm_groq import GroqLLMProvider
from .llm_openai import OpenAICompatLLMProvider
from .llm_stub import StubLLMProvider
from .search_stub import StubSearchProvider
from .search_tavily import TavilySearchProvider


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "groq":
        # Production path: Groq via Secure Vault surrogate or GROQ_API_KEY.
        return GroqLLMProvider(model=settings.groq_model)
    if settings.llm_provider == "openai_compat":
        return OpenAICompatLLMProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    if settings.llm_provider == "stub":
        return StubLLMProvider()
    raise ValueError(f"unknown LLM_PROVIDER={settings.llm_provider!r}")


def build_search(settings: Settings) -> SearchProvider:
    if settings.search_provider == "tavily":
        return TavilySearchProvider(api_key=settings.tavily_api_key)
    if settings.search_provider == "stub":
        return StubSearchProvider()
    raise ValueError(f"unknown SEARCH_PROVIDER={settings.search_provider!r}")
