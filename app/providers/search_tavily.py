"""Tavily web-search provider (https://tavily.com).

Structured failures: transport/rate-limit problems raise SearchProviderError
(retryable); a valid response with no results returns [] (not an error) so the
supervisor can route around it instead of crashing the run.
"""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import SearchHit, SearchProvider, SearchProviderError


class TavilySearchProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str, timeout: float = 30.0):
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required for the tavily provider")
        self.api_key = api_key
        self.timeout = timeout

    @retry(
        retry=retry_if_exception_type(SearchProviderError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        try:
            resp = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                    "include_answer": False,
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise SearchProviderError(f"tavily transport error: {e}", retryable=True) from e
        if resp.status_code == 429:
            raise SearchProviderError("tavily rate limited (429)", retryable=True)
        if resp.status_code >= 500:
            raise SearchProviderError(f"tavily HTTP {resp.status_code}", retryable=True)
        if resp.status_code >= 400:
            raise SearchProviderError(f"tavily HTTP {resp.status_code}: {resp.text[:200]}",
                                       retryable=False)
        results = resp.json().get("results", [])
        return [
            SearchHit(url=r.get("url", ""), title=r.get("title", ""),
                      snippet=(r.get("content", "") or "")[:2000])
            for r in results if r.get("url")
        ]
