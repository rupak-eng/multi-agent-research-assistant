"""Tavily web-search provider.

Verified pattern (2026-09-30): POST https://api.tavily.com/search with the
key as `Authorization: Bearer <key>`. Do NOT also put the key in the JSON
body — Tavily prioritizes the body field, which never gets replaced at
egress.

Structured failures: transport/rate-limit problems raise SearchProviderError
(retryable); a valid response with no results returns [] (not an error) so
the supervisor can route around it instead of crashing the run.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import SearchHit, SearchProvider, SearchProviderError
from .credentials import require_credential

BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, SearchProviderError) and exc.retryable


class TavilySearchProvider(SearchProvider):
    name = "tavily"
    API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, timeout: float = 30.0,
                 search_depth: str = "advanced"):
        # May be an hsurr: surrogate — opaque here, swapped at egress.
        self._key = api_key or require_credential(
            "custom.tavily", "TAVILY_API_KEY", "Tavily search provider")
        self.timeout = timeout
        self.search_depth = search_depth

    @retry(retry=_is_retryable, stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        body = json.dumps({
            "query": query,
            "max_results": max_results,
            "search_depth": self.search_depth,
            "include_answer": False,
        }).encode()
        req = urllib.request.Request(
            self.API_URL, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": BROWSER_UA,
                     "Authorization": f"Bearer {self._key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:200]
            if e.code == 429 or 500 <= e.code < 600:
                raise SearchProviderError(f"tavily HTTP {e.code}: {detail}",
                                          retryable=True) from e
            raise SearchProviderError(f"tavily HTTP {e.code}: {detail}",
                                      retryable=False) from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise SearchProviderError(f"tavily transport error: {e}",
                                      retryable=True) from e
        results = data.get("results", [])
        return [
            SearchHit(url=r.get("url", ""), title=r.get("title", ""),
                      snippet=(r.get("content", "") or "")[:2000])
            for r in results if r.get("url")
        ]
