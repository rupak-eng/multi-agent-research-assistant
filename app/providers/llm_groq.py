"""Groq LLM provider (OpenAI-compatible chat completions).

Verified notes (2026-09-30):
- Endpoint: https://api.groq.com/openai/v1/chat/completions
- A browser-like User-Agent header is REQUIRED, otherwise Cloudflare
  returns 403 (error 1010).
- Models on this key: openai/gpt-oss-20b (default, fast/cheap),
  openai/gpt-oss-120b (quality), qwen/qwen3.8-27b.
- gpt-oss usage includes reasoning_tokens; content is parsed defensively.

Auth: Bearer <key-or-surrogate>. The surrogate is swapped for the real key
by the egress proxy; it is never logged or printed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMProvider, LLMRequest, LLMResponse
from .credentials import require_credential
from .tokens import TokenCounter

BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# Per-model output-token ceilings enforced by Groq's OTPM limits.
# qwen/qwen3.8-27b: 1000 output tokens/min on the dev tier, and Groq
# rejects the request when Used + max_tokens would exceed the limit,
# so request strictly below the ceiling (900) to leave headroom.
MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "qwen/qwen3.8-27b": 900,
}


class LLMProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, LLMProviderError) and exc.retryable


class GroqLLMProvider(LLMProvider):
    name = "groq"
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None,
                 model: str = "openai/gpt-oss-20b",
                 timeout: float = 120.0,
                 min_interval_sec: float = 0.0):
        # May be an hsurr: surrogate — opaque here, swapped at egress.
        self._key = api_key or require_credential(
            "custom.groq", "GROQ_API_KEY", "Groq LLM provider")
        self.model = model
        self.timeout = timeout
        self.counter = TokenCounter()
        # Pacing between calls (seconds). Set >0 to stay under TPM limits
        # when the key is shared or the tier is low.
        self.min_interval_sec = min_interval_sec
        self._last_call_ts = 0.0

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "User-Agent": BROWSER_UA,
            "Authorization": f"Bearer {self._key}",
        }

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.API_URL, data=json.dumps(payload).encode(),
            headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if e.code == 403 and "1010" in body:
                raise LLMProviderError(
                    "groq Cloudflare 403 (error 1010): request blocked — "
                    "browser User-Agent header is required",
                    retryable=False) from e
            if e.code == 429 or 500 <= e.code < 600:
                raise LLMProviderError(f"groq HTTP {e.code}: {body}",
                                       retryable=True) from e
            raise LLMProviderError(f"groq HTTP {e.code}: {body}",
                                   retryable=False) from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise LLMProviderError(f"groq transport error: {e}",
                                   retryable=True) from e

    @retry(retry=_is_retryable,
           stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60),
           reraise=True)
    def _post_with_retry(self, payload: dict) -> dict:
        return self._post(payload)

    def complete(self, req: LLMRequest) -> LLMResponse:
        start = time.monotonic()
        # Pace calls to respect TPM limits on shared/low-tier keys.
        if self.min_interval_sec > 0:
            dt = time.monotonic() - self._last_call_ts
            if dt < self.min_interval_sec:
                time.sleep(self.min_interval_sec - dt)
        self._last_call_ts = time.monotonic()
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            "temperature": req.temperature,
            # Clamp to the model's OTPM ceiling (Groq rejects the request
            # outright when max_tokens exceeds it).
            "max_tokens": min(
                req.max_tokens, MODEL_MAX_OUTPUT_TOKENS.get(self.model, req.max_tokens)
            ),
        }
        if req.response_model is not None:
            payload["response_format"] = {"type": "json_object"}
        # Structured output from small models is occasionally malformed
        # (wrong enum value, int id instead of str, ...). Temperature > 0
        # makes each attempt a fresh sample, so a bounded retry is the
        # honest fix; validation still runs on every attempt.
        attempts = 3 if req.response_model is not None else 1
        last_err: LLMProviderError | None = None
        text = ""
        for _ in range(attempts):
            data = self._post_with_retry(payload)
            text = self._extract_text(data)
            if req.response_model is None:
                break
            try:
                self._validate_json(text, req.response_model)
                break
            except LLMProviderError as e:
                last_err = e
        else:
            raise last_err  # type: ignore[misc]
        usage = data.get("usage") or {}
        tokens_in = usage.get("prompt_tokens") or self.counter.count_messages(
            req.system, req.user)
        tokens_out = (usage.get("completion_tokens")
                      or self.counter.count(text))
        return LLMResponse(
            text=text, tokens_in=tokens_in, tokens_out=tokens_out,
            model=self.model,
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Defensive: gpt-oss may return reasoning-heavy shapes."""
        try:
            choices = data.get("choices") or []
            message = (choices[0].get("message") if choices else {}) or {}
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMProviderError(
                f"unexpected groq response shape: {str(data)[:300]}",
                retryable=False) from e
        text = message.get("content") or ""
        if not isinstance(text, str):
            text = ""
        text = text.strip()
        if not text:
            # Surface why: reasoning-only or empty finish.
            finish = (choices[0].get("finish_reason") if choices else "?")
            raise LLMProviderError(
                f"groq returned no content (finish_reason={finish})",
                retryable=False)
        return text

    @staticmethod
    def _validate_json(text: str, model: type[BaseModel]) -> None:
        try:
            model.model_validate_json(text)
        except ValidationError as e:
            raise LLMProviderError(
                f"groq returned invalid JSON for {model.__name__}: {e}",
                retryable=False) from e
