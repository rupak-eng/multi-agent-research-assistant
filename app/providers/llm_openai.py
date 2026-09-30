"""OpenAI-compatible chat-completions LLM provider.

Works with OpenAI, Ollama (http://localhost:11434/v1), vLLM, or any server
implementing POST {base_url}/chat/completions. When the caller passes a
response_model we request json_object mode and validate strictly.
"""

from __future__ import annotations

import time

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import LLMProvider, LLMRequest, LLMResponse
from .tokens import TokenCounter


class LLMProviderError(RuntimeError):
    pass


class OpenAICompatLLMProvider(LLMProvider):
    name = "openai_compat"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.counter = TokenCounter()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, LLMProviderError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = httpx.post(
            f"{self.base_url}/chat/completions", json=payload,
            headers=headers, timeout=self.timeout,
        )
        if resp.status_code >= 500 or resp.status_code == 429:
            raise LLMProviderError(f"provider HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code >= 400:
            raise LLMProviderError(f"provider HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def complete(self, req: LLMRequest) -> LLMResponse:
        start = time.monotonic()
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.response_model is not None:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self._post(payload)
        except Exception as e:
            raise LLMProviderError(str(e)) from e
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMProviderError(f"unexpected provider response shape: {str(data)[:300]}") from e
        if req.response_model is not None:
            self._validate_json(text, req.response_model)
        usage = data.get("usage") or {}
        tokens_in = usage.get("prompt_tokens") or self.counter.count_messages(req.system, req.user)
        tokens_out = usage.get("completion_tokens") or self.counter.count(text)
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    @staticmethod
    def _validate_json(text: str, model: type[BaseModel]) -> None:
        try:
            model.model_validate_json(text)
        except ValidationError as e:
            raise LLMProviderError(f"provider returned invalid JSON for {model.__name__}: {e}") from e
