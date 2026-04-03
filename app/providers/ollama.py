from __future__ import annotations

import httpx

from ..constants import ErrorCategory
from ..errors import ProviderError
from ..models import LLMRequest, LLMResponse, ProviderConfig
from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """Local Ollama HTTP API provider."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = (config.base_url or "http://localhost:11434").rstrip("/")

    def generate(self, request: LLMRequest) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "options": {
                "temperature": request.temperature if request.temperature is not None else self.config.temperature,
                "top_p": request.top_p if request.top_p is not None else self.config.top_p,
                "num_predict": request.max_output_tokens
                if request.max_output_tokens is not None
                else self.config.max_output_tokens,
            },
        }

        try:
            with self._http_client(request.timeout_seconds or self.config.timeout_seconds) as client:
                response = client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Ollama request timed out: {exc}",
                category=ErrorCategory.TIMEOUT,
                transient=True,
                provider_name=self.config.name,
                model_name=self.config.model,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Ollama request failed: {exc}",
                category=ErrorCategory.PROVIDER_UNAVAILABLE,
                transient=True,
                provider_name=self.config.name,
                model_name=self.config.model,
            ) from exc

        self._raise_for_status(response, provider_name=self.config.name, model_name=self.config.model)
        payload_json = response.json()
        content = ((payload_json.get("message") or {}).get("content")) or ""
        if not content:
            raise ProviderError(
                "Ollama response does not contain message content.",
                category=ErrorCategory.INVALID_API_RESPONSE,
                provider_name=self.config.name,
                model_name=self.config.model,
            )

        if str(payload_json.get("done_reason") or "") == "length":
            raise ProviderError(
                "Ollama stopped because the response reached the token limit.",
                category=ErrorCategory.OUTPUT_TOO_LONG,
                provider_name=self.config.name,
                model_name=self.config.model,
            )

        return LLMResponse(
            provider_name=self.config.name,
            model_name=self.config.model,
            text=str(content),
            raw_payload=payload_json,
        )
