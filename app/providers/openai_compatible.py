from __future__ import annotations

import httpx

from ..constants import ErrorCategory
from ..errors import ProviderError
from ..models import LLMRequest, LLMResponse, ProviderConfig
from .base import BaseProvider


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI-compatible chat completions provider."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = (config.base_url or "").rstrip("/")

    def generate(self, request: LLMRequest) -> LLMResponse:
        api_key = self._require_api_key()
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature if request.temperature is not None else self.config.temperature,
            "top_p": request.top_p if request.top_p is not None else self.config.top_p,
            "max_tokens": request.max_output_tokens
            if request.max_output_tokens is not None
            else self.config.max_output_tokens,
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            with self._http_client(request.timeout_seconds or self.config.timeout_seconds) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"OpenAI-compatible request timed out: {exc}",
                category=ErrorCategory.TIMEOUT,
                transient=True,
                provider_name=self.config.name,
                model_name=self.config.model,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"OpenAI-compatible request failed: {exc}",
                category=ErrorCategory.PROVIDER_UNAVAILABLE,
                transient=True,
                provider_name=self.config.name,
                model_name=self.config.model,
            ) from exc

        self._raise_for_status(response, provider_name=self.config.name, model_name=self.config.model)
        payload_json = response.json()
        return LLMResponse(
            provider_name=self.config.name,
            model_name=self.config.model,
            text=self._extract_openai_content(payload_json),
            raw_payload=payload_json,
        )
