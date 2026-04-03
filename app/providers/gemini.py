from __future__ import annotations

from typing import Any

import httpx

from ..constants import ErrorCategory
from ..errors import ProviderError
from ..models import LLMRequest, LLMResponse, ProviderConfig
from .base import BaseProvider

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(BaseProvider):
    """Official Gemini API provider."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = (config.base_url or DEFAULT_GEMINI_BASE_URL).rstrip("/")

    def generate(self, request: LLMRequest) -> LLMResponse:
        api_key = self._require_api_key()
        url = f"{self.base_url}/models/{self.config.model}:generateContent"
        params = {"key": api_key}
        payload = {
            "systemInstruction": {"parts": [{"text": request.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": request.user_prompt}]}],
            "generationConfig": {
                "temperature": request.temperature if request.temperature is not None else self.config.temperature,
                "topP": request.top_p if request.top_p is not None else self.config.top_p,
                "maxOutputTokens": request.max_output_tokens
                if request.max_output_tokens is not None
                else self.config.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }

        try:
            with self._http_client(request.timeout_seconds or self.config.timeout_seconds) as client:
                response = client.post(url, params=params, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Gemini request timed out: {exc}",
                category=ErrorCategory.TIMEOUT,
                transient=True,
                provider_name=self.config.name,
                model_name=self.config.model,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Gemini request failed: {exc}",
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
            text=self._extract_text(payload_json),
            raw_payload=payload_json,
        )

    def _extract_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            raise ProviderError(
                "Gemini response does not contain `candidates`.",
                category=ErrorCategory.INVALID_API_RESPONSE,
                provider_name=self.config.name,
                model_name=self.config.model,
            )

        candidate = candidates[0] or {}
        finish_reason = str(candidate.get("finishReason") or "")
        if finish_reason == "MAX_TOKENS":
            raise ProviderError(
                "Gemini stopped because the response reached the token limit.",
                category=ErrorCategory.OUTPUT_TOO_LONG,
                provider_name=self.config.name,
                model_name=self.config.model,
            )

        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text_parts = [str(part.get("text", "")) for part in parts if isinstance(part, dict) and part.get("text")]
        if not text_parts:
            raise ProviderError(
                "Gemini response does not contain text parts.",
                category=ErrorCategory.INVALID_API_RESPONSE,
                provider_name=self.config.name,
                model_name=self.config.model,
            )

        return "\n".join(text_parts)
