from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..constants import ErrorCategory
from ..errors import ProviderError
from ..models import LLMRequest, LLMResponse, ProviderConfig


class BaseProvider(ABC):
    """Abstract provider interface."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Send a request to the provider and return raw text."""

    def _http_client(self, timeout_seconds: float) -> httpx.Client:
        return httpx.Client(timeout=timeout_seconds, follow_redirects=True)

    def _require_api_key(self) -> str:
        if not self.config.api_key_env:
            raise ProviderError(
                f"Provider `{self.config.name}` requires no API key, but `_require_api_key` was called unexpectedly.",
                category=ErrorCategory.INVALID_REQUEST,
                provider_name=self.config.name,
                model_name=self.config.model,
            )

        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Environment variable `{self.config.api_key_env}` is not set for provider `{self.config.name}`.",
                category=ErrorCategory.AUTH_ERROR,
                provider_name=self.config.name,
                model_name=self.config.model,
            )
        return api_key

    def _raise_for_status(
        self,
        response: httpx.Response,
        *,
        provider_name: str,
        model_name: str,
    ) -> None:
        if response.is_success:
            return

        message = response.text[:800]
        category = ErrorCategory.UNKNOWN
        transient = False

        if response.status_code in {401, 403}:
            category = ErrorCategory.AUTH_ERROR
        elif response.status_code == 408:
            category = ErrorCategory.TIMEOUT
            transient = True
        elif response.status_code == 413:
            category = ErrorCategory.CONTEXT_TOO_LONG
        elif response.status_code == 422:
            category = ErrorCategory.INVALID_REQUEST
        elif response.status_code == 429:
            lowered = message.lower()
            if "quota" in lowered or "resource exhausted" in lowered or "insufficient" in lowered:
                category = ErrorCategory.QUOTA_EXHAUSTED
            else:
                category = ErrorCategory.RATE_LIMITED
                transient = True
        elif 500 <= response.status_code < 600:
            category = ErrorCategory.PROVIDER_UNAVAILABLE
            transient = True

        raise ProviderError(
            f"HTTP {response.status_code} from provider `{provider_name}`: {message}",
            category=category,
            status_code=response.status_code,
            response_text=message,
            transient=transient,
            provider_name=provider_name,
            model_name=model_name,
        )

    def _extract_openai_content(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise ProviderError(
                "Provider response does not contain `choices`.",
                category=ErrorCategory.INVALID_API_RESPONSE,
                provider_name=self.config.name,
                model_name=self.config.model,
            )

        choice = choices[0] or {}
        finish_reason = str(choice.get("finish_reason") or "")
        if finish_reason == "length":
            raise ProviderError(
                "Provider stopped because the response reached the token limit.",
                category=ErrorCategory.OUTPUT_TOO_LONG,
                provider_name=self.config.name,
                model_name=self.config.model,
            )

        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            if text_parts:
                return "\n".join(text_parts)

        raise ProviderError(
            "Provider response does not contain text content.",
            category=ErrorCategory.INVALID_API_RESPONSE,
            provider_name=self.config.name,
            model_name=self.config.model,
        )
