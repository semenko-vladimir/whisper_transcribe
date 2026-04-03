from __future__ import annotations

import json
import logging
import re
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .constants import (
    AUTH_MARKERS,
    CONTEXT_MARKERS,
    DEGRADE_ELIGIBLE_CATEGORIES,
    ErrorCategory,
    FallbackAction,
    INVALID_REQUEST_MARKERS,
    OUTPUT_MARKERS,
    QUOTA_MARKERS,
    RATE_LIMIT_MARKERS,
    TIMEOUT_MARKERS,
    TRANSIENT_ERROR_CATEGORIES,
)
from .errors import (
    AllProvidersFailedError,
    AttemptRecord,
    InvalidStructuredResponseError,
    ProviderError,
)
from .models import ClassifiedError, FallbackDecision, LLMRequest, ProviderConfig
from .providers import build_provider

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(?P<body>[\s\S]*?)\s*```", re.IGNORECASE)
T = TypeVar("T", bound=BaseModel)


class ErrorClassifier:
    """Classify provider failures into policy-friendly buckets."""

    def classify(self, error: Exception) -> ClassifiedError:
        if isinstance(error, ProviderError):
            category = error.category or self._classify_message(str(error))
            transient = error.transient if error.transient is not None else category in TRANSIENT_ERROR_CATEGORIES
            return ClassifiedError(category=category, message=str(error), transient=transient)

        if isinstance(error, httpx.TimeoutException):
            return ClassifiedError(
                category=ErrorCategory.TIMEOUT,
                message=str(error),
                transient=True,
            )

        if isinstance(error, json.JSONDecodeError):
            return ClassifiedError(
                category=ErrorCategory.MALFORMED_RESPONSE,
                message=str(error),
                transient=False,
            )

        return ClassifiedError(
            category=self._classify_message(str(error)),
            message=str(error),
            transient=False,
        )

    def _classify_message(self, message: str) -> ErrorCategory:
        lowered = message.lower()

        if any(marker in lowered for marker in QUOTA_MARKERS):
            return ErrorCategory.QUOTA_EXHAUSTED
        if any(marker in lowered for marker in RATE_LIMIT_MARKERS):
            return ErrorCategory.RATE_LIMITED
        if any(marker in lowered for marker in TIMEOUT_MARKERS):
            return ErrorCategory.TIMEOUT
        if any(marker in lowered for marker in CONTEXT_MARKERS):
            return ErrorCategory.CONTEXT_TOO_LONG
        if any(marker in lowered for marker in OUTPUT_MARKERS):
            return ErrorCategory.OUTPUT_TOO_LONG
        if any(marker in lowered for marker in AUTH_MARKERS):
            return ErrorCategory.AUTH_ERROR
        if any(marker in lowered for marker in INVALID_REQUEST_MARKERS):
            return ErrorCategory.INVALID_REQUEST
        if "malformed" in lowered or "invalid json" in lowered or "schema" in lowered:
            return ErrorCategory.MALFORMED_RESPONSE
        if "unavailable" in lowered or "service down" in lowered or "502" in lowered or "503" in lowered:
            return ErrorCategory.PROVIDER_UNAVAILABLE
        return ErrorCategory.UNKNOWN


class RetryPolicy:
    """Decide when a retry is preferable to immediate fallback."""

    def __init__(self, base_delay_seconds: float = 1.5) -> None:
        self.base_delay_seconds = base_delay_seconds

    def should_retry(self, classified: ClassifiedError, attempt_index: int, max_retries: int) -> bool:
        return classified.transient and attempt_index < max_retries

    def delay_seconds(self, attempt_index: int) -> float:
        return self.base_delay_seconds * (attempt_index + 1)


class FallbackPolicy:
    """Decide when to degrade request parameters or move to the next provider."""

    def __init__(self, max_degrade_attempts: int = 2) -> None:
        self.max_degrade_attempts = max_degrade_attempts

    def decide(
        self,
        *,
        classified: ClassifiedError,
        degrade_attempts: int,
        has_more_providers: bool,
    ) -> FallbackDecision:
        if classified.category in DEGRADE_ELIGIBLE_CATEGORIES and degrade_attempts < self.max_degrade_attempts:
            return FallbackDecision(
                action=FallbackAction.DEGRADE,
                reason="Reduce output pressure and retry the same provider configuration.",
            )

        if has_more_providers:
            return FallbackDecision(
                action=FallbackAction.FALLBACK,
                reason="Switch to the next provider configuration in the fallback chain.",
            )

        return FallbackDecision(
            action=FallbackAction.FAIL,
            reason="No providers remain and no further degrade step is available.",
        )

    def apply_degrade(self, request: LLMRequest) -> LLMRequest:
        current_max_output = request.max_output_tokens or 1024
        degraded_max_output = max(256, int(current_max_output * 0.75))
        degraded_temperature = min(request.temperature or 0.2, 0.2)

        return request.with_updates(
            max_output_tokens=degraded_max_output,
            temperature=degraded_temperature,
            metadata={**request.metadata, "degraded": True},
        )


def extract_json_payload(text: str) -> str:
    """Extract JSON from a raw model response, including fenced snippets."""

    stripped = text.strip()
    if not stripped:
        raise InvalidStructuredResponseError(
            "Provider returned an empty response.",
            category=ErrorCategory.MALFORMED_RESPONSE,
        )

    match = JSON_BLOCK_RE.search(stripped)
    if match:
        stripped = match.group("body").strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped

    first_object = stripped.find("{")
    last_object = stripped.rfind("}")
    if first_object != -1 and last_object != -1 and last_object > first_object:
        return stripped[first_object : last_object + 1]

    first_array = stripped.find("[")
    last_array = stripped.rfind("]")
    if first_array != -1 and last_array != -1 and last_array > first_array:
        return stripped[first_array : last_array + 1]

    raise InvalidStructuredResponseError(
        "Provider response did not contain a JSON object or array.",
        category=ErrorCategory.MALFORMED_RESPONSE,
    )


def parse_structured_response(text: str, model_type: type[T]) -> T:
    """Validate structured model output against a pydantic schema."""

    try:
        payload = extract_json_payload(text)
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise InvalidStructuredResponseError(
            f"Provider response was not valid JSON: {exc}",
            category=ErrorCategory.MALFORMED_RESPONSE,
        ) from exc

    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        raise InvalidStructuredResponseError(
            f"Provider JSON did not match the expected schema: {exc}",
            category=ErrorCategory.INVALID_API_RESPONSE,
        ) from exc


class ProviderFallbackExecutor:
    """Execute a request across providers with retry, degrade, and fallback policies."""

    def __init__(
        self,
        providers: list[ProviderConfig],
        *,
        logger: logging.Logger,
        retry_policy: RetryPolicy,
        fallback_policy: FallbackPolicy,
        error_classifier: ErrorClassifier,
    ) -> None:
        self.providers = providers
        self.logger = logger
        self.retry_policy = retry_policy
        self.fallback_policy = fallback_policy
        self.error_classifier = error_classifier

    def generate_structured(
        self,
        request: LLMRequest,
        model_type: type[T],
        *,
        provider_name: str | None = None,
    ) -> tuple[T, list[AttemptRecord]]:
        attempt_records: list[AttemptRecord] = []
        selected_providers = self._select_providers(provider_name=provider_name)

        if not selected_providers:
            raise AllProvidersFailedError(
                message="No providers matched the requested selection.",
                attempts=[],
                last_category=ErrorCategory.UNKNOWN,
            )

        for provider_index, provider_config in enumerate(selected_providers):
            provider = build_provider(provider_config)
            retry_attempt = 0
            degrade_attempts = 0
            current_request = request.with_updates(
                temperature=request.temperature if request.temperature is not None else provider_config.temperature,
                top_p=request.top_p if request.top_p is not None else provider_config.top_p,
                max_output_tokens=request.max_output_tokens
                if request.max_output_tokens is not None
                else provider_config.max_output_tokens,
                timeout_seconds=request.timeout_seconds
                if request.timeout_seconds is not None
                else provider_config.timeout_seconds,
            )

            while True:
                try:
                    response = provider.generate(current_request)
                    parsed = parse_structured_response(response.text, model_type)
                    attempt_records.append(
                        AttemptRecord(
                            provider_name=provider_config.name,
                            model_name=provider_config.model,
                            stage=request.stage.value,
                            attempt_number=retry_attempt + 1,
                            action=FallbackAction.SUCCESS,
                            message="Structured response parsed successfully.",
                        )
                    )
                    return parsed, attempt_records
                except Exception as exc:
                    classified = self.error_classifier.classify(exc)
                    attempt_records.append(
                        AttemptRecord(
                            provider_name=provider_config.name,
                            model_name=provider_config.model,
                            stage=request.stage.value,
                            attempt_number=retry_attempt + 1,
                            action="error",
                            message=classified.message,
                            category=classified.category.value,
                            degraded=bool(current_request.metadata.get("degraded")),
                        )
                    )
                    self.logger.warning(
                        "Provider failed: provider=%s model=%s stage=%s category=%s message=%s",
                        provider_config.name,
                        provider_config.model,
                        request.stage.value,
                        classified.category.value,
                        classified.message,
                    )

                    if self.retry_policy.should_retry(classified, retry_attempt, provider_config.max_retries):
                        delay = self.retry_policy.delay_seconds(retry_attempt)
                        retry_attempt += 1
                        self.logger.info(
                            "Retrying provider %s after %.1fs (%s/%s).",
                            provider_config.name,
                            delay,
                            retry_attempt,
                            provider_config.max_retries,
                        )
                        time.sleep(delay)
                        continue

                    has_more_providers = provider_index < len(selected_providers) - 1
                    decision = self.fallback_policy.decide(
                        classified=classified,
                        degrade_attempts=degrade_attempts,
                        has_more_providers=has_more_providers,
                    )

                    if decision.action == FallbackAction.DEGRADE:
                        degrade_attempts += 1
                        current_request = self.fallback_policy.apply_degrade(current_request)
                        self.logger.info(
                            "Applying degrade strategy for provider=%s model=%s stage=%s.",
                            provider_config.name,
                            provider_config.model,
                            request.stage.value,
                        )
                        continue

                    if decision.action == FallbackAction.FALLBACK:
                        break

                    raise AllProvidersFailedError(
                        message=f"All provider configurations failed during stage `{request.stage.value}`.",
                        attempts=attempt_records,
                        last_category=classified.category,
                    )

        last_category = ErrorCategory.UNKNOWN
        if attempt_records and attempt_records[-1].category:
            last_category = ErrorCategory(attempt_records[-1].category)
        raise AllProvidersFailedError(
            message=f"All provider configurations failed during stage `{request.stage.value}`.",
            attempts=attempt_records,
            last_category=last_category,
        )

    def _select_providers(self, *, provider_name: str | None) -> list[ProviderConfig]:
        if provider_name is None:
            return self.providers

        return [provider for provider in self.providers if provider.name == provider_name]
