from __future__ import annotations

from enum import Enum
from pathlib import Path

APP_NAME = "transcript-md-summarizer"
DEFAULT_CONFIG_PATH = Path("config/models.yaml")
DEFAULT_LOG_FILE = Path("logs/app.log")
DEFAULT_LANGUAGE = "ru"
DEFAULT_SOURCE_TYPE = "transcript"
DEFAULT_STATUS = "готово"
TIME_RANGE_SEPARATOR = "–"
JSON_CODE_FENCE = "```"


class ProviderKind(str, Enum):
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"


class StageName(str, Enum):
    CHUNK_SUMMARY = "chunk_summary"
    FINAL_MERGE = "final_merge"
    PROVIDER_TEST = "provider_test"


class ErrorCategory(str, Enum):
    QUOTA_EXHAUSTED = "quota_exhausted"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    CONTEXT_TOO_LONG = "context_too_long"
    OUTPUT_TOO_LONG = "output_too_long"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MALFORMED_RESPONSE = "malformed_response"
    INVALID_API_RESPONSE = "invalid_api_response"
    AUTH_ERROR = "auth_error"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN = "unknown"


class FallbackAction(str, Enum):
    RETRY = "retry"
    DEGRADE = "degrade"
    FALLBACK = "fallback"
    FAIL = "fail"
    SUCCESS = "success"


TRANSIENT_ERROR_CATEGORIES = {
    ErrorCategory.RATE_LIMITED,
    ErrorCategory.TIMEOUT,
    ErrorCategory.PROVIDER_UNAVAILABLE,
}

DEGRADE_ELIGIBLE_CATEGORIES = {
    ErrorCategory.OUTPUT_TOO_LONG,
    ErrorCategory.MALFORMED_RESPONSE,
    ErrorCategory.INVALID_API_RESPONSE,
}

QUOTA_MARKERS = (
    "quota",
    "insufficient_quota",
    "resource exhausted",
    "billing",
    "credit balance",
)

RATE_LIMIT_MARKERS = (
    "rate limit",
    "too many requests",
    "retry later",
)

TIMEOUT_MARKERS = (
    "timeout",
    "timed out",
    "deadline exceeded",
)

CONTEXT_MARKERS = (
    "context length",
    "too many tokens",
    "prompt is too long",
    "input too long",
    "maximum context",
    "token limit",
)

OUTPUT_MARKERS = (
    "finish_reason=length",
    "max tokens",
    "maximum output",
    "response too large",
)

AUTH_MARKERS = (
    "unauthorized",
    "authentication",
    "invalid api key",
    "permission denied",
    "forbidden",
)

INVALID_REQUEST_MARKERS = (
    "invalid request",
    "validation error",
    "unsupported parameter",
    "bad request",
)
