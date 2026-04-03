from __future__ import annotations

from dataclasses import dataclass, field

from .constants import ErrorCategory


class AppError(Exception):
    """Base application error."""


class ConfigError(AppError):
    """Raised when configuration cannot be loaded or validated."""


class TranscriptLoadError(AppError):
    """Raised when transcript JSON cannot be parsed."""


class MarkdownRenderError(AppError):
    """Raised when markdown cannot be rendered or written."""


class SummarizationError(AppError):
    """Raised for end-to-end summarization failures."""


class ProviderError(AppError):
    """Raised when a provider request fails."""

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory | None = None,
        status_code: int | None = None,
        response_text: str | None = None,
        transient: bool | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.response_text = response_text
        self.transient = transient
        self.provider_name = provider_name
        self.model_name = model_name


class InvalidStructuredResponseError(ProviderError):
    """Raised when a provider returns content that is not valid structured JSON."""


@dataclass(slots=True)
class AttemptRecord:
    provider_name: str
    model_name: str
    stage: str
    attempt_number: int
    action: str
    message: str
    category: str | None = None
    degraded: bool = False


@dataclass(slots=True)
class AllProvidersFailedError(SummarizationError):
    """Raised after all configured providers are exhausted."""

    message: str
    attempts: list[AttemptRecord] = field(default_factory=list)
    last_category: ErrorCategory | None = None

    def __str__(self) -> str:
        return self.message
