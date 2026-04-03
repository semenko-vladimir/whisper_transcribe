from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LANGUAGE,
    DEFAULT_LOG_FILE,
    ErrorCategory,
    ProviderKind,
    StageName,
)


class TranscriptSegment(BaseModel):
    """Normalized transcript segment."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    start: float = 0.0
    end: float = 0.0
    text: str
    source: str = "segment"

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.split())

    @model_validator(mode="after")
    def normalize_range(self) -> "TranscriptSegment":
        if self.start < 0:
            self.start = 0.0
        if self.end < self.start:
            self.end = self.start
        return self


class TranscriptDocument(BaseModel):
    """Loaded transcript plus metadata."""

    model_config = ConfigDict(extra="ignore")

    input_source: str | None = None
    original_input: str | None = None
    source_type: str | None = None
    detected_language: str | None = None
    segments: list[TranscriptSegment]
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranscriptChunk(BaseModel):
    """Chunked transcript prepared for stage-one summarization."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: int
    start: float
    end: float
    start_timecode: str
    end_timecode: str
    text: str
    segment_count: int
    character_count: int


class SummarySection(BaseModel):
    """Semantic markdown-ready section."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: str
    start_timecode: str
    end_timecode: str
    main_point: str
    bullet_points: list[str] = Field(default_factory=list)
    conclusion: str

    @field_validator("bullet_points")
    @classmethod
    def normalize_bullets(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class StructuredSummary(BaseModel):
    """Structured model output used for intermediate JSON and final markdown."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: str
    overview: str
    sections: list[SummarySection] = Field(default_factory=list)
    key_ideas: list[str] = Field(default_factory=list)
    practical_takeaways: list[str] = Field(default_factory=list)

    @field_validator("key_ideas", "practical_takeaways")
    @classmethod
    def normalize_string_list(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class ChunkSummaryArtifact(BaseModel):
    """Saved result of a single chunk summarization."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: int
    chunk_start_timecode: str
    chunk_end_timecode: str
    chunk_character_count: int
    summary: StructuredSummary


class ProviderConfig(BaseModel):
    """Single provider entry from YAML config."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    name: str
    provider: ProviderKind
    api_key_env: str | None = None
    base_url: str | None = None
    model: str
    temperature: float = 0.2
    top_p: float = 0.9
    max_output_tokens: int = 4096
    timeout_seconds: float = 60.0
    max_retries: int = 2
    priority: int = 1
    enabled: bool = True
    context_window_tokens: int | None = None

    @model_validator(mode="after")
    def validate_provider_specific_fields(self) -> "ProviderConfig":
        if self.provider in {ProviderKind.GEMINI, ProviderKind.OPENAI_COMPATIBLE} and not self.api_key_env:
            raise ValueError("api_key_env is required for cloud providers.")
        if self.provider in {ProviderKind.OPENAI_COMPATIBLE, ProviderKind.OLLAMA} and not self.base_url:
            raise ValueError("base_url is required for openai_compatible and ollama providers.")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        return self


class AppConfig(BaseModel):
    """General application configuration."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    chunk_target_chars: int = 12000
    chunk_overlap_chars: int = 1000
    log_level: str = "INFO"
    intermediate_json: bool = True
    save_intermediate_files: bool = True
    prompt_overrides_path: str | None = None
    log_file: str = str(DEFAULT_LOG_FILE)
    min_chunk_target_chars: int = 3000
    max_chunk_reduce_attempts: int = 3
    merge_batch_target_chars: int = 16000
    max_degrade_attempts_per_provider: int = 2
    retry_backoff_seconds: float = 1.5
    default_language: str = DEFAULT_LANGUAGE

    @model_validator(mode="after")
    def validate_ranges(self) -> "AppConfig":
        if self.chunk_target_chars <= 0:
            raise ValueError("chunk_target_chars must be positive.")
        if self.chunk_overlap_chars < 0:
            raise ValueError("chunk_overlap_chars cannot be negative.")
        if self.min_chunk_target_chars <= 0:
            raise ValueError("min_chunk_target_chars must be positive.")
        if self.min_chunk_target_chars > self.chunk_target_chars:
            raise ValueError("min_chunk_target_chars cannot exceed chunk_target_chars.")
        if self.max_chunk_reduce_attempts < 0:
            raise ValueError("max_chunk_reduce_attempts cannot be negative.")
        if self.merge_batch_target_chars <= 0:
            raise ValueError("merge_batch_target_chars must be positive.")
        return self


class ProjectConfig(BaseModel):
    """Top-level YAML config."""

    model_config = ConfigDict(extra="ignore")

    providers: list[ProviderConfig]
    app: AppConfig = Field(default_factory=AppConfig)

    @model_validator(mode="after")
    def validate_providers(self) -> "ProjectConfig":
        enabled = [provider for provider in self.providers if provider.enabled]
        if not enabled:
            raise ValueError("At least one provider must be enabled.")
        provider_names = [provider.name for provider in enabled]
        if len(provider_names) != len(set(provider_names)):
            raise ValueError("Provider names must be unique.")
        self.providers = sorted(enabled, key=lambda item: item.priority)
        return self

    def provider_by_name(self, name: str) -> ProviderConfig:
        for provider in self.providers:
            if provider.name == name:
                return provider
        raise KeyError(name)


class LLMRequest(BaseModel):
    """Prompt request handed to a concrete provider."""

    model_config = ConfigDict(extra="ignore")

    stage: StageName
    system_prompt: str
    user_prompt: str
    expected_schema_name: str
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def with_updates(self, **updates: Any) -> "LLMRequest":
        return self.model_copy(update=updates)


class LLMResponse(BaseModel):
    """Raw provider response before structured validation."""

    model_config = ConfigDict(extra="ignore")

    provider_name: str
    model_name: str
    text: str
    raw_payload: dict[str, Any] | list[Any] | None = None


class ClassifiedError(BaseModel):
    """Classification result used by retry/fallback policy."""

    model_config = ConfigDict(extra="ignore")

    category: ErrorCategory
    message: str
    transient: bool = False


class FallbackDecision(BaseModel):
    """Retry/fallback/degrade policy decision."""

    model_config = ConfigDict(extra="ignore")

    action: str
    reason: str


class MarkdownMetadata(BaseModel):
    """Metadata rendered into YAML frontmatter."""

    model_config = ConfigDict(extra="ignore")

    title: str
    source_type: str = "transcript"
    language: str = DEFAULT_LANGUAGE
    date_created: str
    status: str = "готово"


class PipelineResult(BaseModel):
    """Result of a full summarize command."""

    model_config = ConfigDict(extra="ignore")

    input_path: Path
    output_path: Path
    chunk_count: int
    artifact_dir: Path | None = None


class ProviderSmokeTestResponse(BaseModel):
    """Structured response used by test-provider CLI command."""

    model_config = ConfigDict(extra="ignore")

    status: str
    provider: str


CONFIG_MODEL_PATH = DEFAULT_CONFIG_PATH
