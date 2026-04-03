from __future__ import annotations

from ..constants import ProviderKind
from ..models import ProviderConfig
from .base import BaseProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider


def build_provider(config: ProviderConfig) -> BaseProvider:
    """Instantiate a provider implementation from YAML config."""

    if config.provider == ProviderKind.GEMINI:
        return GeminiProvider(config)
    if config.provider == ProviderKind.OPENAI_COMPATIBLE:
        return OpenAICompatibleProvider(config)
    if config.provider == ProviderKind.OLLAMA:
        return OllamaProvider(config)
    raise ValueError(f"Unsupported provider kind: {config.provider}")


__all__ = [
    "BaseProvider",
    "GeminiProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "build_provider",
]
