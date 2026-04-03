from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .constants import DEFAULT_CONFIG_PATH
from .errors import ConfigError
from .models import ProjectConfig

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional until dependency installation
    load_dotenv = None


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    return path if path.is_absolute() else (Path.cwd() / path)


def load_environment() -> None:
    """Load .env file when python-dotenv is available."""

    if load_dotenv is not None:
        load_dotenv()


def load_project_config(config_path: str | Path | None = None) -> ProjectConfig:
    """Load and validate YAML configuration."""

    load_environment()
    path = resolve_config_path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw_data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed to read config file: {exc}") from exc

    try:
        return ProjectConfig.model_validate(raw_data)
    except Exception as exc:
        raise ConfigError(f"Invalid config schema: {exc}") from exc


def provider_environment_status(config: ProjectConfig) -> list[tuple[str, str, bool]]:
    """Return environment variable readiness for each provider."""

    import os

    rows: list[tuple[str, str, bool]] = []
    for provider in config.providers:
        env_name = provider.api_key_env or "-"
        is_ready = True if provider.api_key_env is None else bool(os.getenv(provider.api_key_env))
        rows.append((provider.name, env_name, is_ready))
    return rows
