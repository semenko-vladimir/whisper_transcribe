from __future__ import annotations

import argparse
import json
import sys

from .config import load_project_config, provider_environment_status
from .constants import DEFAULT_CONFIG_PATH, StageName
from .errors import AppError, ConfigError
from .fallback import ErrorClassifier, FallbackPolicy, ProviderFallbackExecutor, RetryPolicy
from .logging_utils import setup_logging
from .models import LLMRequest, ProviderSmokeTestResponse
from .pipeline import SummarizationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize transcript JSON into Obsidian markdown.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="Create markdown summary from transcript JSON.")
    summarize.add_argument("input_json", help="Path to transcript JSON.")
    summarize.add_argument("output_md", help="Path to output markdown file.")

    subparsers.add_parser("validate-config", help="Validate YAML configuration and print provider readiness.")
    subparsers.add_parser("list-providers", help="List configured providers in priority order.")

    dry_run = subparsers.add_parser("dry-run", help="Load transcript and show chunking statistics.")
    dry_run.add_argument("input_json", help="Path to transcript JSON.")

    test_provider = subparsers.add_parser("test-provider", help="Run a JSON smoke test against one provider.")
    test_provider.add_argument("--name", required=True, help="Provider name from config/models.yaml.")

    return parser


def command_summarize(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    logger = setup_logging(config.app.log_level, config.app.log_file)
    pipeline = SummarizationPipeline(config, logger)
    result = pipeline.summarize(args.input_json, args.output_md)
    print(f"Markdown saved: {result.output_path}")
    if result.artifact_dir is not None:
        print(f"Artifacts saved: {result.artifact_dir}")
    return 0


def command_validate_config(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    print("Config is valid.")
    for name, env_name, ready in provider_environment_status(config):
        status = "ready" if ready else "missing"
        print(f"- {name}: env={env_name} status={status}")
    return 0


def command_list_providers(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    for provider in config.providers:
        print(
            f"- priority={provider.priority} name={provider.name} provider={provider.provider.value} "
            f"model={provider.model} base_url={provider.base_url or '-'}"
        )
    return 0


def command_dry_run(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    logger = setup_logging(config.app.log_level, config.app.log_file)
    pipeline = SummarizationPipeline(config, logger)
    stats = pipeline.dry_run(args.input_json)
    print(
        f"segments={stats['segments']} chunks={stats['chunks']} "
        f"target_chars={stats['chunk_target_chars']} overlap_chars={stats['chunk_overlap_chars']}"
    )
    return 0


def command_test_provider(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    logger = setup_logging(config.app.log_level, config.app.log_file)
    executor = ProviderFallbackExecutor(
        config.providers,
        logger=logger,
        retry_policy=RetryPolicy(base_delay_seconds=config.app.retry_backoff_seconds),
        fallback_policy=FallbackPolicy(max_degrade_attempts=config.app.max_degrade_attempts_per_provider),
        error_classifier=ErrorClassifier(),
    )
    request = LLMRequest(
        stage=StageName.PROVIDER_TEST,
        system_prompt="Верни только JSON.",
        user_prompt=(
            f'Верни JSON-объект {{"status":"ok","provider":"{args.name}"}} '
            "без комментариев и без Markdown."
        ),
        expected_schema_name="ProviderSmokeTestResponse",
        max_output_tokens=128,
    )
    response, attempts = executor.generate_structured(
        request,
        ProviderSmokeTestResponse,
        provider_name=args.name,
    )
    print(f"Provider test OK: {json.dumps(response.model_dump(), ensure_ascii=False)}")
    print(f"Attempts: {len(attempts)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "summarize":
            return command_summarize(args)
        if args.command == "validate-config":
            return command_validate_config(args)
        if args.command == "list-providers":
            return command_list_providers(args)
        if args.command == "dry-run":
            return command_dry_run(args)
        if args.command == "test-provider":
            return command_test_provider(args)
    except (AppError, ConfigError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
