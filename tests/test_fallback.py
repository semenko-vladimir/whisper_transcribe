from __future__ import annotations

from app.constants import ErrorCategory, StageName
from app.errors import ProviderError
from app.fallback import ErrorClassifier, FallbackPolicy, ProviderFallbackExecutor, RetryPolicy
from app.models import LLMRequest, LLMResponse, ProviderConfig, StructuredSummary


class FakeProvider:
    def __init__(self, config, plan):
        self.config = config
        self.plan = plan

    def generate(self, request):
        step = self.plan[self.config.name].pop(0)
        if isinstance(step, Exception):
            raise step
        return LLMResponse(
            provider_name=self.config.name,
            model_name=self.config.model,
            text=step,
            raw_payload=None,
        )


def test_fallback_moves_to_next_provider(monkeypatch):
    providers = [
        ProviderConfig(
            name="primary",
            provider="ollama",
            base_url="http://localhost:11434",
            model="model-a",
            priority=1,
            max_retries=0,
        ),
        ProviderConfig(
            name="backup",
            provider="ollama",
            base_url="http://localhost:11434",
            model="model-b",
            priority=2,
            max_retries=0,
        ),
    ]
    plan = {
        "primary": [
            ProviderError("quota exhausted", category=ErrorCategory.QUOTA_EXHAUSTED),
        ],
        "backup": [
            '{"title":"Итог","overview":"Кратко","sections":[],"key_ideas":["A"],"practical_takeaways":["B"]}',
        ],
    }

    monkeypatch.setattr(
        "app.fallback.build_provider",
        lambda config: FakeProvider(config, plan),
    )

    executor = ProviderFallbackExecutor(
        providers,
        logger=__import__("logging").getLogger("test"),
        retry_policy=RetryPolicy(base_delay_seconds=0),
        fallback_policy=FallbackPolicy(max_degrade_attempts=1),
        error_classifier=ErrorClassifier(),
    )
    request = LLMRequest(
        stage=StageName.CHUNK_SUMMARY,
        system_prompt="system",
        user_prompt="user",
        expected_schema_name="StructuredSummary",
    )

    result, attempts = executor.generate_structured(request, StructuredSummary)

    assert result.title == "Итог"
    assert len(attempts) == 2


def test_retry_policy_retries_transient_errors(monkeypatch):
    providers = [
        ProviderConfig(
            name="only",
            provider="ollama",
            base_url="http://localhost:11434",
            model="model-a",
            priority=1,
            max_retries=1,
        ),
    ]
    plan = {
        "only": [
            ProviderError("timeout", category=ErrorCategory.TIMEOUT, transient=True),
            '{"title":"Итог","overview":"Кратко","sections":[],"key_ideas":[],"practical_takeaways":[]}',
        ]
    }

    monkeypatch.setattr(
        "app.fallback.build_provider",
        lambda config: FakeProvider(config, plan),
    )

    executor = ProviderFallbackExecutor(
        providers,
        logger=__import__("logging").getLogger("test"),
        retry_policy=RetryPolicy(base_delay_seconds=0),
        fallback_policy=FallbackPolicy(max_degrade_attempts=1),
        error_classifier=ErrorClassifier(),
    )
    request = LLMRequest(
        stage=StageName.CHUNK_SUMMARY,
        system_prompt="system",
        user_prompt="user",
        expected_schema_name="StructuredSummary",
    )

    result, attempts = executor.generate_structured(request, StructuredSummary)

    assert result.title == "Итог"
    assert len(attempts) == 2
