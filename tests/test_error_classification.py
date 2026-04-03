from __future__ import annotations

from app.constants import ErrorCategory
from app.errors import ProviderError
from app.fallback import ErrorClassifier


def test_error_classifier_detects_quota_errors():
    classifier = ErrorClassifier()

    classified = classifier.classify(
        ProviderError("Resource exhausted because quota exceeded.")
    )

    assert classified.category == ErrorCategory.QUOTA_EXHAUSTED


def test_error_classifier_detects_context_issues():
    classifier = ErrorClassifier()

    classified = classifier.classify(Exception("maximum context length exceeded"))

    assert classified.category == ErrorCategory.CONTEXT_TOO_LONG
