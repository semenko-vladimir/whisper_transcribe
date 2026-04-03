from __future__ import annotations

from app.markdown_writer import render_markdown
from app.models import MarkdownMetadata, StructuredSummary, SummarySection


def test_markdown_rendering_contains_expected_sections():
    summary = StructuredSummary(
        title="Материал о продуктивности",
        overview="Это общий обзор материала.",
        sections=[
            SummarySection(
                title="Вступление",
                start_timecode="00:00",
                end_timecode="02:34",
                main_point="Автор объясняет контекст.",
                bullet_points=["Первый тезис", "Второй тезис"],
                conclusion="Нужно смотреть на систему целиком.",
            )
        ],
        key_ideas=["Система важнее мотивации"],
        practical_takeaways=["Выделить повторяемый процесс"],
    )
    metadata = MarkdownMetadata(
        title=summary.title,
        source_type="transcript",
        language="ru",
        date_created="2026-04-03",
        status="готово",
    )

    markdown = render_markdown(summary, metadata)

    assert "---" in markdown
    assert "# О чём материал в целом" in markdown
    assert "## [00:00–02:34] Вступление" in markdown
    assert "# Главные идеи" in markdown
