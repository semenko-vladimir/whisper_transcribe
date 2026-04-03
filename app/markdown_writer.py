from __future__ import annotations

import json
from pathlib import Path

from .errors import MarkdownRenderError
from .models import MarkdownMetadata, StructuredSummary


def _yaml_line(key: str, value: str) -> str:
    return f"{key}: {json.dumps(value, ensure_ascii=False)}"


def render_markdown(summary: StructuredSummary, metadata: MarkdownMetadata) -> str:
    """Convert structured summary JSON into an Obsidian-ready markdown note."""

    lines: list[str] = [
        "---",
        _yaml_line("title", metadata.title),
        _yaml_line("source_type", metadata.source_type),
        _yaml_line("language", metadata.language),
        _yaml_line("date_created", metadata.date_created),
        _yaml_line("status", metadata.status),
        "---",
        "",
        "# О чём материал в целом",
        "",
        summary.overview.strip(),
        "",
        "# Конспект по смысловым блокам",
        "",
    ]

    for section in summary.sections:
        lines.extend(
            [
                f"## [{section.start_timecode}–{section.end_timecode}] {section.title}",
                "",
                "**Основная мысль:**  ",
                section.main_point.strip(),
                "",
                "**Ключевые тезисы:**  ",
            ]
        )
        if section.bullet_points:
            lines.extend(f"- {bullet}" for bullet in section.bullet_points)
        else:
            lines.append("- [неуверенно] В этом фрагменте не удалось выделить стабильные тезисы.")

        lines.extend(
            [
                "",
                "**Вывод:**  ",
                section.conclusion.strip(),
                "",
            ]
        )

    lines.extend(
        [
            "# Главные идеи",
            "",
        ]
    )
    lines.extend(f"- {idea}" for idea in summary.key_ideas)
    lines.extend(
        [
            "",
            "# Практические выводы",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary.practical_takeaways)
    lines.append("")

    return "\n".join(lines)


def write_markdown(summary: StructuredSummary, metadata: MarkdownMetadata, output_path: str | Path) -> Path:
    """Render and save markdown to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        path.write_text(render_markdown(summary, metadata), encoding="utf-8")
    except OSError as exc:
        raise MarkdownRenderError(f"Failed to write markdown output: {exc}") from exc

    return path
