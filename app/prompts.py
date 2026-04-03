from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from .models import ChunkSummaryArtifact, StructuredSummary, TranscriptChunk

SCHEMA_SNIPPET = """
{
  "title": "string",
  "overview": "string",
  "sections": [
    {
      "title": "string",
      "start_timecode": "MM:SS",
      "end_timecode": "MM:SS",
      "main_point": "string",
      "bullet_points": ["string", "string"],
      "conclusion": "string"
    }
  ],
  "key_ideas": ["string"],
  "practical_takeaways": ["string"]
}
""".strip()

DEFAULT_SYSTEM_PROMPT = """
Ты профессиональный редактор и аналитик, который превращает расшифровки речи в качественные русскоязычные конспекты для Obsidian.

Требования:
- Пиши по-русски.
- Не фантазируй и не добавляй факты, которых нет в исходном тексте.
- Сохраняй смысл автора и причинно-следственные связи.
- Убирай словесный мусор, повторы, междометия и устные сбивки.
- Превращай устную речь в чистую письменную форму.
- Если фрагмент испорчен, явно помечай сомнительные формулировки как [неуверенно].
- Возвращай только корректный JSON без Markdown, комментариев и пояснений.
""".strip()

DEFAULT_CHUNK_TEMPLATE = """
Ниже фрагмент транскрипта с таймкодами. Сконспектируй его в структурированный JSON.

Фрагмент: chunk #{chunk_id}
Диапазон чанка: [{start_timecode}–{end_timecode}]

Формат JSON:
{schema}

Правила:
- Поле `title` должно кратко отражать центральную тему этого чанка.
- Поле `overview` должно быть коротким связным абзацем на 2-4 предложения.
- В `sections` выдели смысловые блоки именно этого чанка.
- Для каждого блока укажи `start_timecode` и `end_timecode` по имеющимся таймкодам.
- `bullet_points` должны содержать только конкретные тезисы из текста.
- `conclusion` кратко фиксирует вывод или промежуточный итог блока.

Транскрипт:
{transcript}
""".strip()

DEFAULT_FINAL_MERGE_TEMPLATE = """
Ниже находятся промежуточные JSON-конспекты по частям одного материала. Выполни финальный merge и синтезируй единый качественный структурированный конспект в JSON.

Формат JSON:
{schema}

Правила:
- `title` должен отражать тему материала в целом.
- `overview` должен быть кратким summary всего материала.
- `sections` должны идти в хронологическом порядке.
- Объединяй дублирующиеся тезисы и не повторяй одно и то же разными словами.
- Сохраняй таймкоды и диапазоны блоков.
- `key_ideas` должны содержать главные идеи материала.
- `practical_takeaways` должны содержать прикладные выводы или действия, если они есть в тексте.
- Если какой-то фрагмент исходного материала испорчен, помечай соответствующее место как [неуверенно].

Промежуточные конспекты:
{chunk_summaries}
""".strip()


@dataclass(slots=True)
class PromptTemplates:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    chunk_template: str = DEFAULT_CHUNK_TEMPLATE
    final_merge_template: str = DEFAULT_FINAL_MERGE_TEMPLATE


def load_prompt_templates(path: str | Path | None) -> PromptTemplates:
    """Load optional prompt overrides from YAML."""

    if path is None:
        return PromptTemplates()

    prompt_path = Path(path)
    if not prompt_path.exists():
        return PromptTemplates()

    raw = yaml.safe_load(prompt_path.read_text(encoding="utf-8")) or {}
    return PromptTemplates(
        system_prompt=str(raw.get("system_prompt") or DEFAULT_SYSTEM_PROMPT),
        chunk_template=str(raw.get("chunk_template") or DEFAULT_CHUNK_TEMPLATE),
        final_merge_template=str(raw.get("final_merge_template") or DEFAULT_FINAL_MERGE_TEMPLATE),
    )


def build_chunk_prompt(chunk: TranscriptChunk, templates: PromptTemplates) -> str:
    """Render the stage-one prompt for a single transcript chunk."""

    return templates.chunk_template.format(
        chunk_id=chunk.chunk_id,
        start_timecode=chunk.start_timecode,
        end_timecode=chunk.end_timecode,
        transcript=chunk.text,
        schema=SCHEMA_SNIPPET,
    )


def _serialize_summary(summary: StructuredSummary, index: int) -> str:
    sections = []
    for section in summary.sections:
        bullets = "\n".join(f"    - {bullet}" for bullet in section.bullet_points)
        sections.append(
            f"  - title: {section.title}\n"
            f"    range: [{section.start_timecode}–{section.end_timecode}]\n"
            f"    main_point: {section.main_point}\n"
            f"    bullet_points:\n{bullets or '    - (нет тезисов)'}\n"
            f"    conclusion: {section.conclusion}"
        )

    key_ideas = "\n".join(f"  - {item}" for item in summary.key_ideas) or "  - (нет данных)"
    takeaways = "\n".join(f"  - {item}" for item in summary.practical_takeaways) or "  - (нет данных)"
    section_blob = "\n".join(sections) if sections else "  - (нет секций)"

    return (
        f"Chunk summary #{index}\n"
        f"title: {summary.title}\n"
        f"overview: {summary.overview}\n"
        f"sections:\n{section_blob}\n"
        f"key_ideas:\n{key_ideas}\n"
        f"practical_takeaways:\n{takeaways}"
    )


def build_final_merge_prompt(
    summaries: Sequence[ChunkSummaryArtifact | StructuredSummary],
    templates: PromptTemplates,
) -> str:
    """Render the final merge prompt for stage two."""

    serialized_items: list[str] = []
    for index, item in enumerate(summaries, start=1):
        summary = item.summary if isinstance(item, ChunkSummaryArtifact) else item
        serialized_items.append(_serialize_summary(summary, index))

    return templates.final_merge_template.format(
        schema=SCHEMA_SNIPPET,
        chunk_summaries="\n\n".join(serialized_items),
    )
