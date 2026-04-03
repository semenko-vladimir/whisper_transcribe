from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import TIME_RANGE_SEPARATOR
from .errors import TranscriptLoadError
from .models import TranscriptChunk, TranscriptDocument, TranscriptSegment


def clean_transcript_text(value: str) -> str:
    """Normalize whitespace while keeping content readable for prompts."""

    return " ".join(value.split())


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS with minute overflow allowed."""

    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02}:{secs:02}"


def format_time_range(start: float, end: float) -> str:
    """Render a time range in the requested Obsidian-friendly format."""

    return f"[{format_timestamp(start)}{TIME_RANGE_SEPARATOR}{format_timestamp(end)}]"


def _normalize_entries(items: list[dict[str, Any]], source: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        raw_text = clean_transcript_text(str(item.get("text", "")))
        if not raw_text:
            continue

        start = float(item.get("start", 0.0) or 0.0)
        end = float(item.get("end", start) or start)
        segments.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=raw_text,
                source=source,
            )
        )

    return segments


def load_transcript(input_path: str | Path) -> TranscriptDocument:
    """Load transcript JSON and normalize it into a predictable document model."""

    path = Path(input_path)
    if not path.exists():
        raise TranscriptLoadError(f"Transcript file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranscriptLoadError(f"Transcript JSON is invalid: {exc}") from exc
    except OSError as exc:
        raise TranscriptLoadError(f"Failed to read transcript file: {exc}") from exc

    metadata: dict[str, Any] = {}
    segments: list[TranscriptSegment] = []

    if isinstance(raw, dict):
        raw_segments = raw.get("segments")
        raw_blocks = raw.get("blocks")

        if isinstance(raw_segments, list) and raw_segments:
            segments = _normalize_entries(raw_segments, "segment")
        elif isinstance(raw_blocks, list) and raw_blocks:
            segments = _normalize_entries(raw_blocks, "block")
        else:
            raise TranscriptLoadError("Transcript JSON must contain a non-empty `segments` or `blocks` array.")

        metadata = {key: value for key, value in raw.items() if key not in {"segments", "blocks"}}
        return TranscriptDocument(
            input_source=raw.get("input_source"),
            original_input=raw.get("original_input"),
            source_type=raw.get("source_type"),
            detected_language=raw.get("detected_language"),
            segments=segments,
            metadata=metadata,
        )

    if isinstance(raw, list):
        dict_items = [item for item in raw if isinstance(item, dict)]
        if not dict_items:
            raise TranscriptLoadError("Transcript JSON array is empty or has unsupported item types.")
        segments = _normalize_entries(dict_items, "segment")
        return TranscriptDocument(segments=segments, metadata={})

    raise TranscriptLoadError("Transcript JSON root must be an object or an array.")


def chunk_transcript(
    document: TranscriptDocument,
    *,
    target_chars: int,
    overlap_chars: int,
) -> list[TranscriptChunk]:
    """Split transcript segments into prompt-friendly chunks with overlapping context."""

    if target_chars <= 0:
        raise ValueError("target_chars must be positive.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative.")
    if not document.segments:
        return []

    chunks: list[TranscriptChunk] = []
    segments = document.segments
    start_index = 0
    chunk_id = 1

    while start_index < len(segments):
        end_index = start_index
        char_count = 0
        lines: list[str] = []

        while end_index < len(segments):
            segment = segments[end_index]
            line = f"{format_time_range(segment.start, segment.end)} {segment.text}"
            prospective = char_count + len(line) + (1 if lines else 0)
            if lines and prospective > target_chars:
                break

            lines.append(line)
            char_count = prospective
            end_index += 1

            if char_count >= target_chars:
                break

        if not lines:
            segment = segments[start_index]
            lines.append(f"{format_time_range(segment.start, segment.end)} {segment.text}")
            end_index = start_index + 1
            char_count = len(lines[0])

        chunk_segments = segments[start_index:end_index]
        chunks.append(
            TranscriptChunk(
                chunk_id=chunk_id,
                start=chunk_segments[0].start,
                end=chunk_segments[-1].end,
                start_timecode=format_timestamp(chunk_segments[0].start),
                end_timecode=format_timestamp(chunk_segments[-1].end),
                text="\n".join(lines),
                segment_count=len(chunk_segments),
                character_count=char_count,
            )
        )
        chunk_id += 1

        if end_index >= len(segments):
            break

        overlap_start = end_index
        overlap_count = 0
        while overlap_start > start_index + 1 and overlap_count < overlap_chars:
            overlap_start -= 1
            overlap_count += len(segments[overlap_start].text)

        start_index = overlap_start if overlap_start < end_index else end_index

    return chunks
