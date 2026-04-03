from __future__ import annotations

import json
import uuid
from pathlib import Path

from app.transcript_loader import load_transcript


def _write_local_test_json(name: str, payload: dict) -> Path:
    temp_dir = Path("tests/.tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{name}_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_loader_prefers_segments():
    source = {
        "input_source": "demo",
        "segments": [
            {"start": 0, "end": 5, "text": " Привет   мир "},
        ],
        "blocks": [
            {"start": 0, "end": 10, "text": "Не должно использоваться"},
        ],
    }
    path = _write_local_test_json("input", source)
    try:
        document = load_transcript(path)
        assert len(document.segments) == 1
        assert document.segments[0].text == "Привет мир"
        assert document.input_source == "demo"
    finally:
        path.unlink(missing_ok=True)


def test_loader_uses_blocks_when_segments_missing():
    source = {
        "blocks": [
            {"start": 10, "end": 20, "text": "Первый блок"},
            {"start": 20, "end": 30, "text": "Второй блок"},
        ]
    }
    path = _write_local_test_json("blocks", source)
    try:
        document = load_transcript(path)
        assert len(document.segments) == 2
        assert document.segments[0].source == "block"
    finally:
        path.unlink(missing_ok=True)
