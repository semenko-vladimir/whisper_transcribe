from __future__ import annotations

from app.models import TranscriptDocument, TranscriptSegment
from app.transcript_loader import chunk_transcript


def test_chunking_creates_multiple_chunks_with_overlap():
    document = TranscriptDocument(
        segments=[
            TranscriptSegment(start=0, end=5, text="Первая мысль."),
            TranscriptSegment(start=5, end=10, text="Вторая мысль длиннее."),
            TranscriptSegment(start=10, end=15, text="Третья мысль тоже важна."),
            TranscriptSegment(start=15, end=20, text="Четвертая мысль завершает блок."),
        ]
    )

    chunks = chunk_transcript(document, target_chars=70, overlap_chars=20)

    assert len(chunks) >= 2
    assert "Вторая мысль" in chunks[0].text
    assert "Вторая мысль" in chunks[1].text
