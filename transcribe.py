from __future__ import annotations

# Dependencies:
# - faster-whisper
# - yt-dlp
# - tkinter (stdlib)
# - ffmpeg (optional external dependency for some YouTube/media cases)

import json
import logging
import os
import queue
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local environment
    WhisperModel = None  # type: ignore[assignment]
    FASTER_WHISPER_IMPORT_ERROR = exc

try:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError as YtDlpDownloadError

    YT_DLP_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local environment
    YoutubeDL = None  # type: ignore[assignment]
    YT_DLP_IMPORT_ERROR = exc

    class YtDlpDownloadError(Exception):
        pass


try:
    from app.config import load_environment as load_app_environment
except Exception:  # pragma: no cover - GUI should still start even if summarizer deps are unavailable
    def load_app_environment() -> None:
        return None


APP_TITLE = "Whisper Transcribe"
MODEL_NAME = "large-v3"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BEAM_SIZE = 5
REQUESTED_LANGUAGE: Optional[str] = None
MIN_SILENCE_DURATION_MS = 2000
TARGET_BLOCK_CHARS = 1200
MAX_BLOCK_CHARS = 1700
BREAK_ON_GAP_SECONDS = 4.0
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_NOTES_DIR = DEFAULT_OUTPUT_DIR / "notes"
SUMMARY_CONFIG_PATH = BASE_DIR / "config" / "models.yaml"
SUMMARY_FILENAME_FALLBACK_PREFIX = "transcript_summary"


load_app_environment()


def resolve_output_dir_from_env(env_name: str, default: Path) -> Path:
    raw_value = os.getenv(env_name)
    if not raw_value or not raw_value.strip():
        return default

    path = Path(os.path.expandvars(os.path.expanduser(raw_value.strip())))
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve(strict=False)


OBSIDIAN_OUTPUT_DIR = resolve_output_dir_from_env("OBSIDIAN_OUTPUT_DIR", DEFAULT_NOTES_DIR)

MEDIA_FILE_TYPES = [
    (
        "Media files",
        "*.aac *.avi *.flac *.m4a *.mkv *.mov *.mp3 *.mp4 *.ogg *.opus *.wav *.webm",
    ),
    ("All files", "*.*"),
]

MEDIA_EXTENSIONS = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}

INVALID_OUTPUT_SUFFIXES = {
    ".description",
    ".gif",
    ".jpeg",
    ".jpg",
    ".json",
    ".part",
    ".png",
    ".srt",
    ".txt",
    ".vtt",
    ".webp",
    ".ytdl",
}

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


class AppError(Exception):
    pass


class ValidationError(AppError):
    pass


class DependencyError(AppError):
    pass


class DownloadFailedError(AppError):
    pass


class TranscriptionFailedError(AppError):
    pass


class SummarizationPartialError(AppError):
    def __init__(self, message: str, json_path: Path) -> None:
        super().__init__(message)
        self.json_path = json_path


@dataclass
class Block:
    start: float
    end: float
    text: str
    segment_count: int


@dataclass
class PreparedSource:
    source_type: str
    original_input: str
    input_source: str
    transcription_path: Path
    output_stem: str
    downloaded_file: Optional[Path] = None
    temp_dir: Optional[tempfile.TemporaryDirectory] = None


@dataclass
class WorkflowResult:
    json_path: Path
    markdown_path: Optional[Path] = None


@dataclass
class ProgressReporter:
    emit: Callable[[str, Any], None]

    def status(self, message: str) -> None:
        self.emit("status", message)

    def log(self, message: str) -> None:
        self.emit("log", message)

    def progress(self, value: float) -> None:
        bounded = max(0.0, min(100.0, value))
        self.emit("progress", bounded)


def to_hms(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def normalize_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None

    normalized = language.strip().lower()
    if normalized in ("", "auto"):
        return None
    return normalized


def clean_segment_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = text.replace(" – ", " — ")
    return text.strip()


def looks_like_sentence_end(text: str) -> bool:
    return text.endswith((".", "!", "?", "…"))


def build_blocks(
    segments: list[Any],
    target_chars: int = TARGET_BLOCK_CHARS,
    max_chars: int = MAX_BLOCK_CHARS,
    break_on_gap_seconds: float = BREAK_ON_GAP_SECONDS,
) -> list[Block]:
    blocks: list[Block] = []

    current_texts: list[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_count = 0
    current_len = 0
    prev_end: Optional[float] = None

    for seg in segments:
        text = clean_segment_text(seg.text)
        if not text:
            continue

        if current_start is None:
            current_start = seg.start

        gap = None if prev_end is None else seg.start - prev_end
        should_flush = False

        if gap is not None and gap >= break_on_gap_seconds and current_texts:
            should_flush = True

        if current_len >= target_chars and current_texts:
            if looks_like_sentence_end(" ".join(current_texts)):
                should_flush = True

        if current_len >= max_chars and current_texts:
            should_flush = True

        if should_flush:
            joined = " ".join(current_texts).strip()
            blocks.append(
                Block(
                    start=current_start if current_start is not None else 0.0,
                    end=current_end if current_end is not None else current_start or 0.0,
                    text=joined,
                    segment_count=current_count,
                )
            )
            current_texts = []
            current_start = seg.start
            current_end = None
            current_count = 0
            current_len = 0

        current_texts.append(text)
        current_end = seg.end
        current_count += 1
        current_len += len(text) + 1
        prev_end = seg.end

    if current_texts:
        joined = " ".join(current_texts).strip()
        blocks.append(
            Block(
                start=current_start if current_start is not None else 0.0,
                end=current_end if current_end is not None else current_start or 0.0,
                text=joined,
                segment_count=current_count,
            )
        )

    return blocks


def strip_wrapping_quotes(value: str) -> str:
    cleaned = value.strip().replace("\r", " ").replace("\n", " ")
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def normalize_user_input(value: str) -> str:
    return strip_wrapping_quotes(value)


def looks_like_url(value: str) -> bool:
    lowered = value.lower()
    if re.match(r"^[a-z][a-z0-9+.-]*://", lowered):
        return True
    return lowered.startswith(
        (
            "www.",
            "youtube.com/",
            "m.youtube.com/",
            "music.youtube.com/",
            "youtu.be/",
        )
    )


def is_youtube_url(value: str) -> bool:
    candidate = value if re.match(r"^[a-z][a-z0-9+.-]*://", value.lower()) else f"https://{value}"
    parsed = urlparse(candidate)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host in {"youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}


def normalize_local_path(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve(strict=False)
    else:
        path = path.resolve(strict=False)
    return path


def sanitize_filename(value: str, fallback: str = "transcript") -> str:
    cleaned = strip_wrapping_quotes(value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = INVALID_FILENAME_CHARS.sub("_", cleaned)
    cleaned = cleaned.rstrip(" .")
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    if not cleaned or not cleaned.strip(" ._"):
        cleaned = fallback

    if cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_file"

    cleaned = cleaned[:120].rstrip(" .")
    return cleaned or fallback


def build_output_path(base_name: str) -> Path:
    return build_unique_output_path(
        DEFAULT_OUTPUT_DIR,
        base_name,
        ".json",
        fallback_prefix="transcript",
    )


def build_unique_output_path(
    directory: Path,
    base_name: str,
    suffix: str,
    *,
    fallback_prefix: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(base_name, fallback="")
    if not safe_name:
        safe_name = f"{fallback_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    candidate = directory / f"{safe_name}{suffix}"
    index = 2

    while candidate.exists():
        candidate = directory / f"{safe_name} ({index}){suffix}"
        index += 1

    return candidate


def pick_downloaded_media_file(directory: Path) -> Path:
    candidates: list[Path] = []

    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() in INVALID_OUTPUT_SUFFIXES:
            continue
        candidates.append(path)

    if not candidates:
        raise DownloadFailedError("YouTube download failed: не удалось найти скачанный медиафайл.")

    candidates.sort(key=lambda item: (item.stat().st_size, item.stat().st_mtime), reverse=True)
    return candidates[0]


def save_transcript_json(
    path: Path,
    source: PreparedSource,
    model_name: str,
    requested_language: Optional[str],
    detected_language: str,
    language_probability: float,
    segments: list[Any],
    blocks: list[Block],
) -> None:
    data = {
        "input_source": source.input_source,
        "source_type": source.source_type,
        "original_input": source.original_input,
        "downloaded_file": str(source.downloaded_file) if source.downloaded_file else None,
        "model": model_name,
        "requested_language": requested_language,
        "detected_language": detected_language,
        "language_probability": language_probability,
        "blocks": [
            {
                "block_id": index + 1,
                "start": block.start,
                "end": block.end,
                "segment_count": block.segment_count,
                "text": block.text,
            }
            for index, block in enumerate(blocks)
        ],
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": clean_segment_text(segment.text),
            }
            for segment in segments
            if clean_segment_text(segment.text)
        ],
    }

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def download_youtube_media(
    url: str,
    reporter: ProgressReporter,
) -> tuple[Path, str, tempfile.TemporaryDirectory]:
    if YoutubeDL is None:
        raise DependencyError(
            "yt-dlp недоступен. Установите пакет `yt-dlp`, чтобы работать с YouTube-ссылками."
        )

    temp_dir = tempfile.TemporaryDirectory(prefix="whisper_transcribe_")
    temp_path = Path(temp_dir.name)
    last_percent = -1

    def progress_hook(data: dict[str, Any]) -> None:
        nonlocal last_percent

        status = data.get("status")
        if status == "downloading":
            total_bytes = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            downloaded_bytes = data.get("downloaded_bytes") or 0
            if total_bytes:
                percent = int(downloaded_bytes * 100 / total_bytes)
                if percent > last_percent:
                    last_percent = percent
                    reporter.progress(5 + percent * 0.30)
                    reporter.status(f"Скачивание YouTube: {percent}%")
            elif last_percent < 0:
                last_percent = 0
                reporter.progress(10)
                reporter.status("Скачиваю аудио из YouTube...")
        elif status == "finished":
            reporter.progress(35)
            reporter.status("Скачивание завершено, подготавливаю файл...")

    ydl_options = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": str(temp_path / "%(title).200B [%(id)s].%(ext)s"),
        "windowsfilenames": True,
        "progress_hooks": [progress_hook],
    }

    try:
        reporter.status("Проверяю YouTube-ссылку...")
        reporter.log(f"YouTube URL: {url}")

        with YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(url, download=True)

        if info is None:
            raise DownloadFailedError("YouTube download failed: yt-dlp не вернул данные о видео.")

        if isinstance(info, dict) and info.get("entries"):
            first_entry = next((entry for entry in info["entries"] if entry), None)
            if first_entry is not None:
                info = first_entry

        downloaded_file = pick_downloaded_media_file(temp_path)
        title = ""
        if isinstance(info, dict):
            title = str(info.get("title") or info.get("id") or "")

        if not title:
            title = downloaded_file.stem

        reporter.log(f"Скачанный файл: {downloaded_file}")
        return downloaded_file, title, temp_dir
    except YtDlpDownloadError as exc:
        temp_dir.cleanup()
        message = str(exc)
        lowered = message.lower()
        if "ffmpeg" in lowered or "ffprobe" in lowered:
            raise DependencyError(
                "yt-dlp сообщил, что не найден ffmpeg/ffprobe. Установите ffmpeg и добавьте его в PATH."
            ) from exc
        raise DownloadFailedError(f"YouTube download failed: {message}") from exc
    except AppError:
        temp_dir.cleanup()
        raise
    except Exception as exc:
        temp_dir.cleanup()
        lowered = str(exc).lower()
        if "ffmpeg" in lowered or "ffprobe" in lowered:
            raise DependencyError(
                "Не найден ffmpeg/ffprobe. Установите ffmpeg и добавьте его в PATH."
            ) from exc
        raise DownloadFailedError(f"YouTube download failed: {exc}") from exc


def prepare_source(original_input: str, reporter: ProgressReporter) -> PreparedSource:
    normalized_original = original_input.strip()
    cleaned_input = normalize_user_input(normalized_original)
    if not cleaned_input:
        raise ValidationError("Введите ссылку YouTube или локальный путь к файлу.")

    if looks_like_url(cleaned_input):
        if not is_youtube_url(cleaned_input):
            raise ValidationError("Ссылка не распознана как корректный YouTube URL.")

        reporter.progress(3)
        reporter.status("Подготавливаю YouTube-источник...")
        downloaded_file, title, temp_dir = download_youtube_media(cleaned_input, reporter)
        safe_title = sanitize_filename(title, fallback="youtube_video")
        return PreparedSource(
            source_type="youtube",
            original_input=normalized_original,
            input_source=cleaned_input,
            transcription_path=downloaded_file,
            output_stem=safe_title,
            downloaded_file=downloaded_file,
            temp_dir=temp_dir,
        )

    local_path = normalize_local_path(cleaned_input)
    if not local_path.exists():
        raise ValidationError(f"Файл не найден: {local_path}")
    if not local_path.is_file():
        raise ValidationError(f"Ожидался файл, но указана папка: {local_path}")

    suffix = local_path.suffix.lower()
    if suffix and suffix not in MEDIA_EXTENSIONS:
        reporter.log(
            "Расширение файла не из списка частых медиаформатов, но попробую передать его в faster-whisper как есть."
        )

    reporter.progress(10)
    reporter.status("Локальный файл найден.")
    reporter.log(f"Источник: {local_path}")

    return PreparedSource(
        source_type="local_file",
        original_input=normalized_original,
        input_source=str(local_path),
        transcription_path=local_path,
        output_stem=sanitize_filename(local_path.stem, fallback="local_file"),
    )


class TranscriptionService:
    def __init__(self) -> None:
        self._model: Any = None

    def get_model(self, reporter: ProgressReporter) -> Any:
        if WhisperModel is None:
            raise DependencyError(
                "faster-whisper недоступен. Установите пакет `faster-whisper` перед запуском транскрибации."
            ) from FASTER_WHISPER_IMPORT_ERROR

        if self._model is None:
            reporter.progress(40)
            reporter.status(f"Загружаю модель {MODEL_NAME}...")
            reporter.log(
                f"Настройки модели: model={MODEL_NAME}, device={DEVICE}, "
                f"compute_type={COMPUTE_TYPE}, beam_size={BEAM_SIZE}"
            )
            try:
                self._model = WhisperModel(
                    MODEL_NAME,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                )
            except Exception as exc:
                raise DependencyError(f"Не удалось загрузить faster-whisper: {exc}") from exc
        else:
            reporter.progress(40)
            reporter.status(f"Использую уже загруженную модель {MODEL_NAME}...")

        return self._model

    def transcribe_input(self, raw_input: str, reporter: ProgressReporter) -> Path:
        source: Optional[PreparedSource] = None

        try:
            source = prepare_source(raw_input, reporter)
            model = self.get_model(reporter)
            return self._transcribe_source(model, source, reporter)
        finally:
            if source is not None and source.temp_dir is not None:
                source.temp_dir.cleanup()
                reporter.log("Временные YouTube-файлы удалены.")

    def _transcribe_source(self, model: Any, source: PreparedSource, reporter: ProgressReporter) -> Path:
        requested_language = normalize_language(REQUESTED_LANGUAGE)

        reporter.progress(45)
        reporter.status("Запускаю транскрибацию...")

        try:
            segments_gen, info = model.transcribe(
                str(source.transcription_path),
                beam_size=BEAM_SIZE,
                language=requested_language,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": MIN_SILENCE_DURATION_MS},
            )
        except Exception as exc:
            raise TranscriptionFailedError(f"Whisper завершился с ошибкой: {exc}") from exc

        total_duration = float(getattr(info, "duration", 0.0) or 0.0)
        if total_duration > 0:
            reporter.log(f"Длительность медиа: {to_hms(total_duration)}")

        reporter.status("Транскрибация идёт...")
        segments: list[Any] = []
        last_percent = -1

        try:
            for segment in segments_gen:
                text = clean_segment_text(segment.text)
                if text:
                    segments.append(segment)

                if total_duration > 0:
                    percent = min(100, int((float(segment.end) / total_duration) * 100))
                    if percent > last_percent:
                        last_percent = percent
                        reporter.progress(45 + percent * 0.50)
                        reporter.status(f"Транскрибация: {percent}%")
        except Exception as exc:
            raise TranscriptionFailedError(
                f"Whisper завершился с ошибкой во время обработки сегментов: {exc}"
            ) from exc

        blocks = build_blocks(
            segments,
            target_chars=TARGET_BLOCK_CHARS,
            max_chars=MAX_BLOCK_CHARS,
            break_on_gap_seconds=BREAK_ON_GAP_SECONDS,
        )

        reporter.progress(97)
        reporter.status("Сохраняю JSON...")
        reporter.log(f"Сегментов: {len(segments)} | Блоков: {len(blocks)}")

        output_path = build_output_path(source.output_stem)
        detected_language = str(getattr(info, "language", "") or "unknown")
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)

        save_transcript_json(
            output_path,
            source,
            MODEL_NAME,
            requested_language,
            detected_language,
            language_probability,
            segments,
            blocks,
        )

        reporter.progress(88)
        reporter.status(f"Транскрипт сохранён: {output_path}")
        reporter.log(
            f"Определённый язык: {detected_language} "
            f"(вероятность {language_probability:.3f})"
        )
        reporter.log(f"JSON сохранён: {output_path}")
        return output_path


class GuiReporterHandler(logging.Handler):
    def __init__(self, reporter: ProgressReporter) -> None:
        super().__init__()
        self.reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        if message:
            self.reporter.log(message)


class SummarizationService:
    @staticmethod
    def _prepare_gui_config(config: Any) -> Any:
        """Keep the Obsidian vault clean for the GUI workflow."""

        if not getattr(config.app, "save_intermediate_files", False):
            return config

        app_config = config.app.model_copy(
            update={
                "save_intermediate_files": False,
                "intermediate_json": False,
            }
        )
        return config.model_copy(update={"app": app_config})

    def summarize_transcript(self, json_path: Path, reporter: ProgressReporter) -> Path:
        reporter.progress(90)
        reporter.status("Транскрипт готов. Запускаю конспектирование...")
        reporter.log(f"Входной JSON для конспекта: {json_path}")

        logger: Optional[logging.Logger] = None
        gui_handler: Optional[logging.Handler] = None

        try:
            from app.config import load_project_config
            from app.constants import ErrorCategory as SummaryErrorCategory
            from app.errors import (
                AllProvidersFailedError,
                ConfigError,
                MarkdownRenderError,
                ProviderError,
                SummarizationError,
            )
            from app.logging_utils import setup_logging
            from app.pipeline import SummarizationPipeline
        except Exception as exc:
            raise SummarizationPartialError(
                "Транскрипт сохранён, но модуль конспектирования не удалось загрузить.",
                json_path,
            ) from exc

        try:
            config = load_project_config(SUMMARY_CONFIG_PATH)
            config = self._prepare_gui_config(config)
            logger = setup_logging(config.app.log_level, config.app.log_file)
            gui_handler = GuiReporterHandler(reporter)
            gui_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
            logger.addHandler(gui_handler)

            markdown_path = build_unique_output_path(
                OBSIDIAN_OUTPUT_DIR,
                json_path.stem,
                ".md",
                fallback_prefix=SUMMARY_FILENAME_FALLBACK_PREFIX,
            )
            reporter.log(f"Markdown будет сохранён в: {markdown_path}")

            pipeline = SummarizationPipeline(config, logger)
            result = pipeline.summarize(json_path, markdown_path)

            reporter.progress(100)
            reporter.status(f"Готово. Конспект сохранён: {result.output_path}")
            reporter.log(f"Markdown сохранён: {result.output_path}")
            return result.output_path
        except ConfigError as exc:
            raise SummarizationPartialError(
                f"Транскрипт сохранён, но настройки конспектирования не удалось загрузить: {exc}",
                json_path,
            ) from exc
        except MarkdownRenderError as exc:
            raise SummarizationPartialError(
                f"Транскрипт сохранён, но не удалось записать Markdown в папку Obsidian: {exc}",
                json_path,
            ) from exc
        except ProviderError as exc:
            category = getattr(exc, "category", None)
            message = f"Транскрипт сохранён, но конспект не создан: {exc}"
            if category == SummaryErrorCategory.AUTH_ERROR:
                message = (
                    "Транскрипт сохранён, но конспект не создан: проверьте API-ключи "
                    "GEMINI_API_KEY и OPENROUTER_API_KEY."
                )
            elif category == SummaryErrorCategory.QUOTA_EXHAUSTED:
                message = "Транскрипт сохранён, но конспект не создан: у провайдера закончилась квота."
            elif category == SummaryErrorCategory.RATE_LIMITED:
                message = "Транскрипт сохранён, но конспект не создан: провайдер временно ограничил запросы."
            elif category == SummaryErrorCategory.PROVIDER_UNAVAILABLE:
                message = "Транскрипт сохранён, но конспект не создан: сервис конспектирования временно недоступен."
            raise SummarizationPartialError(message, json_path) from exc
        except AllProvidersFailedError as exc:
            message = "Транскрипт сохранён, но конспект не создан: исчерпаны все fallback-провайдеры."
            if getattr(exc, "last_category", None) == SummaryErrorCategory.AUTH_ERROR:
                message = (
                    "Транскрипт сохранён, но конспект не создан: проверьте API-ключи "
                    "GEMINI_API_KEY и OPENROUTER_API_KEY."
                )
            raise SummarizationPartialError(message, json_path) from exc
        except OSError as exc:
            raise SummarizationPartialError(
                f"Транскрипт сохранён, но не удалось записать Markdown в папку Obsidian: {exc}",
                json_path,
            ) from exc
        except SummarizationError as exc:
            raise SummarizationPartialError(
                f"Транскрипт сохранён, но конспект не создан: {exc}",
                json_path,
            ) from exc
        finally:
            if logger is not None and gui_handler is not None:
                logger.removeHandler(gui_handler)


class TranscribeApp:
    POLL_INTERVAL_MS = 100
    CLIPBOARD_EMPTY_MESSAGE = "Буфер обмена пуст или недоступен."

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("900x560")
        self.root.minsize(760, 480)

        self.service = TranscriptionService()
        self.summarization_service = SummarizationService()
        self.event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.latest_output_path: Optional[Path] = None

        self.input_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Готово к запуску.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.input_context_menu: Optional[tk.Menu] = None

        self._build_ui()
        self._write_startup_notes()
        self.root.after_idle(self.focus_input_entry)
        self.root.after(self.POLL_INTERVAL_MS, self.process_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(4, weight=1)

        ttk.Label(
            container,
            text="YouTube ссылка или локальный путь к аудио/видео",
        ).grid(row=0, column=0, sticky="w")

        input_frame = ttk.Frame(container)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(6, 10))
        input_frame.columnconfigure(0, weight=1)

        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=0, sticky="ew")
        self.input_entry.bind("<Return>", self.handle_entry_return)
        self.input_entry.bind("<Button-1>", self.handle_entry_click, add="+")
        self.input_entry.bind("<Button-3>", self.show_input_context_menu, add="+")
        self.input_entry.bind("<Control-v>", self.handle_paste_shortcut)
        self.input_entry.bind("<Control-V>", self.handle_paste_shortcut)
        self.input_entry.bind("<Shift-Insert>", self.handle_paste_shortcut)
        self.input_entry.bind("<Control-c>", self.handle_copy_shortcut)
        self.input_entry.bind("<Control-C>", self.handle_copy_shortcut)
        self.input_entry.bind("<Control-x>", self.handle_cut_shortcut)
        self.input_entry.bind("<Control-X>", self.handle_cut_shortcut)
        self.input_entry.bind("<Control-a>", self.handle_select_all_shortcut)
        self.input_entry.bind("<Control-A>", self.handle_select_all_shortcut)

        self.paste_button = ttk.Button(
            input_frame,
            text="Вставить",
            command=self.paste_from_button,
        )
        self.paste_button.grid(row=0, column=1, padx=(8, 0))

        self.browse_button = ttk.Button(
            input_frame,
            text="Выбрать файл",
            command=self.choose_local_file,
        )
        self.browse_button.grid(row=0, column=2, padx=(8, 0))

        self.input_context_menu = tk.Menu(self.root, tearoff=0)
        self.input_context_menu.add_command(label="Вырезать", command=self.cut_from_context_menu)
        self.input_context_menu.add_command(label="Копировать", command=self.copy_from_context_menu)
        self.input_context_menu.add_command(label="Вставить", command=self.paste_from_context_menu)
        self.input_context_menu.add_command(label="Выделить всё", command=self.select_all_from_context_menu)
        self.input_context_menu.add_separator()
        self.input_context_menu.add_command(label="Очистить", command=self.clear_input_entry)

        actions_frame = ttk.Frame(container)
        actions_frame.grid(row=2, column=0, sticky="ew")

        self.start_button = ttk.Button(
            actions_frame,
            text="Запустить транскрибацию",
            command=self.start_transcription,
        )
        self.start_button.pack(side=tk.LEFT)

        self.open_folder_button = ttk.Button(
            actions_frame,
            text="Открыть папку с результатом",
            command=self.open_result_folder,
            state=tk.DISABLED,
        )
        self.open_folder_button.pack(side=tk.LEFT, padx=(8, 0))

        self.progress_bar = ttk.Progressbar(
            container,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
        )
        self.progress_bar.grid(row=3, column=0, sticky="ew", pady=(12, 4))

        self.status_label = ttk.Label(container, textvariable=self.status_var, anchor="w")
        self.status_label.grid(row=4, column=0, sticky="new")

        self.log_text = ScrolledText(container, wrap=tk.WORD, height=18, state=tk.DISABLED)
        self.log_text.grid(row=5, column=0, sticky="nsew", pady=(8, 0))
        container.rowconfigure(5, weight=1)

    def _write_startup_notes(self) -> None:
        self.append_log("Окно готово. Вставьте YouTube-ссылку или выберите локальный файл.")
        self.append_log(f"JSON по умолчанию сохраняется в: {DEFAULT_OUTPUT_DIR}")
        self.append_log(f"Итоговый Markdown-конспект сохраняется в: {OBSIDIAN_OUTPUT_DIR}")

        if FASTER_WHISPER_IMPORT_ERROR is not None:
            self.append_log(
                "Внимание: faster-whisper не импортирован. "
                "Установите зависимости перед запуском транскрибации."
            )

        if YT_DLP_IMPORT_ERROR is not None:
            self.append_log(
                "Внимание: yt-dlp не импортирован. "
                "YouTube-ссылки не будут работать, пока пакет не установлен."
            )

        if shutil.which("ffmpeg") is None:
            self.append_log(
                "Примечание: ffmpeg не найден в PATH. "
                "Для локальных файлов это обычно не мешает, но для некоторых YouTube-сценариев он может понадобиться."
            )

    def append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def report_input_notice(self, message: str) -> None:
        self.status_var.set(message)
        self.append_log(message)

    def focus_input_entry(self) -> None:
        if str(self.input_entry.cget("state")) != tk.DISABLED:
            self.input_entry.focus_set()

    def handle_entry_click(self, _event: tk.Event) -> None:
        self.root.after_idle(self.focus_input_entry)

    def handle_entry_return(self, _event: tk.Event) -> str:
        self.start_transcription()
        return "break"

    def handle_paste_shortcut(self, _event: tk.Event) -> str:
        self.paste_clipboard_into_entry()
        return "break"

    def handle_copy_shortcut(self, _event: tk.Event) -> str:
        self.copy_selection()
        return "break"

    def handle_cut_shortcut(self, _event: tk.Event) -> str:
        self.cut_selection()
        return "break"

    def handle_select_all_shortcut(self, _event: tk.Event) -> str:
        self.select_all_input()
        return "break"

    def show_input_context_menu(self, event: tk.Event) -> str:
        self.focus_input_entry()
        try:
            self.input_entry.icursor(f"@{event.x}")
        except tk.TclError:
            pass

        if self.input_context_menu is not None:
            try:
                self.input_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.input_context_menu.grab_release()
        return "break"

    def copy_selection(self) -> None:
        if str(self.input_entry.cget("state")) == tk.DISABLED:
            return
        self.focus_input_entry()
        try:
            self.input_entry.event_generate("<<Copy>>")
        except tk.TclError:
            pass

    def cut_selection(self) -> None:
        if str(self.input_entry.cget("state")) == tk.DISABLED:
            return
        self.focus_input_entry()
        try:
            self.input_entry.event_generate("<<Cut>>")
        except tk.TclError:
            pass

    def select_all_input(self) -> None:
        if str(self.input_entry.cget("state")) == tk.DISABLED:
            return
        self.focus_input_entry()
        self.input_entry.selection_range(0, tk.END)
        self.input_entry.icursor(tk.END)
        self.input_entry.xview_moveto(1.0)

    def clear_input_entry(self) -> None:
        if str(self.input_entry.cget("state")) == tk.DISABLED:
            return
        self.input_entry.delete(0, tk.END)
        self.focus_input_entry()

    def paste_clipboard_into_entry(self) -> bool:
        if str(self.input_entry.cget("state")) == tk.DISABLED:
            return False

        try:
            clipboard_text = self.root.clipboard_get()
        except tk.TclError:
            self.report_input_notice(self.CLIPBOARD_EMPTY_MESSAGE)
            self.focus_input_entry()
            return False

        if clipboard_text == "":
            self.report_input_notice(self.CLIPBOARD_EMPTY_MESSAGE)
            self.focus_input_entry()
            return False

        self.focus_input_entry()

        try:
            selection_start = self.input_entry.index(tk.SEL_FIRST)
            selection_end = self.input_entry.index(tk.SEL_LAST)
            self.input_entry.delete(selection_start, selection_end)
            insert_at = selection_start
        except tk.TclError:
            insert_at = self.input_entry.index(tk.INSERT)

        self.input_entry.insert(insert_at, clipboard_text)
        self.input_entry.icursor(insert_at + len(clipboard_text))
        self.input_entry.selection_clear()
        return True

    def paste_from_button(self) -> None:
        if self.paste_clipboard_into_entry():
            self.report_input_notice("Текст вставлен в поле ввода.")

    def cut_from_context_menu(self) -> None:
        self.cut_selection()
        self.focus_input_entry()

    def copy_from_context_menu(self) -> None:
        self.copy_selection()
        self.focus_input_entry()

    def paste_from_context_menu(self) -> None:
        self.paste_clipboard_into_entry()
        self.focus_input_entry()

    def select_all_from_context_menu(self) -> None:
        self.select_all_input()
        self.focus_input_entry()

    def choose_local_file(self) -> None:
        selected_file = filedialog.askopenfilename(
            title="Выберите аудио- или видеофайл",
            filetypes=MEDIA_FILE_TYPES,
        )
        if selected_file:
            self.input_var.set(selected_file)

    def start_transcription(self) -> None:
        raw_input = self.input_var.get()
        if not raw_input.strip():
            self.report_input_notice("Введите ссылку YouTube или выберите локальный файл.")
            messagebox.showwarning(
                "Нет входных данных",
                "Вставьте ссылку YouTube или выберите локальный файл.",
            )
            self.focus_input_entry()
            return

        if self.worker_thread is not None and self.worker_thread.is_alive():
            return

        self.latest_output_path = None
        self.open_folder_button.config(state=tk.DISABLED)
        self.progress_var.set(0.0)
        self.status_var.set("Запуск задачи...")
        self.append_log("")
        self.append_log("=" * 70)
        self.append_log(f"Новый запуск: {raw_input.strip()}")
        self.set_running_state(True)

        self.worker_thread = threading.Thread(
            target=self.worker_main,
            args=(raw_input,),
            daemon=True,
        )
        self.worker_thread.start()

    def worker_main(self, raw_input: str) -> None:
        reporter = ProgressReporter(self.enqueue_event)

        try:
            json_path = self.service.transcribe_input(raw_input, reporter)
            markdown_path = self.summarization_service.summarize_transcript(json_path, reporter)
        except SummarizationPartialError as exc:
            self.enqueue_event(
                "partial_done",
                {
                    "json_path": str(exc.json_path),
                    "message": str(exc),
                },
            )
        except AppError as exc:
            self.enqueue_event("error", str(exc))
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            self.enqueue_event("error", f"Непредвиденная ошибка: {exc}")
        else:
            self.enqueue_event(
                "done",
                {
                    "json_path": str(json_path),
                    "markdown_path": str(markdown_path),
                },
            )
        finally:
            self.enqueue_event("running", False)

    def enqueue_event(self, event_type: str, payload: Any) -> None:
        self.event_queue.put((event_type, payload))

    def process_queue(self) -> None:
        try:
            while True:
                event_type, payload = self.event_queue.get_nowait()
                if event_type == "status":
                    self.status_var.set(str(payload))
                elif event_type == "log":
                    self.append_log(str(payload))
                elif event_type == "progress":
                    self.progress_var.set(float(payload))
                elif event_type == "error":
                    message = str(payload)
                    self.status_var.set(message)
                    self.append_log(f"Ошибка: {message}")
                    messagebox.showerror("Ошибка", message)
                elif event_type == "partial_done":
                    payload_data = dict(payload)
                    json_path = Path(str(payload_data["json_path"]))
                    message = str(payload_data["message"])
                    self.latest_output_path = json_path
                    self.progress_var.set(100.0)
                    self.status_var.set(message)
                    self.append_log(message)
                    self.append_log(f"JSON сохранён: {json_path}")
                    self.open_folder_button.config(state=tk.NORMAL)
                elif event_type == "done":
                    payload_data = dict(payload)
                    json_path = Path(str(payload_data["json_path"]))
                    markdown_path = Path(str(payload_data["markdown_path"]))
                    self.latest_output_path = markdown_path
                    self.progress_var.set(100.0)
                    self.status_var.set(f"Готово. Транскрипт и конспект сохранены: {markdown_path}")
                    self.append_log(f"JSON сохранён: {json_path}")
                    self.append_log(f"Markdown сохранён: {markdown_path}")
                    self.open_folder_button.config(state=tk.NORMAL)
                elif event_type == "running":
                    self.set_running_state(bool(payload))
        except queue.Empty:
            pass
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(self.POLL_INTERVAL_MS, self.process_queue)
            except tk.TclError:
                pass

    def set_running_state(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.start_button.config(state=state)
        self.browse_button.config(state=state)
        self.paste_button.config(state=state)
        self.input_entry.config(state=state)

        if running:
            self.open_folder_button.config(state=tk.DISABLED)
        elif self.latest_output_path is not None:
            self.open_folder_button.config(state=tk.NORMAL)
        else:
            self.focus_input_entry()

    def open_result_folder(self) -> None:
        target_dir = self.latest_output_path.parent if self.latest_output_path else DEFAULT_OUTPUT_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            os.startfile(str(target_dir))
        except Exception as exc:  # pragma: no cover - Windows specific UI action
            messagebox.showerror("Не удалось открыть папку", str(exc))

    def on_close(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            should_close = messagebox.askyesno(
                "Закрыть окно",
                "Транскрибация ещё выполняется. Закрыть окно и прервать фоновую задачу?",
            )
            if not should_close:
                return

        self.root.destroy()


def main() -> None:
    # Developer smoke-check:
    # 1. Start the app and verify the caret is immediately available in the input field.
    # 2. Paste a long YouTube URL with `Ctrl+V`, `Shift+Insert`, right click -> `Вставить`, and the `Вставить` button.
    # 3. Confirm `Ctrl+A`, `Ctrl+C`, `Ctrl+X`, and right-click actions work without exceptions.
    # 4. Click `Запустить транскрибацию` with an empty field and verify the Russian warning/status message appears.
    # 5. Complete one transcription and confirm that summarization starts automatically and saves `.md` into the Obsidian folder.
    root = tk.Tk()
    app = TranscribeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
