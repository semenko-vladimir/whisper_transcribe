from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from .constants import DEFAULT_LANGUAGE, ErrorCategory, StageName
from .errors import AllProvidersFailedError, SummarizationError
from .fallback import ErrorClassifier, FallbackPolicy, ProviderFallbackExecutor, RetryPolicy
from .markdown_writer import write_markdown
from .models import (
    ChunkSummaryArtifact,
    LLMRequest,
    MarkdownMetadata,
    PipelineResult,
    ProjectConfig,
    StructuredSummary,
    TranscriptChunk,
)
from .prompts import PromptTemplates, build_chunk_prompt, build_final_merge_prompt, load_prompt_templates
from .transcript_loader import chunk_transcript, load_transcript


class SummarizationPipeline:
    """Two-stage transcript summarization pipeline with provider fallback."""

    def __init__(self, config: ProjectConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.prompt_templates: PromptTemplates = load_prompt_templates(config.app.prompt_overrides_path)
        self.executor = ProviderFallbackExecutor(
            config.providers,
            logger=logger,
            retry_policy=RetryPolicy(base_delay_seconds=config.app.retry_backoff_seconds),
            fallback_policy=FallbackPolicy(max_degrade_attempts=config.app.max_degrade_attempts_per_provider),
            error_classifier=ErrorClassifier(),
        )

    def summarize(self, input_path: str | Path, output_path: str | Path) -> PipelineResult:
        """Run full transcript -> markdown conversion."""

        transcript = load_transcript(input_path)
        chunks = self._summarize_chunks_with_chunk_degrade(transcript)
        final_summary = self._merge_chunk_summaries(chunks)

        metadata = MarkdownMetadata(
            title=final_summary.title,
            source_type="transcript",
            language=transcript.detected_language or self.config.app.default_language or DEFAULT_LANGUAGE,
            date_created=date.today().isoformat(),
            status="готово",
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(final_summary, metadata, output_path)

        artifact_dir: Path | None = None
        if self.config.app.save_intermediate_files:
            artifact_dir = self._save_artifacts(
                output_path=output_path,
                transcript=transcript,
                chunks=chunks,
                final_summary=final_summary,
            )

        self.logger.info("Markdown summary saved to %s", output_path)
        return PipelineResult(
            input_path=Path(input_path),
            output_path=output_path,
            chunk_count=len(chunks),
            artifact_dir=artifact_dir,
        )

    def dry_run(self, input_path: str | Path) -> dict[str, int]:
        """Return chunking stats without calling any model provider."""

        transcript = load_transcript(input_path)
        chunks = chunk_transcript(
            transcript,
            target_chars=self.config.app.chunk_target_chars,
            overlap_chars=self.config.app.chunk_overlap_chars,
        )
        return {
            "segments": len(transcript.segments),
            "chunks": len(chunks),
            "chunk_target_chars": self.config.app.chunk_target_chars,
            "chunk_overlap_chars": self.config.app.chunk_overlap_chars,
        }

    def _summarize_chunks_with_chunk_degrade(self, transcript) -> list[ChunkSummaryArtifact]:
        target_chars = self.config.app.chunk_target_chars

        for reduction_attempt in range(self.config.app.max_chunk_reduce_attempts + 1):
            chunks = chunk_transcript(
                transcript,
                target_chars=target_chars,
                overlap_chars=self.config.app.chunk_overlap_chars,
            )
            self.logger.info(
                "Chunking transcript with target_chars=%s overlap_chars=%s produced %s chunks.",
                target_chars,
                self.config.app.chunk_overlap_chars,
                len(chunks),
            )

            try:
                return [self._summarize_chunk(chunk) for chunk in chunks]
            except AllProvidersFailedError as exc:
                if (
                    exc.last_category == ErrorCategory.CONTEXT_TOO_LONG
                    and target_chars > self.config.app.min_chunk_target_chars
                    and reduction_attempt < self.config.app.max_chunk_reduce_attempts
                ):
                    next_target = max(
                        self.config.app.min_chunk_target_chars,
                        int(target_chars * 0.7),
                    )
                    self.logger.warning(
                        "Context pressure detected. Reducing chunk_target_chars from %s to %s and retrying.",
                        target_chars,
                        next_target,
                    )
                    target_chars = next_target
                    continue
                raise

        raise SummarizationError("Failed to summarize transcript chunks after all degrade attempts.")

    def _summarize_chunk(self, chunk: TranscriptChunk) -> ChunkSummaryArtifact:
        self.logger.info(
            "Summarizing chunk %s [%s-%s] (%s chars).",
            chunk.chunk_id,
            chunk.start_timecode,
            chunk.end_timecode,
            chunk.character_count,
        )
        request = LLMRequest(
            stage=StageName.CHUNK_SUMMARY,
            system_prompt=self.prompt_templates.system_prompt,
            user_prompt=build_chunk_prompt(chunk, self.prompt_templates),
            expected_schema_name="StructuredSummary",
        )
        summary, _attempts = self.executor.generate_structured(request, StructuredSummary)
        return ChunkSummaryArtifact(
            chunk_id=chunk.chunk_id,
            chunk_start_timecode=chunk.start_timecode,
            chunk_end_timecode=chunk.end_timecode,
            chunk_character_count=chunk.character_count,
            summary=summary,
        )

    def _merge_chunk_summaries(self, chunk_artifacts: list[ChunkSummaryArtifact]) -> StructuredSummary:
        summaries: list[StructuredSummary] = [artifact.summary for artifact in chunk_artifacts]
        level = 0

        while True:
            try:
                return self._merge_summary_batch(summaries, level=level, batch_index=None)
            except AllProvidersFailedError as exc:
                if exc.last_category != ErrorCategory.CONTEXT_TOO_LONG or len(summaries) <= 1:
                    raise

                batches = self._batch_summaries_for_merge(summaries)
                self.logger.warning(
                    "Final merge hit context pressure. Switching to hierarchical merge: %s batches.",
                    len(batches),
                )
                merged_batches: list[StructuredSummary] = []
                for batch_index, batch in enumerate(batches, start=1):
                    merged_batches.append(self._merge_summary_batch(batch, level=level + 1, batch_index=batch_index))
                summaries = merged_batches
                level += 1

    def _merge_summary_batch(
        self,
        summaries: list[StructuredSummary],
        *,
        level: int,
        batch_index: int | None,
    ) -> StructuredSummary:
        label = f"level={level}"
        if batch_index is not None:
            label += f" batch={batch_index}"

        self.logger.info("Running final merge for %s with %s summaries.", label, len(summaries))
        request = LLMRequest(
            stage=StageName.FINAL_MERGE,
            system_prompt=self.prompt_templates.system_prompt,
            user_prompt=build_final_merge_prompt(summaries, self.prompt_templates),
            expected_schema_name="StructuredSummary",
        )
        summary, _attempts = self.executor.generate_structured(request, StructuredSummary)
        return summary

    def _batch_summaries_for_merge(self, summaries: list[StructuredSummary]) -> list[list[StructuredSummary]]:
        batches: list[list[StructuredSummary]] = []
        current_batch: list[StructuredSummary] = []
        current_chars = 0

        for summary in summaries:
            serialized = json.dumps(summary.model_dump(), ensure_ascii=False)
            if current_batch and current_chars + len(serialized) > self.config.app.merge_batch_target_chars:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

            current_batch.append(summary)
            current_chars += len(serialized)

        if current_batch:
            batches.append(current_batch)

        return batches

    def _save_artifacts(self, *, output_path: Path, transcript, chunks, final_summary: StructuredSummary) -> Path | None:
        if not self.config.app.intermediate_json:
            return None

        artifact_base = output_path.with_suffix("")
        artifact_dir = artifact_base.parent / f"{artifact_base.name}_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        normalized_path = artifact_dir / "normalized_transcript.json"
        chunk_summaries_path = artifact_dir / "chunk_summaries.json"
        final_summary_path = artifact_dir / "final_summary.json"

        normalized_path.write_text(
            json.dumps(transcript.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        chunk_summaries_path.write_text(
            json.dumps([chunk.model_dump(mode="json") for chunk in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        final_summary_path.write_text(
            json.dumps(final_summary.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return artifact_dir
