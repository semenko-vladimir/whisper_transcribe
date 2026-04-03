# Transcript JSON -> Obsidian Markdown

Production-ready Python CLI project that converts transcript JSON into a structured Russian Markdown summary for Obsidian.

For the current local workflow, the main entry point is now the GUI in `transcribe.py`: it performs transcription first and then automatically starts markdown summarization.

## Features

- Loads transcript JSON with `segments` or `blocks`
- Normalizes transcript structure before processing
- Splits long transcripts into timestamped chunks
- Runs a 2-stage summarization pipeline:
  - chunk-level structured JSON summaries
  - final merge into one structured summary
- Renders final result as Obsidian-friendly Markdown with YAML frontmatter
- Supports multiple providers with retry, degrade, and automatic fallback
- Logs both to console and to a rotating file
- Saves intermediate JSON artifacts

## Project Structure

```text
app/
  main.py
  config.py
  constants.py
  errors.py
  fallback.py
  logging_utils.py
  markdown_writer.py
  models.py
  pipeline.py
  prompts.py
  transcript_loader.py
  providers/
    __init__.py
    base.py
    gemini.py
    openai_compatible.py
config/
  models.yaml
  prompts.example.yaml
tests/
  test_chunking.py
  test_error_classification.py
  test_fallback.py
  test_loader.py
  test_markdown.py
.env.example
requirements.txt
README.md
run.bat
```

## Requirements

- Python 3.12+
- `pip`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Start

Windows launcher:

```bat
.\run.bat
```

Why not `start`:

- `start` is already used by Windows shells
- in `cmd.exe` it is a built-in command
- in PowerShell it maps to `Start-Process`
- because of that, bare `start` is not a reliable project-local launcher name
- the recommended project command is `.\run.bat`

The launcher:

- switches to the project root
- uses `.venv\Scripts\pythonw.exe` when available
- falls back to `.venv\Scripts\python.exe`
- shows a clear message if `.venv` is missing

After launch:

1. the GUI window opens
2. you paste a YouTube link or choose a local file
3. transcription creates JSON
4. summarization starts automatically
5. final Markdown is saved into the directory from OBSIDIAN_OUTPUT_DIR or, if it is not set, into output\\notes

## Environment Setup

1. Copy `.env.example` to `.env`
2. Fill in your real API keys
3. Optionally set OBSIDIAN_OUTPUT_DIR if you want Markdown to go straight into your Obsidian vault

Example:

```env
GEMINI_API_KEY=your_real_gemini_key
OPENROUTER_API_KEY=your_real_openrouter_key
# Optional: OBSIDIAN_OUTPUT_DIR=C:/path/to/your/Obsidian/vault/folder
```

## Provider Configuration

Edit `config/models.yaml`.

Default setup:

1. `gemini-2.5-flash`
2. `gemini-2.5-flash-lite`
3. `openrouter/free`

The first two Gemini providers use the same `GEMINI_API_KEY`. Only the `model` field changes between primary and backup.

OpenRouter is configured as an OpenAI-compatible endpoint and uses `OPENROUTER_API_KEY`.

The same provider config is also used automatically by the GUI after transcription succeeds.

Example:

```yaml
providers:
  - name: gemini_flash_primary
    provider: gemini
    api_key_env: GEMINI_API_KEY
    model: gemini-2.5-flash
    temperature: 0.2
    top_p: 0.9
    max_output_tokens: 4096
    timeout_seconds: 90
    max_retries: 2
    priority: 1

  - name: gemini_flash_lite_backup
    provider: gemini
    api_key_env: GEMINI_API_KEY
    model: gemini-2.5-flash-lite
    temperature: 0.2
    top_p: 0.9
    max_output_tokens: 3072
    timeout_seconds: 60
    max_retries: 2
    priority: 2

  - name: openrouter_free_backup
    provider: openai_compatible
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1
    model: openrouter/free
    temperature: 0.2
    top_p: 0.9
    max_output_tokens: 3072
    timeout_seconds: 90
    max_retries: 2
    priority: 3

app:
  chunk_target_chars: 10000
  chunk_overlap_chars: 800
  log_level: INFO
  intermediate_json: true
  save_intermediate_files: true
```

## How Fallback Works

The policy layer has three parts:

- `ErrorClassifier`: maps provider/network/parser errors to categories such as `quota_exhausted`, `rate_limited`, `timeout`, `context_too_long`, `malformed_response`
- `RetryPolicy`: retries transient failures first
- `FallbackPolicy`: if retry is exhausted, either degrades the request or switches to the next provider

Default fallback chain:

1. `gemini_flash_primary` -> `gemini-2.5-flash`
2. `gemini_flash_lite_backup` -> `gemini-2.5-flash-lite`
3. `openrouter_free_backup` -> `openrouter/free`

Degrade strategy includes:

- reducing `max_output_tokens`
- retrying stricter requests after malformed output
- reducing transcript chunk size if the transcript is too large
- switching to later providers in the chain
- hierarchical merge if the final synthesis is too large for a single call

## Supported Providers

### Gemini

- Uses the official Gemini HTTP API
- Reads API key from `GEMINI_API_KEY`
- Default primary model is `gemini-2.5-flash`
- Default backup model is `gemini-2.5-flash-lite`

### OpenRouter via OpenAI-compatible API

- Uses `/chat/completions`
- Uses `base_url: https://openrouter.ai/api/v1`
- Reads API key from `OPENROUTER_API_KEY`
- Acts as the final cloud fallback provider

## Commands

Main day-to-day workflow:

```bat
.\run.bat
```

Optional CLI utilities:

Validate config:

```bash
python -m app.main validate-config
```

List providers:

```bash
python -m app.main list-providers
```

Dry-run transcript loading and chunking:

```bash
python -m app.main dry-run input.json
```

Test a single provider:

```bash
python -m app.main test-provider --name gemini_flash_primary
```

Generate markdown summary:

```bash
python -m app.main summarize input.json output.md
```

## Transcript Input Format

The loader supports:

1. Object with `segments`
2. Object with `blocks`
3. Raw JSON array of segment-like objects

Expected fields per item:

- `start`
- `end`
- `text`

If both `segments` and `blocks` exist, `segments` are preferred.

## Output

The final markdown includes:

- YAML frontmatter
- top-level overview
- semantic sections with timestamps
- bullets with key theses
- main ideas
- practical takeaways

When `save_intermediate_files` is enabled, the app also stores:

- normalized transcript JSON
- chunk summaries JSON
- final structured summary JSON

Current GUI workflow output paths:

- transcript JSON: project-local `output\`
- final Markdown: `OBSIDIAN_OUTPUT_DIR` if set, otherwise project-local `output\\notes\\`

## Running Tests

```bash
pytest
```

## What You Need To Change

Update these files for your real setup:

- `.env`: put your actual `GEMINI_API_KEY` and `OPENROUTER_API_KEY` and optionally `OBSIDIAN_OUTPUT_DIR`
- `config/models.yaml`: adjust model names, priorities, token limits, or timeouts if your setup changes
- `config/models.yaml`: keep the first two Gemini providers on the same `GEMINI_API_KEY`; only the `model` value differs
- `config/models.yaml`: keep OpenRouter on `OPENROUTER_API_KEY` with the OpenAI-compatible endpoint
- make sure `.venv` exists before using `.\run.bat`

## Notes

- The project keeps prompts in `app/prompts.py`
- Prompt templates can be overridden through `app.prompt_overrides_path` in YAML
- A starter override example is included in `config/prompts.example.yaml`
- Logging file path is configurable through `app.log_file`
