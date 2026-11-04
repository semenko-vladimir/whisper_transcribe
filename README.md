# Whisper Transcribe: расшифровка и конспекты для Obsidian

Выберите аудио- или видеофайл либо вставьте ссылку на YouTube. В приложении вы получите расшифровку через Whisper и конспект на русском в Markdown: обзор, тезисы и разделы с таймкодами. Готовую заметку можно сохранить в хранилище Obsidian.

Для расшифровки приложение использует `faster-whisper`; для конспекта отправляет текст в Gemini или OpenRouter. Через командную строку можно обработать готовую расшифровку в JSON.

## Установка и первый запуск

Нужны Python 3.12+ с Tkinter и `pip`. Для некоторых файлов и загрузок с YouTube понадобится FFmpeg.

В PowerShell из папки проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

В `.env` укажите ключи сервисов для конспектирования:

```env
GEMINI_API_KEY=your_real_gemini_key
OPENROUTER_API_KEY=your_real_openrouter_key
# Папка для заметок, если хотите сохранять их в Obsidian:
# OBSIDIAN_OUTPUT_DIR=C:/path/to/your/Obsidian/vault/folder
```

Запустите оконный интерфейс:

```powershell
.\run.bat
```

В окне вставьте ссылку на YouTube или выберите файл. После расшифровки приложение запустит создание конспекта. JSON расшифровки вы найдёте в `output/`, Markdown в `OBSIDIAN_OUTPUT_DIR` или `output/notes/`, если путь не указан.

При первом использовании Whisper потребуется загрузить модель. В коде выбраны `large-v3`, CPU и вычисления `int8`.

## Конспектирование и выбор модели

В [config/models.yaml](config/models.yaml) заданы модели, порядок обращения и ограничения запросов:

| Приоритет | Модель | Ключ |
|---|---|---|
| 1 | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| 2 | `gemini-2.5-flash-lite` | `GEMINI_API_KEY` |
| 3 | `openrouter/free` | `OPENROUTER_API_KEY` |

Доступ к моделям и лимиты зависят от вашего аккаунта у провайдера. Чтобы изменить модель, отредактируйте конфигурацию.

Длинную расшифровку приложение делит на части с таймкодами. Для каждой части запрашивает структурированное резюме, затем объединяет ответы в конспект с YAML-свойствами для Obsidian.

При временных ошибках приложение повторяет запрос. После исчерпания попыток уменьшает размер запроса или переходит к следующему провайдеру. Для слишком большого итогового конспекта использует объединение в несколько этапов.

## Команды для готового JSON

Проверка настроек и список провайдеров:

```powershell
.\.venv\Scripts\python.exe -m app.main validate-config
.\.venv\Scripts\python.exe -m app.main list-providers
```

Проверка загрузки и разбиения расшифровки без генерации конспекта:

```powershell
.\.venv\Scripts\python.exe -m app.main dry-run input.json
```

Проверка одного провайдера и создание конспекта:

```powershell
.\.venv\Scripts\python.exe -m app.main test-provider --name gemini_flash_primary
.\.venv\Scripts\python.exe -m app.main summarize input.json output.md
```

Загрузчик принимает объект с массивом `segments` или `blocks`, а также массив сегментов без внешнего объекта. Поля сегмента: `start`, `end` и `text`. Если в объекте есть оба массива, приложение выбирает `segments`.

## Файлы и настройки

| Путь | Содержимое |
|---|---|
| `transcribe.py` | Оконный интерфейс, загрузка медиа и расшифровка |
| `app/main.py` | Команды для обработки JSON |
| `app/pipeline.py` | Разбиение текста, резюме частей и итоговый конспект |
| `app/providers/` | Подключения к Gemini и OpenAI-совместимому API |
| `app/fallback.py` | Обработка ошибок, повторы и смена провайдера |
| `app/markdown_writer.py` | Запись заметки в Markdown |
| `config/models.yaml` | Модели, лимиты и настройки приложения |
| `tests/` | Проверки загрузки, разбиения, ошибок и Markdown |

При включённом `save_intermediate_files` приложение сохраняет нормализованную расшифровку, резюме частей и итоговое структурированное резюме в JSON. Журнал пишет в консоль и файл с ротацией; путь задаётся через `app.log_file`.

Шаблоны запросов находятся в `app/prompts.py`. Для своих шаблонов укажите `app.prompt_overrides_path`; пример есть в [config/prompts.example.yaml](config/prompts.example.yaml).

## Проверки

```powershell
.\.venv\Scripts\python.exe -m pytest
```
