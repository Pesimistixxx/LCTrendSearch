# LCTrendSearch

Получение материалов → связанные пакеты → LLM-извлечение → проверка → сборка → идентификация сущностей → Neo4j.

[Архитектура: две схемы, пояснения этапов и код](docs/pipeline-before-after.md) · [Что означают списки строк](docs/catalogs.md) · [Веб-интерфейс сбора](frontend/README.md)

Реализована многоэтапная обработка материалов и сборка временного датасета технологий. Сбор и проверка загрузки встроены в основной React-интерфейс. Поиск и ранжирование технологических кандидатов пока показывают демонстрационные данные.

## Быстрый старт: OpenAlex

Все команды ниже запускайте из каталога `LCTrendSearch`. OpenAlex подключён к разбору сохранённого JSON, получению статьи по DOI/ID, тематическому обходу, HTTP API и записи документов в Neo4j. Для первой загрузки карточки и аннотации достаточно базового Python-пакета; `--no-fulltext` отключает PDF и загрузку моделей Docling.

### Самая быстрая проверка без Docker и базы

Нужен Python 3.9+; для полного набора PDF/NER и веб-интерфейса используйте Python 3.12, как в Docker. PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
New-Item -ItemType Directory -Force artifacts | Out-Null
```

В `.env` задайте `OPENALEX_API_KEY`: создайте бесплатную учётную запись и скопируйте ключ из [настроек OpenAlex](https://openalex.org/settings/api). Ключ рекомендован для массовой загрузки; для нескольких пробных запросов API допускает ограниченный доступ без него. `OPENALEX_MAILTO` — необязательный контактный email, не замена ключу. Актуальные правила и лимиты описаны в [документации авторизации OpenAlex](https://help.openalex.org/api/authentication/).

```powershell
.venv\Scripts\python.exe -m lctrend fetch openalex "https://doi.org/10.7717/peerj.4375" --no-fulltext --output artifacts/paper-envelope.json
# Эквивалентно: fetch openalex W2741809807 или fetch openalex "doi:10.7717/peerj.4375"
$paper = Get-Content artifacts/paper-envelope.json -Raw -Encoding UTF8 | ConvertFrom-Json
$paper | Select-Object title, coverage
$paper.chunks | Select-Object -First 2 kind, text
```

`coverage=abstract_only` означает доступную аннотацию, `metadata_only` — только карточку. Этот шаг не требует Neo4j, LLM или GLiNER. Повторный разбор сохранённого исходного ответа API выполняется командой `parse openalex work.json`; `paper-envelope.json` уже содержит внутреннюю схему и не подходит на место `work.json`.

### Docker и веб-интерфейс

Нужен запущенный Docker с Compose. Если `.env` ещё нет, скопируйте `.env.example`; заполните `OPENALEX_API_KEY` и параметры существующей внешней базы `NEO4J_*`. Для извлечения сущностей настройте GigaChat или совместимый LLM API. Быструю загрузку только документов выполняйте с `--no-extract`.

```powershell
docker compose up -d --build --wait
docker compose exec backend python -m lctrend init-graph
docker compose exec backend python -m lctrend fetch openalex W2741809807 --no-fulltext --output /app/artifacts/paper-envelope.json
```

Откройте [сбор материалов](http://localhost:5188/#view=ingest). `FRONTEND_PORT` меняет порт. Ключ OpenAlex также можно сохранить через «Настройки источников»; API показывает только факт наличия ключа. Compose передаёт `.env` только API-сервису; ключ не нужен в `VITE_*`. Первая сборка устанавливает зависимости PDF/NER и может занять больше времени; для получения одной аннотации быстрее локальная команда выше. После изменения `.env` примените `docker compose up -d --force-recreate backend`.

### Быстрая загрузка темы в Neo4j

Настройте `NEO4J_*` в `.env`. Ограниченный обход ниже сохраняет карточки, аннотации, авторов и организации без вызовов LLM/NER и скачивания PDF:

```powershell
.venv\Scripts\python.exe -m lctrend init-graph
.venv\Scripts\python.exe -m lctrend crawl-openalex "edge computing" --limit 20 --per-page 20 --filter "has_abstract:true,type:article" --no-fulltext --no-extract --checkpoint artifacts/openalex-edge.json
```

В Docker используйте `docker compose exec backend python` вместо `.venv\Scripts\python.exe`, а checkpoint задайте как `/app/artifacts/openalex-edge.json`. Для последующего извлечения из аннотаций уберите `--no-extract` и используйте `--extractor llm` с настроенным LLM; локально установите `.[llm]`. Для PDF дополнительно установите `.[pdf]` и уберите `--no-fulltext`.

Checkpoint сохраняет `query`, `filter`, `processed`, `failures` и позицию `cursor` после каждой страницы. Повтор той же команды продолжает обход; `--limit` — общий предел с учётом `processed`, поэтому для следующих 20 записей увеличьте его до 40. Для другой темы, фильтра или повторной обработки с новым режимом укажите новый файл checkpoint. Счётчик `failures` и журнал `logs/lctrend.log` помогают проверить ошибки; это не число успешно извлечённых технологий. Без извлечения создаются документы и текстовые фрагменты, а утверждения и технологические признаки появятся после обработки моделью.

## Какие модели используются

| Роль | Модель | Где задаётся | Когда работает |
|---|---|---|---|
| LLM: извлечение и проверка утверждений | GigaChat, лестница `GigaChat-3-Ultra` → `GigaChat-2-Max` → `GigaChat-2-Pro` → `GigaChat-2`; либо любая модель OpenAI-совместимого API | `LLM_PROVIDER`, `LLM_MODEL*`, [llm.json](src/lctrend/resources/llm.json) | Режимы `hybrid` (по умолчанию) и `llm` |
| NER: подсказки сущностей | GLiNER `urchade/gliner_medium-v2.1` | `GLINER_MODEL`, [runtime.json](src/lctrend/resources/runtime.json) | `hybrid` (подсказки для LLM) и `gliner` (основной извлекатель) |
| Разбор PDF | Docling (собственные модели разметки страниц и OCR) | extra `[pdf]` | Любой PDF: локальный файл, загрузка через API, полный текст OpenAlex |
| Эмбеддинги для сопоставления сущностей | GigaChat `EmbeddingsGigaR` через API `/embeddings` (косинус; локальная альтернатива — `sentence-transformers/all-MiniLM-L6-v2`) + локальный `cross-encoder/stsb-distilroberta-base` (проверка пары) | `DEDUP_*`, [resolver.json](src/lctrend/resources/resolver.json) | Работают в `gliner`, `hybrid` и `llm`. В двух LLM-режимах смысловое сходство сохраняется как кандидат `POSSIBLY_SAME_AS` для проверки |

Эмбеддинг-совпадение никогда не объединяет сущности автоматически: пара получает статус `ambiguous` и остаётся на проверку. Векторного индекса в Neo4j нет — эмбеддинги вычисляются в памяти на время одной обработки.

## Как передать конкретную статью

**Прямую PDF-ссылку сейчас нельзя передать в `ingest file` или поле «Тематика».** Для отдельной статьи используйте локальный файл либо DOI/ID OpenAlex. Поле «Тематика» запускает поиск нескольких материалов по строке запроса.

| Что у вас есть | Как передать | Что будет обработано |
|---|---|---|
| PDF на компьютере | `parse file путь.pdf` / `ingest file путь.pdf` | Текст PDF, полученный Docling; страницы и координаты — если их вернул парсер |
| PDF/DOCX/TXT/MD/HTML при запущенном Docker | `POST /api/ingest/uploads` ([пример](frontend/README.md#как-передать-одну-статью-а-не-тему)) | То же, что `ingest file`, но фоновым заданием; в веб-форме выбора файла пока нет |
| Прямая HTTP(S)-ссылка на PDF | Скачать файл, затем передать локальный путь | Скачанные байты PDF; автоматического ввода произвольного URL в CLI сейчас нет |
| DOI статьи | `fetch openalex "https://doi.org/…"` / `fetch openalex "doi:…"` | Карточка OpenAlex, доступная аннотация и попытка получения PDF |
| ID OpenAlex | `fetch openalex W2741809807` | Одна конкретная публикация |
| Страница издателя / `arxiv.org/abs/…` | Найти прямую PDF-ссылку или DOI и воспользоваться вариантом выше | Произвольные страницы статей автоматически не обходятся |
| Тема, например `edge computing` | Поле «Тематика» в вебе / `crawl-openalex "edge computing"` | Материалы из тематической выдачи, а не одна заданная статья |
| Сохранённый ответ OpenAlex в JSON | `parse openalex work.json` / `ingest openalex work.json` | Сохранённые метаданные и аннотация; эти команды не скачивают PDF |

### PDF по ссылке или с компьютера

Пример для PowerShell вне Docker из каталога `LCTrendSearch`, с активированным Python-окружением и установленным пакетом:

```powershell
python -m pip install -e ".[pdf]"
# Замените URL на прямую PDF-ссылку своего документа.
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2311.01235" -OutFile .\paper.pdf

# Проверить разбор текста без LLM и Neo4j.
python -m lctrend parse file .\paper.pdf --output .\paper-envelope.json

# После настройки LLM и Neo4j извлечь сведения и записать граф.
python -m lctrend ingest file .\paper.pdf --extraction-output .\paper-extraction.json
```

Если файл уже на компьютере, пропустите скачивание. Ответ по ссылке должен содержать PDF, а не HTML, форму входа или CAPTCHA. Для Docker сохраните файл в смонтированную папку `artifacts` и используйте путь внутри контейнера:

```powershell
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2311.01235" -OutFile .\artifacts\paper.pdf
docker compose exec backend python -m lctrend parse file /app/artifacts/paper.pdf --output /app/artifacts/paper-envelope.json
docker compose exec backend python -m lctrend ingest file /app/artifacts/paper.pdf --extraction-output /app/artifacts/paper-extraction.json
```

Локальный PDF хранится как `document_type=report`, источник — `local_file`, название берётся из имени файла. DOI, авторы и дата публикации автоматически не восстанавливаются из PDF; `published_at` остаётся пустым. Это загрузка текста документа. Для научной карточки с библиографическими метаданными используйте OpenAlex.

### Статья по DOI / OpenAlex

```powershell
# Получить одну статью, сохранить DocumentEnvelope; без LLM и записи в граф.
python -m lctrend fetch openalex "https://doi.org/10.7717/peerj.4375" --output .\paper-envelope.json

# Получить, извлечь сведения и записать в Neo4j.
python -m lctrend fetch openalex "https://doi.org/10.7717/peerj.4375" --ingest --extraction-output .\paper-extraction.json

# Оставить только карточку/аннотацию; Docling не требуется.
python -m lctrend fetch openalex W2741809807 --no-fulltext --output .\abstract-envelope.json
```

В Docker добавьте `docker compose exec backend` перед `python`, а выходные файлы задайте в `/app/artifacts/`. DOI с префиксом `doi:` и полные ссылки DOI/ID поддерживаются [API одной записи OpenAlex](https://help.openalex.org/api/get-single-entities/). Произвольный PDF URL идентификатором OpenAlex не является.

`fetch openalex` и `crawl-openalex` по умолчанию ищут `pdf_url` в `best_oa_location`, `primary_location` и `locations`; явно закрытые варианты (`is_oa=false`) пропускаются. Пробуются до трёх разных HTTP(S)-ссылок. Загрузчик проверяет сигнатуру `%PDF-` и размер. Полученный текст добавляется к аннотации, колонтитулы и распознанная библиография отфильтровываются.

Если PDF-ссылок нет, остаётся доступная аннотация/карточка. Если все загрузки или разборы не удались, причины сохраняются в `document.metadata.fulltext.attempts`. Если PDF-ссылка есть, но Docling не установлен, CLI завершится ошибкой: установите `[pdf]` или передайте `--no-fulltext`. Сборщик с полным текстом проверяет наличие Docling ещё до обхода.

### Как убедиться, что получен текст статьи

Проверьте `paper-envelope.json`, а не только сообщение об успешном запуске:

```powershell
$paper = Get-Content .\paper-envelope.json -Raw -Encoding UTF8 | ConvertFrom-Json
$paper.title
$paper.source.canonical_url
$paper.coverage
$paper.chunks.Count
$paper.chunks | Select-Object -First 3 kind, text, locator
$paper.metadata.fulltext       # сетевой OpenAlex
$paper.metadata.parse_warnings # локальный PDF
```

| Поле / значение | Что означает |
|---|---|
| `chunks` | Фрагменты текста, доступные извлечению. Пустой список означает, что текста для анализа нет |
| `coverage=metadata_only` | Получена карточка без аннотации и полного текста |
| `coverage=abstract_only` | Получена только аннотация; вся статья не обработана |
| `coverage=abstract_and_full_text` / `full_text` у OpenAlex | Добавлен текст PDF; качество и полноту нужно проверить по фрагментам |
| `coverage=parsed_text` у локального PDF | Получен текст Docling; это не гарантия распознавания всех страниц |
| `metadata.fulltext.status=parsed` | Один PDF OpenAlex скачан и дал непустой текст |
| `no_pdf_url` / `failed` | PDF-ссылок нет / все попытки получения полного текста не удались |
| `parse_warnings`, `fulltext.warnings`, `quality_status` | Ограничения OCR, таблиц, координат и разбора |

Для результата извлечения отдельно смотрите `paper-extraction.json`: `run.status`, `run.metadata.issues` и `run.metadata.coverage`. LLM может обработать лишь часть уже загруженных фрагментов. Наличие JSON и код завершения CLI 0 сами по себе не гарантируют `succeeded`.

## Запуск

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Укажите внешнюю Neo4j в NEO4J_* и настройте LLM-провайдера.
docker compose up -d --build --remove-orphans --wait
```

Открыть [Сбор материалов](http://localhost:5188/#view=ingest).
Compose запускает три сервиса: `frontend` (React + Nginx), `backend`
(FastAPI, LLM, GLiNER, Docling) и `postgres`. Локальной Neo4j в Compose
нет: API использует внешнюю базу из `.env`. `--remove-orphans` удаляет
контейнер Neo4j от старой конфигурации; его том с данными сохраняется.
Порт интерфейса задаётся `FRONTEND_PORT` (по умолчанию 5188), PostgreSQL —
`POSTGRES_PORT` (5432). API доступен через `/api` на том же адресе.

JSON заданий, результаты и исходные материалы сохраняются в `artifacts/`,
логи — в `logs/`. Реестр обхода SQLite хранится в Docker-томе
`ingestion_ledger`: режим WAL требует файловой системы Linux, поэтому
папка `artifacts/ingestion/crawl` внутри контейнера перекрыта этим томом.
Кэш моделей также хранится в отдельном Docker-томе. Настройки LLM и OpenAlex
из интерфейса сохраняются в `artifacts/ingestion/settings.env` и действуют
после пересоздания контейнеров. Они имеют приоритет над соответствующими настройками
из `.env`; чтобы снова использовать `.env`, удалите файл сохранённых
настроек и пересоздайте API. Первое обращение к GLiNER/Docling может
потребовать загрузки моделей. Первая сборка образа устанавливает их
зависимости, включая CPU-версию PyTorch.

Проверка и остановка:

```powershell
docker compose ps
docker compose logs -f backend
docker compose down
```

CLI можно запускать внутри API-контейнера:

```powershell
docker compose exec backend python -m lctrend init-graph
docker compose exec backend python -m lctrend ingest file /app/artifacts/report.md --extraction-output /app/artifacts/report-extraction.json
```

CLI читает `.env` без перезаписи экспортированных переменных (в Docker
дополнительно читает сохранённые настройки LLM и OpenAlex). Neo4j хранит документы,
исходные блоки, сущности, утверждения, цитаты и аудит обработки.
PostgreSQL зарезервирован под будущую ранжированную выдачу сигналов:
сейчас код в него не пишет и не читает, но `backend` ждёт его healthcheck.

LLM-провайдер использует endpoint `/chat/completions`. Задайте модель или лестницу; для GigaChat уже есть штатная лестница. Отдельные модели стадий задаются через `LLM_EXTRACT_MODEL` и `LLM_REVIEW_MODEL`. Извлекатель и проверяющий используют разные промпты. Отсутствующая настройка не переключает обработку на фиктивные ответы. Зависимость httpx входит в базовый пакет и используется сетевыми коннекторами и LLM.

## Массовая загрузка

Сетевые операции асинхронные: LLM, Neo4j, OpenAlex, GitHub, PyPI и скачивание PDF. Задания веб-интерфейса выполняются в отдельном event loop. Разбор Docling, GLiNER и сопоставление сущностей работают в потоках и не блокируют остальные документы.

| Настройка | По умолчанию | Что ограничивает |
|---|---|---|
| `LCTREND_WORKERS` / `runtime.json` → `ingestion.workers` | 4 | Документов одного задания или пачки обхода одновременно, 1–16 |
| `crawl-openalex --workers N` | как выше | То же для CLI |
| `LLM_MAX_CONCURRENCY` / `llm.json` → `max_concurrent_requests` | GigaChat: 1, OpenAI-совместимый: 4 | Одновременных запросов к LLM на процесс |
| `pipeline.json` → `model_calls_per_packet`, `max_document_model_calls` | 3 и 120 | Бюджет вызовов LLM растёт с числом пакетов документа, но не выше предела |
| `pipeline.json` → `max_retries`, `retry_delay_seconds`, `max_retry_delay_seconds` | 4, 2 с, 60 с | Повторы LLM при 429/5xx/таймауте с экспоненциальной паузой или по `Retry-After` |
| `sources.json` → `http` | 5 попыток, пауза 1–60 с | Повторы OpenAlex/GitHub/PyPI/PDF при 408/429/5xx; лимит GitHub ждёт `X-RateLimit-Reset` до 15 минут |
| `pipeline.json` → `file_limits.pdf_max_pages`, `pdf_timeout_seconds` | 200 страниц, 900 с | Один конвертер Docling на процесс; слишком длинный или зависший PDF не останавливает задание |

Для GigaChat параллельность документов ускоряет в основном скачивание и разбор PDF. Сами запросы к модели выполняются по одному, если не увеличить `LLM_MAX_CONCURRENCY` в пределах тарифа. Полный текст увеличивает число вызовов LLM на статью: примерно 3 вызова на 10 фрагментов, не более 120 на документ. Перед большим обходом оцените бюджет токенов.

Повторная загрузка того же материала бесплатна. Версия документа OpenAlex/PyPI/GitHub не зависит от времени скачивания. Если для версии в графе уже есть успешная обработка, документ помечается `already_processed` и не отправляется в LLM. Результат `partial` публикуется в граф, если у версии ещё нет активного результата. Более поздний полный результат его заменяет. Повторный `partial` поверх опубликованного результата сохраняется только как staged.

Задание отображается в `job.json`. Прогресс записывается не чаще раза в секунду, смена статусов — сразу. Реестр концептов читается из Neo4j один раз на задание и дальше обновляется в памяти.

## Источники и команды

```powershell
python -m lctrend parse openalex .\work.json --output work-envelope.json
python -m lctrend parse file .\report.docx --output report-envelope.json
python -m lctrend ingest epo .\patent.xml
python -m lctrend fetch github urchade/GLiNER --ingest
python -m lctrend fetch pypi gliner --ingest --no-extract
python -m lctrend crawl-openalex "edge computing" --limit 10
python -m lctrend crawl-pypi --packages gliner pydantic
```

`parse` формирует DocumentEnvelope. `fetch --ingest`, `ingest` и сборщики публикуют в Neo4j; по умолчанию используют LLM-пакеты. `--no-extract` сохраняет только документ и chunks. `--extraction-output` у одиночной загрузки дополнительно сохраняет проверяемый JSON результата, включая историю стадий и покрытие.

Локальные файлы: UTF-8 TXT/MD/HTML, DOCX и PDF. Для PDF установите `python -m pip install -e ".[pdf]"`; Docling может загружать свои модели. DOCX/HTML не дают полной геометрии таблиц, PDF/OCR требует проверки качества: ограничения фиксируются в метаданных. Лимит файла по умолчанию — 50 МиБ. `work.json`, `report.docx` и `patent.xml` в примерах — ваши входные файлы, в поставке их нет. `work.json` должен содержать исходный ответ OpenAlex, а не уже сформированный DocumentEnvelope.

Снимки исходных байтов хранятся в artifacts/raw; каталог задаётся LCTREND_RAW_DIR. Это источник для воспроизводимости, хранилище графа остаётся Neo4j. GitHub фиксирует commit SHA; даты публикации версии, получения и наблюдения метрик разделены. Неизвестная независимость источника сохраняется неизвестной.

### Что означает каждая команда

| Команда | Назначение |
|---|---|
| `init-graph` | Создать ограничения схемы Neo4j; не загружает документы |
| `parse KIND INPUT` | Разобрать локальный файл и вывести DocumentEnvelope без LLM и базы. `KIND`: `file`, `openalex`, `github`, `pypi`, `epo`; для API нужен JSON, для EPO — XML |
| `ingest KIND INPUT` | Такой же разбор, затем извлечение и запись в Neo4j |
| `fetch KIND IDENTIFIER` | Запросить API: `openalex` — DOI/ID, `github` — `owner/repo`, `pypi` — имя пакета. По умолчанию только вывести документ; с `--ingest` обработать и записать в граф |
| `crawl-openalex QUERY` | Получать страницы статей по поисковой строке и записывать результат в Neo4j с сохранением позиции обхода |
| `crawl-pypi` | Обрабатывать `--packages` либо равномерную выборку из списка имён PyPI; тематического поиска по реестру нет |
| `export-features --snapshot DATE` | Выгрузить признаки технологий из графа на дату среза в CSV |
| `build-training-set` | Выгрузить исторические срезы технологий, будущие результаты реализации и временные разбиения; модель не обучает |

### Что означает каждый параметр CLI

| Параметр | Где действует | Значение |
|---|---|---|
| `--output PATH` | `parse`, `fetch`, CSV-команды | Файл документа/CSV. Без него `parse`/`fetch` печатают JSON. Каталог выходного файла для них должен уже существовать |
| `--ingest` | `fetch` | После получения документа выполнить извлечение и запись в граф |
| `--no-extract` | `ingest`, `fetch --ingest`, сборщики | Записать документ и chunks без LLM/NER; скачивание PDF остаётся включённым |
| `--extractor hybrid\|llm\|gliner` | Команды с извлечением | Режим обработки; по умолчанию `hybrid`, подробнее ниже |
| `--ner-model NAME` | Команды с извлечением | Модель GLiNER; перекрывает `GLINER_MODEL`, по умолчанию `urchade/gliner_medium-v2.1` |
| `--extraction-output PATH` | `ingest`, `fetch --ingest` | Сохранить ExtractionResult отдельным JSON; при `--no-extract` файл не создаётся. `--output` сохраняет другой объект — DocumentEnvelope |
| `--no-fulltext` | `fetch openalex`, `crawl-openalex` | Не загружать PDF, использовать доступную карточку/аннотацию |
| `--limit N` | Сборщики | Предел статей OpenAlex (500) / успешных загрузок пакетов PyPI (5000). С `--packages` равен числу заданных имён |
| `--per-page N` | `crawl-openalex` | Размер страницы API, по умолчанию 100; допустимо 1–100 |
| `--filter TEXT` | `crawl-openalex` | Фильтр OpenAlex, например `is_oa:true,has_abstract:true,type:article`, вместе с поисковой строкой |
| `--checkpoint PATH` | Сборщики | JSON с позицией обхода, по умолчанию `.openalex-crawl.json` / `.pypi-crawl.json`. Для другого запроса/списка используйте отдельный файл |
| `--packages NAME …` | `crawl-pypi` | Явный список пакетов вместо выборки |
| `--sample-phase FLOAT` | `crawl-pypi` | Сдвиг равномерной выборки в диапазоне `[0, 1)`, по умолчанию 0; не относится к тематике |
| `--snapshot DATE` | `export-features` | Обязательная дата среза `YYYY-MM-DD` |
| `--start-year N` | `build-training-set` | Первый год срезов, по умолчанию 2015 |
| `--horizon-years N` | `build-training-set` | Число лет после среза, по умолчанию 3 |
| `--min-documents N` | `build-training-set` | Минимум видимых документов до среза для включения технологии, по умолчанию 2 |
| `--end-date DATE` | `build-training-set` | Последняя дата среза `YYYY-MM-DD`; по умолчанию конец корпуса |
| `--subgraphs-output PATH` | `build-training-set` | Дополнительно сохранить ограниченные подграфы каждого среза в JSONL |
| `--no-taxonomy` | `export-features`, `build-training-set` | Отключить семантические, таксономические и графовые признаки новизны |
| `--help` | Любая команда | Справка, например `python -m lctrend fetch --help` |

### Временной датасет технологий

Одна строка — технология на дату `snapshot_date` (`snapshot_id`). Обе CSV-команды используют единый `TemporalCorpus`: признаки строятся только из версий документов, упоминаний, связей, зрелости, экономических свидетельств и наблюдений метрик, доступных к этой дате. Поздняя выгрузка старой статьи не делает её метрики историческими наблюдениями. Существующий граф без дат версий и наблюдений может дать неполные исторические срезы.

```powershell
python -m lctrend export-features --snapshot 2020-01-01 --output artifacts/features-2020.csv
python -m lctrend build-training-set --start-year 2015 --horizon-years 3 --end-date 2025-01-01 --output artifacts/dataset.csv --subgraphs-output artifacts/subgraphs.jsonl
```

Признаки покрывают динамику публикаций, кода, пакетов и патентов; независимость и концентрацию источников; участников и связи; зрелость и экономические сигналы; семантическую новизну и положение в графе. Недоступное значение сохраняется пустым с индикатором пропуска: отсутствие собранного источника отличается от наблюдаемого нуля. Рядом с CSV сохраняется `<output>.manifest.json` с описанием схемы и отдельным списком входных признаков.

`label_realized` отражает реализацию технологии в интервале после среза до `horizon_end`: независимые подтверждения и результат внедрения либо коммерциализации. Число будущих публикаций само по себе не даёт положительную метку. `future_*` — будущие результаты для формирования и проверки метки; их нельзя подавать модели как признаки. При незавершённом горизонте, неизвестной исходной зрелости или недостаточном покрытии источников метка остаётся пустой с причиной, а не превращается в 0. Пороги и параметры заданы в `resources/dataset.json`; прежние `--positive-future-documents` и `--negative-future-documents` удалены.

Отрицательная метка требует полного наблюдения научных, программных, пакетных, патентных и коммерческих источников в горизонте. Текущие ограниченные CLI-обходы сохраняют аудит с `exhaustive=false` и не доказывают отсутствие реализации; без полного покрытия многие строки останутся без метки.

`split` содержит временное разбиение: последние размеченные срезы идут в `test`; для `valid` выбираются последние более ранние срезы, чей горизонт заканчивается до начала теста. Строки, чей горизонт пересекает начало следующей части, получают `purged`; строки без метки — `unlabeled`. Для обучения используйте только `train`, для оценки — соответствующие `valid`/`test`, и список признаков из описания схемы. Если размеченной истории недостаточно, одна из частей может быть пустой.

JSONL-подграфы содержат типизированные узлы и связи, ограниченные датой среза и лимитами `dataset.json → subgraph`; маска использует 1 для пропуска и 0 для наблюдаемого значения. Рядом сохраняется manifest со схемой признаков узлов и рёбер. `graph.subgraphs.to_pyg(sample, feature_schema=manifest)` преобразует образец в `torch_geometric.data.HeteroData` с одинаковым порядком колонок, когда установлены необязательные `torch` и `torch-geometric`. Экспорт CSV/JSONL не требует этих библиотек. Обучение Graph Transformer и итоговое ранжирование пока не реализованы.

### Извлечение: hybrid по умолчанию

`--extractor hybrid` (по умолчанию, также `LCTREND_EXTRACTOR`) — LLM-пакеты плюс GLiNER как дополнительный фактор. GLiNER находит spans технологий, методов и задач; они попадают в payload экстрактора как `ner_hints` (навигация, не доказательство). После LLM совпавшие по типу spans подтверждают сущности (`confidence` упоминания = score GLiNER), а сильные технологии, которые LLM пропустила (score ≥ 0.7), добавляются как `mention_role=ner_candidate` без утверждений. Сводка — в `run.metadata.ner`, пороги — в разделе `hybrid` файла extraction.json. Если GLiNER не установлен (`pip install -e ".[ner]"`) или упал, документ обрабатывается как `llm` с предупреждением.

`--extractor llm` — только LLM. `--extractor gliner` — прежний путь без LLM; локальные regex-утверждения в нём остаются needs_review/unverified.

### GigaChat и лестница моделей

```
LLM_PROVIDER=gigachat
GIGACHAT_CREDENTIALS=<ключ авторизации из личного кабинета>
GIGACHAT_SCOPE=GIGACHAT_API_PERS          # B2B: GIGACHAT_API_B2B, pay-as-you-go: GIGACHAT_API_CORP
GIGACHAT_CA_BUNDLE_FILE=russian_trusted_root_ca.crt   # если TLS требует сертификат НУЦ Минцифры
```

Клиент получает OAuth-токен и обновляет его перед истечением срока, использует endpoint `https://api.giga.chat/v1` и structured output (`response_format: json_schema`). Срок токена и значения scope описаны в [документации авторизации GigaChat](https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/gigachat-api).

В `resources/llm.json` задана лестница: `GigaChat-3-Ultra` → `GigaChat-2-Max` → `GigaChat-2-Pro` → `GigaChat-2`. Это конфигурация клиента; доступность моделей зависит от аккаунта и ответа API. Модель выбывает, когда `/balance` показывает остаток меньше резерва (20 000 + лимит ответа), либо при HTTP 402/403/404. Запрос переходит к следующей модели; причины — в `run.metadata.model_events`, фактическая модель каждого вызова — в `run.metadata.provider_calls`. `LLM_MODEL` выбирает стартовую модель; нижние ступени остаются резервными, если её имя входит в лестницу. Если имени в лестнице нет, используется только эта модель. `LLM_MODEL_LADDER` задаёт собственный порядок для обоих провайдеров.

### Совместимый LLM-провайдер

```dotenv
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=your-key
LLM_MODEL=your-model-name
```

Это шаблон: замените URL, ключ и имя модели своими значениями. Клиент добавляет `/chat/completions` к базовому URL. Нужны JSON-ответы (`json_object` или `json_schema`, задаётся в `llm.json`). Для удалённого endpoint ключ обязателен; для `localhost`, `127.0.0.1`, `::1` может быть пустым. Вместо общей модели можно задать обе модели стадий или лестницу.

## Что означает каждая настройка .env

В `.env.example` выбран GigaChat. Заполните его ключ авторизации либо смените `LLM_PROVIDER` и задайте параметры совместимого API. Явные параметры CLI имеют приоритет над соответствующими переменными. `.env` не перекрывает экспортированные переменные; в Docker сохранённые настройки LLM из `LCTREND_SETTINGS_FILE` имеют дополнительный приоритет.

| Переменная | Назначение и значение по умолчанию |
|---|---|
| `NEO4J_URI` | Адрес внешней Neo4j, например `neo4j+s://…databases.neo4j.io`. Адрес образца — шаблон, его нужно заменить |
| `NEO4J_USER` | Пользователь Neo4j, по умолчанию `neo4j` |
| `NEO4J_PASSWORD` | Пароль существующей внешней базы; образец `change-me-now` не подключает её автоматически |
| `POSTGRES_DB` | Имя базы PostgreSQL в Compose, `lctrend`; текущая загрузка графа её не использует |
| `POSTGRES_USER` | Пользователь PostgreSQL в Compose, `lctrend` |
| `POSTGRES_PASSWORD` | Пароль PostgreSQL; замените значение образца при первоначальной настройке |
| `POSTGRES_PORT` | Внешний порт PostgreSQL, по умолчанию 5432; backend внутри Docker обращается к порту 5432 контейнера |
| `FRONTEND_PORT` | Порт веб-интерфейса на компьютере, по умолчанию 5188 |
| `GITHUB_TOKEN` | Необязательный токен для запросов к GitHub API |
| `OPENALEX_API_KEY` | Ключ из [настроек OpenAlex](https://openalex.org/settings/api), рекомендован для массовой загрузки; backend передаёт его в запросах API |
| `OPENALEX_MAILTO` | Необязательный контактный email для запросов OpenAlex; это не ключ API или адрес статьи |
| `LLM_PROVIDER` | `gigachat` или `openai_compatible`; в образце — GigaChat, без настройки — совместимый провайдер из `llm.json` |
| `GIGACHAT_CREDENTIALS` | Ключ авторизации из кабинета GigaChat для получения OAuth-токена; не сам временный access token |
| `GIGACHAT_SCOPE` | `GIGACHAT_API_PERS` для физлиц, `GIGACHAT_API_B2B` для пакетного доступа организаций, `GIGACHAT_API_CORP` для pay-as-you-go |
| `GIGACHAT_BASE_URL` | Необязательный базовый URL GigaChat; пусто — `https://api.giga.chat/v1` |
| `GIGACHAT_AUTH_URL` | Необязательный URL токена; по умолчанию `https://ngw.devices.sberbank.ru:9443/api/v2/oauth` |
| `GIGACHAT_CA_BUNDLE_FILE` | Необязательный путь к доверенным TLS-сертификатам GigaChat; указанный файл должен существовать |
| `LLM_BASE_URL` | Базовый URL совместимого API, используется при `openai_compatible`; без `/chat/completions` |
| `LLM_API_KEY` | Ключ совместимого API; GigaChat использует отдельный `GIGACHAT_CREDENTIALS` |
| `LLM_CA_BUNDLE_FILE` | Общий необязательный CA bundle; для GigaChat действует, если его отдельный bundle не задан |
| `LLM_MODEL` | Общая стартовая модель извлечения и проверки; пусто — лестница или модели обеих стадий |
| `LLM_EXTRACT_MODEL` | Стартовая модель извлечения; приоритет выше `LLM_MODEL` |
| `LLM_REVIEW_MODEL` | Стартовая модель проверки поддержки утверждений текстом; приоритет выше `LLM_MODEL` |
| `LLM_MODEL_LADDER` | Имена моделей через запятую в порядке переключения; пусто — порядок из `llm.json` |
| `LCTREND_EXTRACTOR` | Режим CLI: `hybrid`, `llm`, `gliner`; пусто — `hybrid`. Веб-обход по теме использует свой режим `hybrid` |
| `GLINER_MODEL` | Имя модели GLiNER; пусто — `urchade/gliner_medium-v2.1` |
| `DEDUP_EMBEDDING_PROVIDER` | Источник эмбеддингов resolver: `gigachat` (по умолчанию, ключи `GIGACHAT_*`) или `transformers` (локальная модель) |
| `DEDUP_EMBEDDING_MODEL` | Эмбеддинг-модель resolver; пусто — `EmbeddingsGigaR` для GigaChat, `sentence-transformers/all-MiniLM-L6-v2` для `transformers` |
| `DEDUP_IN_LLM` | Необязательный `0`/`1`, переопределяет `resolver.json → semantic.use_in_llm` для режимов llm/hybrid |
| `DEDUP_DECISION_MODEL` | Cross-encoder проверки пары; пусто — `cross-encoder/stsb-distilroberta-base` |
| `DEDUP_COSINE_THRESHOLD` | Минимальный косинус для передачи пары cross-encoder, по умолчанию 0.78 |
| `DEDUP_DECISION_THRESHOLD` | Минимальная оценка cross-encoder для кандидата `ambiguous`, по умолчанию 0.80 |
| `LCTREND_CONFIG_DIR` | Каталог с вашими версиями JSON-каталогов, схемы и промптов; одноимённые файлы заменяются целиком |
| `LCTREND_RAW_DIR` | Каталог снимков исходных байтов, по умолчанию `artifacts/raw` |
| `LCTREND_UPLOAD_DIR` | Файлы, загруженные через HTTP API; по умолчанию `artifacts/ingestion/uploads` |
| `LCTREND_SETTINGS_FILE` | Файл сохранённых настроек LLM и OpenAlex; Compose задаёт `/app/artifacts/ingestion/settings.env`. Без него локальная веб-форма пишет в `.env` |
| `LCTREND_LOG_LEVEL` | Уровень сообщений в консоли, по умолчанию `INFO` |
| `LCTREND_LOG_FILE` | Файл журнала, по умолчанию `logs/lctrend.log` |
| `LCTREND_LOG_FILE_LEVEL` | Уровень файлового журнала, по умолчанию `DEBUG` |

Пути относительны каталогу запуска; внутри Docker нужны пути контейнера. Пустые необязательные настройки означают штатные значения. У совместимого провайдера нужно задать модель/лестницу: встроенной лестницы для него нет. Пустой ключ в веб-форме сохраняет прежний ключ; во время активной обработки настройки менять нельзя. Изменение пароля PostgreSQL в `.env` не меняет пароль уже созданной базы в Docker volume.

### Локальный Python без Docker

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[llm,pdf,dev]"
# Если .env ещё нет: Copy-Item .env.example .env
# Настройте внешнюю Neo4j и выбранного LLM-провайдера.
python -m lctrend init-graph
```

Базовый пакет требует Python >= 3.9; дополнительные библиотеки могут требовать более новую версию. `[llm]` устанавливает httpx, `[pdf]` — Docling, `[ner]` — GLiNER, `[gui]` — HTTP-сервер и загрузку файлов, `[dev]` — pytest. Можно объединить: `python -m pip install -e ".[gui,llm,ner,pdf,dev]"`. Для `parse file` база и LLM не нужны. Все команды запускаются из `LCTrendSearch`, где лежит `.env`.

## Что происходит на каждом этапе

| Этап | Действие | Что проверять |
|---|---|---|
| Получение | Читает локальный файл или API; сетевой OpenAlex пытается получить PDF | `source`, `artifact`, `metadata.fulltext`, сохранённые байты |
| Разбор | Формирует DocumentEnvelope и адресуемые chunks | Текст, `locator`, покрытие и предупреждения |
| Связанные пакеты | Собирает соседние фрагменты в ограниченный контекст LLM | Какие chunks включены, бюджеты `pipeline.json` |
| GLiNER в hybrid | Предлагает упоминания в `ner_hints` | `run.metadata.ner`; подсказка не является доказательством утверждения |
| LLM-извлечение | Извлекает сущности и атомарные утверждения с ролями и цитатами | Evidence, сущности, assertions, журнал вызовов |
| Проверка | Код проверяет цитаты/координаты/типы, отдельный reviewer — поддержку текстом | Validation/review и причины отклонения |
| Сборка и идентификация | Объединяет пакеты, сопоставляет сущности с реестром | Неоднозначности, provisional-сущности |
| Публикация | Сохраняет документ, аудит и допустимые связи в Neo4j | ProcessingRun, статус, опубликованные связи |

## Проверки и статусы

Extractor возвращает локальные сущности и атомарные утверждения. Код проверяет цитаты, координаты, роли, типы и величины. Reviewer отдельно проверяет поддержку утверждения оригинальным текстом. При нехватке контекста, сбое проверки или неоднозначной идентичности подтверждённая связь не создаётся.

ProcessingRun хранит бюджеты, попытки вызовов, результаты проверки, причины ошибок, непроверенные утверждения и неполное покрытие. Повторы ограничены и учитываются в общем бюджете. Поддержка текстом не означает доказанную истинность источника.

Неполный/неудачный разбор сохраняется в аудите со staged_result. Он не стирает активный граф предыдущей успешной обработки; публикация структурной проекции выполняется при succeeded.

Прямое SOLVES допустимо только для accepted/supported положительного solves_task с reported/observed модальностью. Совместное упоминание технологии и задачи больше не считается таким основанием. Новые сущности остаются provisional; коллизии имён — ambiguous. Экономические regex-сведения остаются кандидатами.

## Настройки и проверка

Предметные словари, правила, предикаты, промпты и бюджеты вынесены в src/lctrend/resources. `LCTREND_CONFIG_DIR` заменяет одноимённый файл целиком; остальные берутся из пакета. Промпты переопределяются файлами prompts/extract.txt и prompts/review.txt.

Назначение каждого каталога — в [docs/catalogs.md](docs/catalogs.md). `pipeline.json` задаёт размеры пакетов, лимиты контекста, общий бюджет LLM-вызовов и ограничения файлов; `llm.json` — клиент и ответы; `llm_schema.json` — допустимые отношения и роли. Большая статья может получить неполное покрытие при исчерпании бюджета даже после успешного разбора PDF.

Каталоги также задают экономические утверждения, отрасли, рынки и роли стран. По запросу `search_graph` обе LLM-стадии получают исходные блоки ранее обработанных документов Neo4j как дополнительный контекст. Цитаты утверждений остаются привязаны к текущему документу. Лимиты находятся в `pipeline.json → graph_context`; дополнительные секреты в `.env` не требуются. CLI и веб-сервер проверяют каталоги при старте; новые правила требуют переобработки уже загруженных документов.

```powershell
python -m pytest
```

Тесты используют сохранённые ответы, фиктивные LLM и транзакции, а также временной корпус с будущими наблюдениями для проверки отсутствия утечки. Реальные LLM, Neo4j и качество OCR отдельно требуют проверки на ваших материалах. Существующая база и CSV автоматически не пересчитываются; повторный запуск export-features/build-training-set создаёт новый формат временного датасета.
