# LCTrendSearch

Конвейер загрузки материалов и граф знаний для поиска слабых технологических сигналов.

## Модель графа

Первичная запись — не прямое ребро `Technology -> SOLVES -> Problem`, а узел
`Assertion`, связанный с участниками и буквальным фрагментом-основанием.

Основные узлы: `Source`, `Document`, `DocumentVersion`, `Chunk`, `Technology`,
`Method`, `Company`, `Country`, `Assertion`, `Contributor`,
`Organization`, `Country`, `Domain`, `ProcessingRun`.

Типы извлекаемых сущностей: `Technology`, `Method`, `Task`, `Problem`,
`ApplicationContext`, `Metric`, `Material`, `Domain`, `MarketSegment`,
`ConceptCandidate`.

PostgreSQL хранит только финальную ранжированную выдачу слабых сигналов.

## Запуск

```powershell
Copy-Item .env.example .env
# Поменяйте пароли в .env
docker compose up -d
python -m pip install -e ".[dev]"
python -m lctrend init-graph
```

Neo4j Browser: <http://localhost:7474>. PostgreSQL: `localhost:5432`.

## Парсеры

Поддержаны OpenAlex JSON, GitHub API JSON, PyPI JSON и EPO OPS XML:

```powershell
python -m lctrend parse openalex fixtures/work.json
python -m lctrend ingest epo fixtures/patent.xml
python -m lctrend fetch pypi gliner --output gliner.json --ingest
python -m lctrend fetch github urchade/GLiNER --ingest
```

GitHub использует `GITHUB_TOKEN`, OpenAlex — необязательный `OPENALEX_MAILTO`.
EPO OPS пока принимает сохранённый XML: live-доступ требует OAuth credentials.

## NER

GLiNER подключён тонким адаптером `lctrend.ner.extract_mentions`. Модель ставится
отдельно, потому что она тяжёлая и не нужна парсерам:

```powershell
python -m pip install -e ".[ner]"
```

NER создаёт внутренние записи упоминаний, которые после разрешения сохраняются
как связи `Chunk-[:MENTIONS]->Technology|Method|...`. Страны, университеты и компании
не извлекаются из чанков: они берутся из структурированных метаданных и связываются
непосредственно с `Document` через `WRITTEN_IN` и `HAS_AFFILIATION`. Координаты, исходное написание,
confidence и способ дедупликации находятся в свойствах связи.

`Domain` — одна широкая нормализованная предметная область документа, например
bioinformatics или edge computing. Она выбирается из короткого канонического
словаря по темам источника, а не извлекается из чанков.
Когда технология и задача явно связаны в одном предложении, создаётся
`Technology-[:SOLVES]->Task` с цитатой-доказательством. Для отображения в Neo4j Browser
у технологии также записывается свойство `name` с её каноническим названием.

Экономические сведения создаются только для предложения, где одновременно явно
упомянуты технология и стоимость, инвестиции, рынок, экономия или коммерциализация.
Они хранятся как `Technology-[:HAS_ECONOMIC_EVIDENCE]->Chunk`; цитата, категория,
сумма и валюта находятся в свойствах связи.

Resolver сначала приводит имена к нижнему регистру, удаляет символы, выполняет
лемматизацию и сопоставляет сокращения (`NLP` ↔ `natural language processing`).
Только для оставшихся кандидатов вычисляется cosine similarity embeddings; пару
выше `DEDUP_COSINE_THRESHOLD` дополнительно проверяет cross-encoder. Порог его
решения задаётся `DEDUP_DECISION_THRESHOLD`.

Каждая типизированная сущность хранит каноническое имя и варианты написания в
свойствах `aliases` и `normalized_aliases`. Общего label `Concept` и отдельных
служебных узлов `Mention`,
`ConceptName` и `ResolutionDecision` в графе нет.

Чтобы запустить NER вместе с загрузкой материала:

```powershell
python -m lctrend fetch pypi gliner --ingest --extract
```

Markdown делится по заголовкам и абзацам; длинные секции ограничиваются 1000
символами и получают overlap до 150 символов внутри одной секции. Внешние
идентификаторы (`doi`, `openalex`, `pypi` и другие) хранятся
в свойстве `external_ids` документа, а не как отдельные узлы графа.

## Проверка

```powershell
pytest
docker compose config
```
