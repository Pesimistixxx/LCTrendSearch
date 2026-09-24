# LCTrendSearch

Конвейер загрузки материалов и граф знаний для поиска слабых технологических сигналов.

## Модель графа

Первичная запись — не прямое ребро `Technology -> SOLVES -> Problem`, а узел
`Assertion`, связанный с участниками и буквальным фрагментом-основанием.

Основные узлы: `Source`, `Document`, `DocumentVersion`, `Chunk`, `Mention`,
`Concept`, `ConceptName`, `Assertion`, `ResolutionDecision`, `ClaimGroup`,
`EvidenceFamily`, `Contributor`, `ExternalId`, `ProcessingRun`.

`Concept.kind`: `Technology`, `TechnicalSystem`, `Method`, `Task`, `Problem`,
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

NER создаёт только кандидаты `Mention`. Объединение технологий выполняется
отдельными `ResolutionDecision`; похожесть embeddings сама по себе не означает
идентичность.

Базовый resolver принимает автоматически только точное нормализованное
совпадение с проверенным именем совместимого типа. Всё остальное сохраняется как
`provisional` либо `ambiguous`, поэтому близкие технологии не сливаются молча.

## Проверка

```powershell
pytest
docker compose config
```
