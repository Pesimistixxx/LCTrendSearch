# Списки строк в парсерах

Списки вынесены в файлы данных, но их роли различаются. Перенос в JSON сам по себе не делает эвристику достоверной.

| Файл в src/lctrend/resources | Что за строки |
| --- | --- |
| sources.json | domains — ограниченный словарь широких областей и слов для сопоставления с метаданными. Это не перечень найденных технологий и не критерий новизны. Остальное: API, поля источников, организации, параметры разбиения текста. |
| extraction.json | ner.labels — типы извлекаемых сущностей (Technology, Task, Metric и др.). generic_technologies — слишком общие названия. composite_rules выключены: соседство слов не доказывает составную технологию. Разделы assertions/economics содержат шаблоны кандидатов утверждений, отрицаний, планов, денежных слов и валют. |
| resolver.json | explicit_aliases — явно заданные эквивалентные названия одного типа; сейчас NLP / natural language processing. Semantic-настройки (эмбеддинги MiniLM + cross-encoder) задают модельное предложение идентичности, требующее проверки; они действуют только в режиме `gliner`. Совпадающие инициалы больше не объединяют сущности. |
| graph.json, schema.cypher | Отображения ролей и типов связей, ограничения Neo4j. Это контракт хранения, не исследовательская база. |
| runtime.json | Прежние технические значения по умолчанию: модель, размеры сборки, пути и параметры существовавших команд. Новых настроек поиска нет. |
| pipeline.json | Размеры связанных пакетов, лимиты контекста и вызовов, форматы файлов, разрешённые структурные метаданные, ограничения и предупреждения адаптеров. |
| llm_schema.json | Допустимые отношения, обязательные роли, типы участников и проверки числовых значений. Это контракт извлечения, не список технологий или трендов. |
| llm.json | Настройки клиента LLM, тайм-ауты, лимиты ответа, выключенный по умолчанию кэш. Модель и ключ задаются отдельно. |
| prompts/extract.txt, prompts/review.txt | Инструкции двух LLM-стадий: извлечь атомарные утверждения с основаниями и независимо проверить их поддержку текстом. |

Кандидат Assertion означает непроверенное извлечённое утверждение. Это не технологический кандидат для рекомендаций. Совпадение regex не устанавливает истинность; отсутствие совпадения может пропустить факт.

Enums, состояния моделей, обработчики команд и схема исходного CSV-экспорта остаются программными контрактами. Поиска и ранжирования кандидатов в бэкенде нет: экран «Демо поиска» показывает демонстрационные данные.

Каталоги включены в пакет. Для локального переопределения используйте LCTREND_CONFIG_DIR: одноимённый файл заменяется полностью, остальные берутся из пакета. Изменения извлечения требуют явной переобработки документов.

## Участники, страны, зрелость и таксономия

`llm_schema.json` (v2) добавляет предикаты с ролями `organization`, `country`, `parent`:

| Предикат | Роли | Ребро в графе (только принятое, affirmed, reported/observed) |
| --- | --- | --- |
| developed_by | subject → organization | `(Technology)-[:DEVELOPED_BY]->(Company/University/Organization)` |
| used_by | subject → organization | `USED_BY` |
| funded_by | subject → organization | `FUNDED_BY` |
| developed_in | subject → country | `DEVELOPED_IN` |
| located_in | organization → country | `LOCATED_IN` (с `document_version_id`) |
| subtechnology_of | subject → parent | `SUBTECHNOLOGY_OF` |
| reports_maturity_stage | subject; `qualifiers.stage` обязателен, `qualifiers.trl` только если число написано в цитате | `(Technology)-[:HAS_MATURITY_EVIDENCE {stage, stage_rank, trl}]->(Chunk)` |

Каждое такое ребро несёт `chunk_id`, `quote`, `start`/`end`, `assertion_id`, `run_id`, `observed_at`. Проекции задаются в `graph.json → projections`. Сущность Country из текста получает `country_code` (ISO 3166-1 alpha-2): код становится именем концепта и связывается `SAME_AS` с узлом `Country` из метаданных.

## Метаданные локальных файлов (sidecar)

Для файла `report.pdf` рядом можно положить `report.pdf.meta.json`. Все поля необязательны. Без sidecar дата публикации берётся только из HTML-тегов публикации (`citation_publication_date`, `article:published_time` и др.). Дата создания DOCX/PDF сохраняется как `metadata.file_created_at` и публикацией не считается. Документ без даты в признаки не попадает.

```json
{
  "title": "Обзор рынка сервисной робототехники",
  "published_at": "2025-11-20",
  "language": "ru",
  "document_type": "report",
  "source": {"name": "Минпромторг", "type": "government", "family": "regulatory", "reliability_tier": 3, "url": "https://example.org/report"},
  "authors": [{"name": "Иван Петров", "affiliations": ["ООО Роботех"]}],
  "organizations": [{"name": "ООО Роботех", "type": "company", "country": "RU", "role": "associated"}],
  "countries": ["RU"],
  "domains": ["Robotics"],
  "identifiers": {"doi": "10.1000/example"}
}
```

`document_type`: article, patent, repository, package, report, transcript, standard, job_posting, grant, regulatory, news. Откуда взято каждое значение, записано в `metadata.metadata_basis`.

## Лексический и смысловой слой сопоставления

1. Лексический: нормализация названия + лемма (`simplemma`, теперь обязательная зависимость) + группы синонимов `resolver.json → explicit_aliases`. Совпадение с единственным совместимым концептом — принятое тождество.
2. Смысловой (`resolver.json → semantic`): эмбеддинги GigaChat `EmbeddingsGigaR` (или локальный MiniLM) и cross-encoder. В режимах `llm`/`hybrid` включён параметром `use_in_llm` (переменная `DEDUP_IN_LLM=0/1` переопределяет). Совпадение не склеивает концепты: упоминание получает свой provisional-концепт, а найденный кандидат записывается ребром `POSSIBLY_SAME_AS {score, cosine, review_status: 'pending'}` для проверки. В режиме `gliner` поведение прежнее (`ambiguous`).
3. Недоступность эмбеддингов или cross-encoder не валит документ: сопоставление идёт лексически, слой повторно пробуется через 5 минут, причина пишется в `run.metadata.semantic`.
4. Векторы названий концептов видов из `embedded_kinds` сохраняются в `c.embedding` / `c.embedding_model`; векторный индекс `<label>_embedding` создаётся при первой записи, когда известна размерность. Это задел для таксономии (кластеризация) и поиска похожих концептов; сама таксономия (TaxoGen) пока не реализована.
