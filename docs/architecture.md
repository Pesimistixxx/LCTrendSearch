# Пайплайн LCTrendSearch: от источника до признаков слабого сигнала

## Суть

Единица анализа — **технология в момент времени**, а не отдельная статья. Система собирает документы из разных типов источников. LLM извлекает из текста технологии, участников и утверждения, у каждого из которых есть дословная цитата. Всё складывается во временной граф знаний Neo4j. По графу и таксономии технологий считаются признаки на дату среза: рост, разнообразие источников, участники, зрелость, экономика, новизна. Эти признаки — вход для будущего классификатора слабых сигналов.

Статусы на схемах: зелёный — реализовано, жёлтый — частично, красный пунктир — не реализовано.

## 1. Общая схема

```mermaid
flowchart TD
    classDef done fill:#d6f5d6,stroke:#2e7d32,color:#000
    classDef partial fill:#fff4c2,stroke:#b58900,color:#000
    classDef missing fill:#fde0e0,stroke:#c62828,color:#000,stroke-dasharray: 5 3
    classDef store fill:#dde8ff,stroke:#1a4fb5,color:#000

    subgraph A["1. Сбор — ingest/connectors.py, discovery.py, fulltext.py"]
        A1["fetch_openalex_page / discover_openalex<br/>статьи, аннотации, топики, спонсоры"]:::done
        A2["attach_openalex_fulltext<br/>PDF → Docling, section_role"]:::done
        A3["fetch_github / discover_github<br/>README, релизы, владелец"]:::done
        A4["fetch_pypi / fetch_pypi_projects"]:::done
        A5["parse_epo (локальный XML OPS)"]:::done
        A6["parse_file: txt/md/html/docx/pdf<br/>+ sidecar *.meta.json"]:::done
        A7["Вакансии, гранты, стандарты,<br/>регуляторика, новости"]:::missing
    end

    subgraph B["2. Нормализация — ingest/adapters.py, snapshots.py"]
        B1["persist_snapshot<br/>сырые байты по sha256"]:::done
        B2["parse_openalex / parse_github / parse_pypi / parse_epo<br/>→ DocumentEnvelope"]:::done
        B3["_domains_from_values / _domains_from_topics<br/>домены каталога + подполя OpenAlex"]:::partial
        B4["_organization_type<br/>GmbH/Inc → company, Univ → university"]:::done
        B5["_markdown_chunks, Docling, абзацы → Chunk"]:::done
    end

    subgraph C["3. Извлечение — extraction/processing.py → llm/pipeline.py"]
        C0["process_material(mode=hybrid|llm|gliner|none)"]:::done
        C1["NerRuntime + extract_mentions<br/>GLiNER: подсказки"]:::done
        C2["plan_packets → ContextPacket<br/>PipelineSettings.call_limit"]:::done
        C3["build_payload: чанки + карта документа<br/>+ метаданные + контракт предикатов + подсказки"]:::done
        C4["JsonLLM.generate(Extraction)<br/>prompts/extract.txt"]:::done
        C5["expand_packet: read_chunk / search_chunks<br/>(только внутри документа)"]:::partial
        C6["validate_local_extraction<br/>цитаты, роли, stage, TRL, ISO-код"]:::done
        C7["JsonLLM.generate(Review)<br/>prompts/review.txt + validate_review"]:::done
        C8["extract_economic_evidence + amount_value"]:::done
    end

    subgraph D["4. Сопоставление — extraction/resolver.py"]
        D1["ConceptRegistry / ConceptIndex<br/>реестр концептов из графа"]:::done
        D2["Лексический слой: normalize_name,<br/>lemmatize_name (simplemma), explicit_aliases"]:::done
        D3["SemanticDeduplicator: эмбеддинги GigaChat<br/>+ cross-encoder → POSSIBLY_SAME_AS"]:::done
        D4["resolve_mentions → Concept, ResolutionDecision"]:::done
        D5["_concept_embeddings → векторы названий"]:::done
    end

    subgraph E["5. Граф — graph/store.py: GraphStore"]
        E1[("write_processed: документный слой")]:::store
        E2[("_write_extraction: концепты, MENTIONS,<br/>Assertion, SUPPORTED_BY, эмбеддинги")]:::store
        E3[("_projection_links: SOLVES, DEVELOPED_BY, USED_BY,<br/>FUNDED_BY, DEVELOPED_IN, LOCATED_IN,<br/>SUBTECHNOLOGY_OF, HAS_MATURITY_EVIDENCE")]:::store
        E4["ensure_vector_indexes"]:::done
    end

    subgraph F["6. Таксономия — taxonomy/builder.py"]
        F1["read_taxonomy_input<br/>концепты с эмбеддингом + даты + SUBTECHNOLOGY_OF"]:::done
        F2["build_taxonomy: рекурсивный сферический k-means,<br/>общие термины остаются у родителя"]:::done
        F3["write_taxonomy: TaxonomyNode, CHILD_OF, IN_TAXONOMY"]:::done
        F4["taxonomy_features: semantic_novelty,<br/>new_branch_in_known_area, branch_growth"]:::done
    end

    subgraph G["7. Признаки и модель — graph/training.py"]
        G1["build_feature_rows (export-features)"]:::done
        G2["build_training_rows (build-training-set)<br/>метка: рост за 3 года; без таксономии"]:::partial
        G3["Хайп и шум: только reliability_tier,<br/>independence_group и их разнообразие"]:::partial
        G4["Отбор кандидатов"]:::missing
        G5["Graph Transformer / HGT"]:::missing
        G6["Интерпретируемый классификатор + SHAP"]:::missing
        G7["Объяснение и TOP-15"]:::missing
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    A5 --> B1
    A6 --> B1
    A7 -.-> B1
    B1 --> B2
    B2 --> B3
    B2 --> B4
    B2 --> B5
    B5 --> C0
    C0 --> C1
    C0 --> C2
    C1 --> C3
    C2 --> C3
    C3 --> C4
    C4 -- "context_requests" --> C5
    C5 --> C3
    C4 --> C6
    C6 --> C7
    C7 --> D4
    D1 --> D4
    D2 --> D4
    D3 --> D4
    D4 --> D5
    B5 --> C8
    D4 --> E2
    D5 --> E2
    C8 --> E3
    B2 --> E1
    C7 -- "принято, affirmed, reported/observed" --> E3
    E2 --> E4
    E2 --> D1
    E2 --> F1
    E3 --> F1
    F1 --> F2
    F2 --> F3
    F2 --> F4
    E1 --> G1
    E2 --> G1
    E3 --> G1
    F4 --> G1
    E1 --> G2
    G1 --> G3
    G1 -.-> G4
    G3 -.-> G4
    G4 -.-> G5
    G5 -.-> G6
    G2 -.-> G6
    G6 -.-> G7
```

## 2. Этапы по классам

| # | Этап | Модуль / класс | Вход → выход |
| --- | --- | --- | --- |
| 1 | Сбор | `connectors.fetch_*`, `discovery.discover_*`, `CrawlManager` (веб) | API → сырой JSON/XML/PDF |
| 2 | Снимок | `snapshots.persist_snapshot` | байты → файл `artifacts/raw/<sha256>`; можно перепарсить без повторной загрузки |
| 3 | Адаптер | `adapters.parse_*`, `file_adapters.parse_file` | сырой ответ → `DocumentEnvelope` (метаданные + `Chunk`) |
| 4 | Полный текст | `fulltext.attach_openalex_fulltext` | PDF → чанки `fulltext` с `section_role` |
| 5 | Оркестрация | `processing.process_material`, `JobManager` (веб) | режим `hybrid`/`llm`/`gliner`/`none` |
| 6 | Пакеты | `context.plan_packets`, `ContextPacket`, `PipelineSettings` | чанки → пакеты по 10, бюджет 3 вызова на пакет, не больше 120 |
| 7 | Извлечение | `pipeline.process_document`, `JsonLLM`, `contracts.Extraction` | пакет → `LocalEntity`, `LocalClaim`, `ContextRequest` |
| 8 | Проверка кодом | `validation.validate_local_extraction` | отбраковка сущностей и утверждений без цитат или с неверными ролями |
| 9 | Рецензия | `contracts.Review`, `validation.validate_review` | supported / unsupported / unclear для каждого утверждения |
| 10 | Сопоставление | `resolver.resolve_mentions`, `ConceptRegistry`, `SemanticDeduplicator` | упоминания → `Concept`, `ResolutionDecision` |
| 11 | Экономика | `economics.extract_economic_evidence`, `amount_value` | предложения → `EconomicEvidence` |
| 12 | Результат | `core.models.ExtractionResult` | mentions, concepts, assertions, resolutions, economic_evidence, concept_embeddings |
| 13 | Запись | `GraphStore.write_processed` | документ + результат в одной транзакции |
| 14 | Таксономия | `taxonomy.build_taxonomy`, `GraphStore.write_taxonomy` | эмбеддинги концептов → дерево `TaxonomyNode` на дату среза |
| 15 | Признаки | `training.build_feature_rows`, `taxonomy_features` | граф + дерево → CSV признаков по технологиям |
| 16 | Обучающая выборка | `training.build_training_rows` | граф → строки с меткой «рост за 3 года» |

## 3. Модель данных

```mermaid
classDiagram
    class DocumentEnvelope {
        document_id, document_version_id
        document_type, title, language
        published_at, retrieved_at
        source: SourceRef
        artifact: Artifact
        contributors, organizations
        countries, domains
        chunks: Chunk[]
        metadata, metrics, coverage
    }
    class Chunk {
        chunk_id, kind, text, order
        section_path, locator
    }
    class ExtractionResult {
        run: ProcessingRun
        mentions: Mention[]
        concepts: Concept[]
        resolutions: ResolutionDecision[]
        assertions: Assertion[]
        economic_evidence: EconomicEvidence[]
        concept_embeddings, embedding_model
    }
    class Concept {
        concept_id, kind: ConceptKind
        preferred_label, status, names
    }
    class Assertion {
        predicate, roles
        evidence: EvidenceSpan[]
        polarity, modality
        status, verification_status
    }
    class Taxonomy {
        version, snapshot
        nodes: TaxonomyNode
        placement, general_terms
    }
    DocumentEnvelope "1" --> "*" Chunk
    ExtractionResult "1" --> "*" Concept
    ExtractionResult "1" --> "*" Assertion
    Assertion "*" --> "*" Concept : roles
    Assertion "*" --> "*" Chunk : evidence
    Taxonomy "1" --> "*" Concept : placement
```

## 4. Граф Neo4j

```mermaid
flowchart LR
    Source --- |FROM_SOURCE| DocumentVersion
    Document -->|HAS_VERSION| DocumentVersion
    DocumentVersion -->|HAS_CHUNK| Chunk
    Document -->|WRITTEN_IN / JURISDICTION| Country
    Document -->|ABOUT_DOMAIN| Domain
    Domain -->|SUBDOMAIN_OF| Domain
    Document -->|HAS_AFFILIATION / OWNED_BY / APPLIED_BY / FUNDED_BY| Organization
    Document -->|CONTRIBUTED_BY| Contributor
    Contributor -->|AFFILIATED_WITH| Organization
    Organization -->|LOCATED_IN| Country
    ProcessingRun -->|PROCESSED| DocumentVersion
    ProcessingRun -->|CREATED| Assertion
    DocumentVersion -->|HAS_ASSERTION| Assertion
    Assertion -->|SUBJECT / TASK / ORGANIZATION / COUNTRY / PARENT ...| Technology
    Assertion -->|SUPPORTED_BY| Chunk
    Chunk -->|MENTIONS| Technology
    Technology -->|SOLVES| Task
    Technology -->|DEVELOPED_BY / USED_BY / FUNDED_BY| Company
    Technology -->|DEVELOPED_IN| CountryConcept["Country (из текста)"]
    CountryConcept -->|SAME_AS| Country
    Technology -->|SUBTECHNOLOGY_OF| Technology
    Technology -->|POSSIBLY_SAME_AS| Technology
    Technology -->|HAS_MATURITY_EVIDENCE / HAS_ECONOMIC_EVIDENCE| Chunk
    Technology -->|IN_TAXONOMY| TaxonomyNode
    TaxonomyNode -->|CHILD_OF| TaxonomyNode
```

Правило для прямых связей технологий (`SOLVES`, `DEVELOPED_BY` и остальных): связь создаётся только из утверждения, которое рецензент подтвердил (`supported`), которое утвердительное (`affirmed`) и описывает сообщённое или наблюдаемое (`reported`/`observed`), а не план. На связи хранятся `chunk_id`, `quote`, `start`/`end`, `assertion_id`, `run_id`, `observed_at`.

## 5. Три слоя «понимания» технологий

| Слой | Где | Что делает | Ограничения |
| --- | --- | --- | --- |
| Лексический | `normalize_name`, `lemmatize_name`, `resolver.json → explicit_aliases` | одинаковые названия после нормализации и лемматизации, плюс 14 групп синонимов → один концепт | русские согласованные формы («языковые модели» ≠ «языковая модель»); аббревиатуры склеиваются только по списку |
| Смысловой | `SemanticDeduplicator` | эмбеддинг названия; при сходстве ≥ 0.78 и оценке cross-encoder ≥ 0.80 — связь `POSSIBLY_SAME_AS` для проверки человеком | сам концепты не склеивает; без эмбеддингов работает только лексический слой |
| Таксономия | `build_taxonomy` | дерево тем по эмбеддингам на дату среза; явные `SUBTECHNOLOGY_OF` подтягивают дочерние технологии к родителю | метки узлов — самые представительные названия, а не сгенерированные имена; строится только из концептов с эмбеддингом |

Как работает `build_taxonomy`, по шагам:
1. Берутся концепты видов `Technology`, `Method`, `Material`, известные на дату среза.
2. Вектор дочерней технологии из `SUBTECHNOLOGY_OF` сдвигается к родителю (вес 0.5).
3. Сферический k-means делит множество на `branching`=4 кластера (инициализация k-means++, фиксированный seed).
4. Термин, одинаково близкий к нескольким кластерам (разница < 0.05), или кластер меньше 3 терминов остаётся у родителя как общий. Так делает TaxoGen.
5. Рекурсия до глубины 4.
6. Метка узла — три самых популярных и центральных названия.
7. Для узла считаются документы за последний и предыдущий год и доля новых терминов (появились позже, чем за год до среза).

Признаки таксономии в `export-features`:
- `semantic_novelty` — 1 − косинус до ближайшего концепта, известного год назад;
- `new_branch_in_known_area` — ветка в основном новая, а родительская в основном старая;
- `branch_growth`, `branch_new_share`, `taxonomy_level`, `taxonomy_node_size`, `taxonomy_sibling_count`, `taxonomy_general_term`.

## 6. Все колонки `export-features`

- Активность: `document_count`, `mention_count`, `documents_last_year`, `publication_growth`, `first_seen_date`, `technology_age_days`, `citation_count`.
- Распространение: `country_count`, `company_count`, `university_count`, `domain_count`, `source_type_diversity`, `independence_group_diversity`, `new_relation_count`, `patent_data_available`, `repository_data_available`.
- Участники из текста: `developer_count`, `user_count`, `funder_count`, `text_country_count`, `parent_technology_count`, `task_count`.
- Зрелость и экономика: `max_maturity_rank`, `max_trl`, `economic_evidence_count`, `economic_data_available`.
- Таксономия: 8 колонок из раздела 5.

## 7. Порядок запуска

```bash
lctrend init-graph                                   # ограничения и индексы Neo4j
lctrend crawl-openalex "edge ai" --limit 500         # или веб-интерфейс / crawl-pypi / ingest
lctrend build-taxonomy --snapshot 2026-01-01         # дерево в Neo4j + artifacts/taxonomy/2026-01-01.json
lctrend export-features --snapshot 2026-01-01        # CSV признаков (с таксономией; --no-taxonomy без неё)
lctrend build-training-set                           # строки с меткой роста за 3 года
```

Эмбеддинги концептов появляются, только если во время загрузки работал смысловой слой: `resolver.json → semantic.use_in_llm = true` и заданы ключи GigaChat. Без них таксономия пустая.

## 8. Что не реализовано

- Источники: вакансии, гранты, стандарты, регуляторика, новости (типы документов уже есть в `DocumentType`).
- Поиск контекста по другим документам (RAG) и передача известных концептов в запрос к LLM.
- Склейка организаций из метаданных и из текста.
- Признаки таксономии в `build-training-set` (сейчас только в `export-features`).
- Хайп и шум: нет доли пресс-релизов, поиска дублей, оценки рекламного языка.
- Отбор кандидатов, Graph Transformer, классификатор с SHAP, объяснения, TOP-15.
