import asyncio
import json
import re

from lctrend.core.models import (
    Assertion,
    Chunk,
    Concept,
    ConceptKind,
    ConceptName,
    EconomicEvidence,
    EvidenceSpan,
    ExtractionResult,
    Mention,
    ProcessingRun,
    ResolutionDecision,
)
from lctrend.extraction.resolver import resolve_mentions
from lctrend.graph.store import GraphStore, _concept_from_properties
from lctrend.ingest.adapters import parse_openalex


def test_graph_roundtrip_does_not_promote_unreviewed_aliases():
    original = Concept(
        concept_id="concept:photon",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="Photon sensor",
        names=[
            ConceptName(
                name_id="name:pending",
                text="Carbon capture",
                normalized_text="carbon capture",
                status="provisional",
            )
        ],
    )
    properties = {
        **original.model_dump(mode="json", exclude={"names"}),
        "aliases": ["Photon sensor"],
        "names_json": json.dumps(
            [item.model_dump(mode="json") for item in original.names]
        ),
    }
    loaded = _concept_from_properties(properties)
    mention = Mention(
        mention_id="mention:new-document",
        chunk_id="chunk:new",
        surface_text="Carbon capture",
        canonical_text="Carbon capture",
        start=0,
        end=14,
        type_candidates=[ConceptKind.TECHNOLOGY],
    )
    _, decisions = resolve_mentions([mention], [loaded])
    assert decisions[0].concept_id != original.concept_id
    assert decisions[0].status == "provisional"


def test_legacy_graph_aliases_keep_unknown_review_status():
    loaded = _concept_from_properties(
        {
            "concept_id": "concept:photon",
            "kind": "Technology",
            "preferred_label": "Photon sensor",
            "aliases": ["Photon sensor", "Carbon capture"],
        }
    )
    assert (
        next(
            item for item in loaded.names if item.text == "Carbon capture"
        ).status
        == "provisional"
    )


class Result:
    def consume(self):
        return None

    def __iter__(self):
        # Stored-kind probe of _settle_family_kinds: no stored nodes.
        return iter(())


class Transaction:
    def __init__(self):
        self.queries = []

    def run(self, query, **parameters):
        assert set(re.findall(r"\$(\w+)", query)).issubset(parameters), (
            "Unbound Neo4j parameter"
        )
        self.queries.append((query, parameters))
        return Result()


def test_reparsed_document_preserves_historical_chunks_and_run_input_links():
    document = parse_openalex(
        {"id": "https://openalex.org/W1", "title": "Original title"}
    )
    tx = Transaction()
    asyncio.run(GraphStore._write_document(tx, document))
    cleanup = next(
        query for query, _ in tx.queries if "WHERE NOT c.chunk_id IN" in query
    )
    assert "DELETE active" in cleanup
    assert "DETACH DELETE" not in cleanup
    result = ExtractionResult(
        document_version_id=document.document_version_id,
        run=ProcessingRun(
            run_id="run:history",
            parser="llm_packets",
            config_hash="config",
            started_at="2026-09-26T00:00:00Z",
        ),
    )
    asyncio.run(GraphStore._write_extraction(tx, document, result))
    query, parameters = next(
        (q, p) for q, p in tx.queries if "MERGE (r:ProcessingRun" in q
    )
    assert "USED_CHUNK" in query
    assert parameters["input_chunk_ids"] == [
        chunk.chunk_id for chunk in document.chunks
    ]


def test_document_projection_builds_queries_without_dynamic_cypher_errors():
    document = parse_openalex(
        {
            "id": "https://openalex.org/W1",
            "title": "Example",
            "authorships": [
                {
                    "author": {
                        "id": "https://openalex.org/A1",
                        "display_name": "Ada",
                    },
                    "institutions": [
                        {
                            "id": "https://openalex.org/I1",
                            "display_name": "Example University",
                            "type": "education",
                            "country_code": "GB",
                        }
                    ],
                }
            ],
        }
    )
    tx = Transaction()
    asyncio.run(GraphStore._write_document(tx, document))
    assert any("MERGE (c:Contributor" in query for query, _ in tx.queries)
    assert any("r.roles" in query for query, _ in tx.queries)
    assert not any("ExternalId" in query for query, _ in tx.queries)
    assert any("AFFILIATED_WITH" in query for query, _ in tx.queries)
    assert any("LOCATED_IN" in query for query, _ in tx.queries)
    assert any("HAS_AFFILIATION" in query for query, _ in tx.queries)
    assert any("WRITTEN_IN" in query for query, _ in tx.queries)
    assert any("SET o:University" in query for query, _ in tx.queries)
    assert any("metrics_json" in query for query, _ in tx.queries)
    assert any("first_seen_at" in query for query, _ in tx.queries)
    assert any("observed_at" in query for query, _ in tx.queries)
    document_query = next(
        parameters
        for query, parameters in tx.queries
        if "MERGE (d:Document" in query
    )
    assert document_query["external_ids"] == ["openalex:w1"]


def test_extraction_stores_aliases_and_resolution_on_relationships():
    tx = Transaction()
    result = ExtractionResult(
        document_version_id="v1",
        run=ProcessingRun(
            run_id="run1", parser="test", config_hash="x", started_at="now"
        ),
        mentions=[
            Mention(
                mention_id="m1",
                chunk_id="ch1",
                surface_text="NLP",
                start=0,
                end=3,
                type_candidates=[ConceptKind.TECHNOLOGY],
            )
        ],
        concepts=[
            Concept(
                concept_id="c1",
                kind=ConceptKind.TECHNOLOGY,
                preferred_label="NLP",
            )
        ],
        resolutions=[
            ResolutionDecision(
                resolution_id="r1",
                mention_id="m1",
                status="accepted",
                concept_id="c1",
            )
        ],
        economic_evidence=[
            EconomicEvidence(
                evidence_id="e1",
                technology_concept_id="c1",
                chunk_id="ch1",
                category="cost",
                quote="NLP",
                start=0,
                end=3,
            )
        ],
    )
    asyncio.run(
        GraphStore._write_extraction(
            tx,
            type(
                "Doc",
                (),
                {
                    "document_version_id": "v1",
                    "published_at": None,
                    "chunks": [],
                    "domains": [],
                },
            )(),
            result,
        )
    )
    queries = "\n".join(query for query, _ in tx.queries)
    assert "ConceptName" not in queries
    assert "ResolutionDecision" not in queries
    assert ":Concept" not in queries
    assert "MERGE (chunk)-[r:MENTIONS" in queries
    assert "MERGE (technology)-[r:HAS_ECONOMIC_EVIDENCE" in queries
    assert "c.name = row.preferred_label" in queries
    assert "c.first_seen_at" in queries
    assert "r.observed_at" in queries
    assert ":Mention" not in queries


def test_solution_link_requires_reviewed_assertion_instead_of_a_sentence_cue():
    result = ExtractionResult(
        document_version_id="v1",
        run=ProcessingRun(
            run_id="run1", parser="test", config_hash="x", started_at="now"
        ),
        mentions=[
            Mention(
                mention_id="m1",
                chunk_id="ch1",
                surface_text="NER",
                start=0,
                end=3,
                type_candidates=[ConceptKind.TECHNOLOGY],
            ),
            Mention(
                mention_id="m2",
                chunk_id="ch1",
                surface_text="text classification",
                start=11,
                end=30,
                type_candidates=[ConceptKind.TASK],
            ),
        ],
        concepts=[
            Concept(
                concept_id="tech1",
                kind=ConceptKind.TECHNOLOGY,
                preferred_label="NER",
            ),
            Concept(
                concept_id="task1",
                kind=ConceptKind.TASK,
                preferred_label="text classification",
            ),
        ],
        resolutions=[
            ResolutionDecision(
                resolution_id="r1",
                mention_id="m1",
                status="accepted",
                concept_id="tech1",
            ),
            ResolutionDecision(
                resolution_id="r2",
                mention_id="m2",
                status="accepted",
                concept_id="task1",
            ),
        ],
    )
    document = type(
        "Doc",
        (),
        {
            "chunks": [
                Chunk(
                    chunk_id="ch1",
                    kind="abstract",
                    text="NER is for text classification.",
                    order=0,
                )
            ]
        },
    )()

    links = GraphStore._solution_links(document, result)

    assert links == []

    result.assertions = [
        Assertion(
            assertion_id="a1",
            predicate="solves_task",
            roles={"subject": "tech1", "task": "task1"},
            evidence=[
                EvidenceSpan(
                    chunk_id="ch1",
                    quote="NER is for text classification.",
                    start=0,
                    end=31,
                )
            ],
            status="accepted",
            verification_status="supported",
            polarity="affirmed",
            modality="reported",
        )
    ]
    assert GraphStore._solution_links(document, result)[0][:2] == (
        "tech1",
        "task1",
    )
    result.assertions[0].polarity = "negated"
    assert GraphStore._solution_links(document, result) == []
    result.assertions[0].polarity = "affirmed"
    result.assertions[0].modality = "planned"
    assert GraphStore._solution_links(document, result) == []
    result.assertions[0].modality = "reported"
    result.assertions[0].verification_status = "unverified"
    assert GraphStore._solution_links(document, result) == []


def _observed_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "observed_at":
                yield item
            yield from _observed_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _observed_values(item)


def test_undated_document_evidence_is_not_dated_by_its_upload():
    uploaded = "2026-09-20T10:00:00+00:00"
    document = type(
        "Doc",
        (),
        {
            "document_version_id": "v1",
            "published_at": None,
            "version_published_at": None,
            "retrieved_at": uploaded,
            "chunks": [],
            "domains": [],
        },
    )()
    result = ExtractionResult(
        document_version_id="v1",
        run=ProcessingRun(
            run_id="run1", parser="test", config_hash="x", started_at="now"
        ),
        mentions=[
            Mention(
                mention_id="m1",
                chunk_id="ch1",
                surface_text="NLP",
                start=0,
                end=3,
                type_candidates=[ConceptKind.TECHNOLOGY],
            )
        ],
        concepts=[
            Concept(
                concept_id="c1",
                kind=ConceptKind.TECHNOLOGY,
                preferred_label="NLP",
            )
        ],
        resolutions=[
            ResolutionDecision(
                resolution_id="r1",
                mention_id="m1",
                status="accepted",
                concept_id="c1",
            )
        ],
    )
    tx = Transaction()
    asyncio.run(GraphStore._write_extraction(tx, document, result))
    observed = [
        value
        for _, parameters in tx.queries
        for value in _observed_values(parameters)
    ]
    assert observed and uploaded not in observed


def test_redownload_keeps_the_first_retrieval_for_visibility():
    document = parse_openalex(
        {"id": "https://openalex.org/W1", "title": "Example"}
    )
    tx = Transaction()
    asyncio.run(GraphStore._write_document(tx, document))
    query = " ".join(
        next(q for q, _ in tx.queries if "MERGE (v:DocumentVersion" in q)
        .split()
    )
    # A later download must not move the version out of past snapshots.
    assert "v.retrieved_at =" not in query
    first = re.search(r"v\.first_retrieved_at = CASE (.*?) END", query)
    last = re.search(r"v\.last_retrieved_at = CASE (.*?) END", query)
    assert first and "$retrieved_at <" in first.group(1)
    assert last and "$retrieved_at >" in last.group(1)


class ReadSession:
    def __init__(self, queries):
        self.queries = queries

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def run(self, query, **parameters):
        self.queries.append(query)
        if "db.labels" in query:
            return [{"label": "DocumentVersion"}]
        return []


def test_temporal_read_uses_first_retrieval_with_legacy_fallback():
    queries = []
    store = GraphStore.__new__(GraphStore)
    store._driver = type(
        "Driver", (), {"session": lambda self, **_: ReadSession(queries)}
    )()
    store._database = "neo4j"
    asyncio.run(store.read_temporal_data())
    versions = " ".join(
        next(q for q in queries if "MATCH (d:Document)" in q).split()
    )
    assert (
        "coalesce(v.first_retrieved_at, v.retrieved_at) AS retrieved_at"
        in versions
    )
    assert "AS last_retrieved_at" in versions


def test_metrics_are_a_dated_observation_of_the_unchanged_version():
    work = {"id": "https://openalex.org/W1", "title": "Example"}
    early = parse_openalex(
        {**work, "cited_by_count": 17, "_retrieved_at": "2024-01-01"}
    )
    late = parse_openalex(
        {**work, "cited_by_count": 40, "_retrieved_at": "2026-01-01"}
    )
    assert early.document_version_id == late.document_version_id
    observations = []
    for document in (early, late):
        tx = Transaction()
        asyncio.run(GraphStore._write_document(tx, document))
        observations.extend(
            parameters
            for query, parameters in tx.queries
            if "MERGE (m:MetricsObservation" in query
        )
    assert [item["observed_at"] for item in observations] == [
        "2024-01-01",
        "2026-01-01",
    ]
    assert len({item["observation_id"] for item in observations}) == 2
    assert json.loads(observations[0]["metrics_json"])["citation_count"] == 17
    # The version keeps only a newer observation as its current metrics.
    query = " ".join(
        next(q for q, _ in tx.queries if "MERGE (v:DocumentVersion" in q)
        .split()
    )
    current = re.search(r"v\.metrics_json = CASE (.*?) END", query)
    assert current and ">= v.metrics_observed_at" in current.group(1)


def test_temporal_read_returns_the_metric_history():
    queries = []
    store = GraphStore.__new__(GraphStore)
    store._driver = type(
        "Driver", (), {"session": lambda self, **_: ReadSession(queries)}
    )()
    store._database = "neo4j"
    asyncio.run(store.read_temporal_data())
    versions = next(q for q in queries if "MATCH (d:Document)" in q)
    assert "HAS_METRICS" in versions
    assert "metric_observations" in versions.split("RETURN", 1)[1]


def _work_with_parties(size):
    return parse_openalex(
        {
            "id": "https://openalex.org/W1",
            "title": "Crowded paper",
            "publication_date": "2024-01-01",
            "abstract_inverted_index": {"Sensor": [0], "study": [1]},
            "authorships": [
                {
                    "author": {
                        "id": f"https://openalex.org/A{index}",
                        "display_name": f"Author {index}",
                    },
                    "institutions": [
                        {
                            "id": f"https://openalex.org/I{index}",
                            "display_name": f"University {index}"
                            if index % 2
                            else f"Company {index} Inc",
                            "type": "education" if index % 2 else "company",
                            "country_code": ["DE", "FR", "US"][index % 3],
                        }
                    ],
                }
                for index in range(size)
            ],
            "topics": [
                {
                    "display_name": f"Topic {index}",
                    "subfield": {
                        "id": f"https://openalex.org/subfields/{index}",
                        "display_name": f"Subfield {index}",
                    },
                    "field": {"display_name": "Engineering"},
                }
                for index in range(size)
            ],
        }
    )


def test_document_write_cost_does_not_grow_with_its_parties():
    # D-1: every statement is a network round trip.
    counts = []
    for size in (3, 30):
        document = _work_with_parties(size)
        assert len(document.contributors) == size
        tx = Transaction()
        asyncio.run(GraphStore._write_document(tx, document))
        counts.append(len(tx.queries))
    assert counts[0] == counts[1]
    # 15 for the document, 2 for its work (graph.works).
    assert counts[1] <= 17


def test_extraction_write_cost_does_not_grow_with_its_concepts():
    document = parse_openalex(
        {"id": "https://openalex.org/W1", "title": "Example"}
    )

    def result(size):
        return ExtractionResult(
            document_version_id=document.document_version_id,
            run=ProcessingRun(
                run_id="run1", parser="test", config_hash="x", started_at="t"
            ),
            concepts=[
                Concept(
                    concept_id=f"c{index}",
                    kind=ConceptKind.TECHNOLOGY
                    if index % 2
                    else ConceptKind.TASK,
                    preferred_label=f"Concept {index}",
                )
                for index in range(size)
            ],
        )

    counts = []
    for size in (2, 40):
        tx = Transaction()
        asyncio.run(GraphStore._write_extraction(tx, document, result(size)))
        counts.append(len(tx.queries))
    assert counts[0] == counts[1]


class SchemaSession:
    def __init__(self, statements):
        self.statements = statements

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def run(self, query, **parameters):
        self.statements.append(query)
        return Result()


def test_schema_is_ensured_once_per_process_and_graph(monkeypatch):
    from lctrend.graph import store as store_module

    monkeypatch.setattr(store_module, "_SCHEMA_READY", set())
    statements = []

    def open_store(uri):
        store = GraphStore.__new__(GraphStore)
        store._driver = type(
            "Driver",
            (),
            {"session": lambda self, **_: SchemaSession(statements)},
        )()
        store._database = "neo4j"
        store._schema_key = (uri, "neo4j")
        return store

    asyncio.run(open_store("bolt://a").ensure_schema())
    once = len(statements)
    assert once > 10
    asyncio.run(open_store("bolt://a").ensure_schema())
    assert len(statements) == once
    asyncio.run(open_store("bolt://b").ensure_schema())
    assert len(statements) == 2 * once


def test_published_run_clears_evidence_of_chunks_leaving_the_version():
    # D-2: HAS_CHUNK was cut before extraction cleanup, which only sees the
    # chunks still linked, so evidence of dropped chunks outlived its run.
    document = parse_openalex(
        {
            "id": "https://openalex.org/W1",
            "title": "Example",
            "abstract_inverted_index": {"Sensor": [0], "study": [1]},
        }
    )
    tx = Transaction()
    asyncio.run(GraphStore._write_document(tx, document, publishing=True))
    statements = [" ".join(query.split()) for query, _ in tx.queries]
    stale = [
        index
        for index, query in enumerate(statements)
        if "WHERE NOT c.chunk_id IN $chunk_ids" in query
    ]
    kinds = [statements[index] for index in stale]
    assert "MATCH (c)-[r:MENTIONS]->() DELETE r" in kinds[0]
    assert "HAS_MATURITY_EVIDENCE" in kinds[1]
    assert "DELETE active" in kinds[2]
    assert tx.queries[stale[2]][1]["publishing"] is True


def test_import_without_extraction_keeps_chunks_of_a_published_run():
    # D-2: a re-import without the PDF unlinked the PDF chunks and hid the
    # published run's mentions (50 visible mentions became 2).
    document = parse_openalex({"id": "https://openalex.org/W1", "title": "X"})
    tx = Transaction()
    asyncio.run(GraphStore._write_document(tx, document))
    statements = [" ".join(query.split()) for query, _ in tx.queries]
    assert not any("[r:MENTIONS]" in query for query in statements)
    unlink = next(
        (query, parameters)
        for query, parameters in zip(statements, (p for _, p in tx.queries))
        if "DELETE active" in query
    )
    assert "WHERE $publishing OR NOT EXISTS" in unlink[0]
    assert "run.published = true OR run.status = 'succeeded'" in unlink[0]
    assert unlink[1]["publishing"] is False


def test_registry_read_does_not_transfer_embeddings():
    # D-6: properties(c) pulled every stored vector at each job start.
    queries = []

    class Session(ReadSession):
        async def run(self, query, **parameters):
            queries.append(query)
            return [
                {
                    "properties": {
                        "concept_id": "c1",
                        "kind": "Technology",
                        "preferred_label": "Sparse attention",
                    }
                }
            ]

    store = GraphStore.__new__(GraphStore)
    store._driver = type(
        "Driver", (), {"session": lambda self, **_: Session(queries)}
    )()
    store._database = "neo4j"
    concepts = asyncio.run(store.read_concepts())
    assert concepts and concepts[0].preferred_label == "Sparse attention"
    assert "properties(c)" not in queries[0]
    assert "embedding" not in queries[0]
    assert ".names_json" in queries[0]
    assert ".identity_key" in queries[0]
    assert ".label_counts_json" in queries[0]
