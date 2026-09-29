import asyncio

from lctrend.core.models import (
    Assertion,
    Concept,
    ConceptKind,
    Domain,
    EconomicEvidence,
    EvidenceSpan,
    ExtractionResult,
    ProcessingRun,
)
from lctrend.graph.store import GraphStore
from lctrend.ingest.adapters import parse_openalex


class Result:
    def __init__(self, rows=()):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def consume(self):
        pass


class Session:
    def __init__(self, rows=()):
        self.rows, self.queries = rows, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def run(self, query, **parameters):
        self.queries.append((query, parameters))
        return Result(self.rows)


def store_with_rows(rows):
    session = Session(rows)

    class Driver:
        def session(self, **kwargs):
            return session

    store = object.__new__(GraphStore)
    store._driver, store._database = Driver(), "offline"
    return store, session


def extraction(document, concepts=(), assertions=()):
    return ExtractionResult(
        document_version_id=document.document_version_id,
        run=ProcessingRun(
            run_id="run",
            parser="llm_packets",
            config_hash="c",
            started_at="2026-01-01",
        ),
        concepts=list(concepts),
        assertions=list(assertions),
    )


def test_domain_bridge_requires_exact_unique_canonical_metadata_identity():
    document = parse_openalex(
        {"id": "https://openalex.org/W123", "title": "A"}
    )
    document.domains = [Domain(domain_id="domain:1", name="Computer Science")]
    concept = Concept(
        concept_id="text-domain",
        kind=ConceptKind.DOMAIN,
        preferred_label="  COMPUTER   SCIENCE ",
        status="accepted",
    )
    result = extraction(document, [concept])
    links = GraphStore._domain_identity_links(document, result)
    assert links == [
        {
            "concept_id": "text-domain",
            "domain_id": "domain:1",
            "assertion_ids": [],
        }
    ]
    concept.preferred_label = "Computing"
    assert GraphStore._domain_identity_links(document, result) == []
    concept.preferred_label = "Computer Science"
    document.domains.append(
        Domain(domain_id="domain:2", name="computer science")
    )
    assert GraphStore._domain_identity_links(document, result) == []


def test_domain_bridge_does_not_equate_symbols_or_promote_unreviewed_domain():
    document = parse_openalex(
        {"id": "https://openalex.org/W123", "title": "A"}
    )
    document.domains = [Domain(domain_id="domain:1", name="C++")]
    concept = Concept(
        concept_id="text-domain",
        kind=ConceptKind.DOMAIN,
        preferred_label="C",
        status="accepted",
    )
    result = extraction(document, [concept])
    assert GraphStore._domain_identity_links(document, result) == []
    concept.preferred_label = "C++"
    concept.status = "provisional"
    assert GraphStore._domain_identity_links(document, result) == []


def test_reviewed_domain_role_is_bridged_with_source_assertion_attribution():
    document = parse_openalex(
        {"id": "https://openalex.org/W123", "title": "A"}
    )
    document.domains = [Domain(domain_id="domain:1", name="Computer Science")]
    concept = Concept(
        concept_id="text-domain",
        kind=ConceptKind.DOMAIN,
        preferred_label="Computer Science",
    )
    claim = Assertion(
        assertion_id="claim",
        predicate="belongs_to_domain",
        roles={"subject": "tech", "domain": "text-domain"},
        evidence=[EvidenceSpan(chunk_id="c", quote="A", start=0, end=1)],
        status="accepted",
        verification_status="supported",
    )
    result = extraction(document, [concept], [claim])
    assert GraphStore._domain_identity_links(document, result)[0][
        "assertion_ids"
    ] == ["claim"]


def test_curated_russian_alias_bridges_reviewed_existing_metadata_domain():
    document = parse_openalex(
        {"id": "https://openalex.org/W123", "title": "A"}
    )
    document.domains = [
        Domain(domain_id="domain:vision", name="Computer vision")
    ]
    concept = Concept(
        concept_id="text-domain",
        kind=ConceptKind.DOMAIN,
        preferred_label="  КОМПЬЮТЕРНОЕ   ЗРЕНИЕ ",
    )
    claim = Assertion(
        assertion_id="claim",
        predicate="belongs_to_domain",
        roles={"subject": "tech", "domain": concept.concept_id},
        evidence=[EvidenceSpan(chunk_id="c", quote="A", start=0, end=1)],
        status="accepted",
        verification_status="supported",
    )
    result = extraction(document, [concept], [claim])
    assert GraphStore._domain_identity_links(document, result) == [
        {
            "concept_id": "text-domain",
            "domain_id": "domain:vision",
            "assertion_ids": ["claim"],
            "method": "exact_curated_alias",
        }
    ]
    result.assertions = []
    assert GraphStore._domain_identity_links(document, result) == []
    concept.status = "accepted"
    document.domains = [Domain(domain_id="domain:other", name="Other")]
    assert GraphStore._domain_identity_links(document, result) == []


def test_curated_alias_is_ambiguous_across_catalog_even_with_one_doc_domain(
    monkeypatch,
):
    document = parse_openalex(
        {"id": "https://openalex.org/W123", "title": "A"}
    )
    document.domains = [
        Domain(domain_id="domain:vision", name="Computer vision")
    ]
    concept = Concept(
        concept_id="text-domain",
        kind=ConceptKind.DOMAIN,
        preferred_label="компьютерное зрение",
        status="accepted",
    )
    monkeypatch.setattr(
        "lctrend.graph.store.load_catalog",
        lambda name: {
            "domains": [
                {
                    "name": "Computer vision",
                    "aliases": ["компьютерное зрение"],
                },
                {
                    "name": "Machine learning",
                    "aliases": ["компьютерное зрение"],
                },
            ]
        },
    )
    assert (
        GraphStore._domain_identity_links(
            document, extraction(document, [concept])
        )
        == []
    )


def test_related_context_preserves_originals_provenance_and_all_budgets():
    rows = [
        {
            "chunk_id": "oversized",
            "text": "x" * 100,
            "document_id": "d0",
            "document_version_id": "v0",
        },
        {
            "chunk_id": "primary",
            "text": "Original",
            "document_id": "d1",
            "document_version_id": "current",
        },
        {
            "chunk_id": "c1",
            "text": "Original one",
            "kind": "abstract",
            "title": "Article",
            "document_id": "d2",
            "document_version_id": "v2",
            "locator_json": '{"page":2}',
        },
        {
            "chunk_id": "c1",
            "text": "Original one",
            "document_id": "d2",
            "document_version_id": "v2",
        },
        {
            "chunk_id": "c2",
            "text": "Original two",
            "document_id": "d3",
            "document_version_id": "v3",
        },
    ]
    store, session = store_with_rows(rows)
    result = asyncio.run(
        store.read_related_chunks(" Original ", "current", 1, 3, 30)
    )
    assert len(result) == 1
    assert result[0]["text"] == "Original one"
    assert result[0]["document_version_id"] == "v2"
    assert result[0]["locator"] == {"page": 2}
    query, parameters = session.queries[0]
    assert "run.status = 'succeeded'" in query
    assert "run.published = true" in query
    assert "CONTAINS $search_text" in query
    assert parameters["search_text"] == "original"
    assert parameters["exclude_version_id"] == "current"
    result = asyncio.run(
        store.read_related_chunks("Original", "current", 3, 1, 30)
    )
    assert len(result) == 1
    assert (
        asyncio.run(store.read_related_chunks("", "current", 3, 1, 30)) == []
    )


def test_signal_read_filters_candidates_forecasts_and_reads_new_relations():
    store, session = store_with_rows([])
    asyncio.run(store.read_signal_data())
    query = session.queries[0][0]
    assert "r.status = 'accepted'" in query
    assert "r.modality IN ['reported', 'observed']" in query
    for relation in (
        "MANUFACTURED_IN",
        "TESTED_IN",
        "DEPLOYED_IN",
        "BELONGS_TO_DOMAIN",
        "TARGETS_MARKET",
    ):
        assert relation in query


def test_taxonomy_reads_dated_identities_and_filters_hierarchy_by_snapshot():
    store, session = store_with_rows([])
    asyncio.run(store.read_taxonomy_input(["Technology"], "2024-01-01"))
    concepts, parents = session.queries
    assert "document_id: d.document_id" in concepts[0]
    assert "document_evidence" in concepts[0]
    assert "embedding_model" in concepts[0]
    assert "embedding_dimensions" in concepts[0]
    assert "r.observed_at IS NOT NULL" in parents[0]
    assert parents[1]["snapshot"] == "2024-01-01"


def test_financial_graph_evidence_keeps_review_and_source_provenance():
    document = parse_openalex(
        {
            "id": "https://openalex.org/W123",
            "title": "USD 2 million cost",
            "abstract_inverted_index": {
                "USD": [0],
                "2": [1],
                "million": [2],
                "cost": [3],
            },
        }
    )
    document.version_published_at = "2026-01-01"
    chunk = document.chunks[0]
    concept = Concept(
        concept_id="tech",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="Technology A",
    )
    result = extraction(document, [concept])
    result.economic_evidence = [
        EconomicEvidence(
            evidence_id="economic",
            technology_concept_id="tech",
            chunk_id=chunk.chunk_id,
            category="cost",
            quote=chunk.text,
            start=0,
            end=len(chunk.text),
            amount_text="USD 2 million",
            amount_value=2000000,
            currency="USD",
            unit="per year",
            period="2026",
            assertion_id="reviewed-claim",
            status="accepted",
        )
    ]
    tx = Session()
    asyncio.run(GraphStore._write_extraction(tx, document, result))
    query, parameters = next(
        (query, parameters)
        for query, parameters in tx.queries
        if "MERGE (technology)-[r:HAS_ECONOMIC_EVIDENCE" in query
    )
    # Batched (D-1): per-evidence values are UNWIND rows.
    (row,) = parameters["rows"]
    assert row["assertion_id"] == "reviewed-claim"
    assert row["unit"] == "per year"
    assert row["period"] == "2026"
    assert row["amount_text"] == "USD 2 million"
    assert row["observed_at"] == "2026-01-01"
    assert parameters["recorded_at"] == result.run.started_at
    assert "r.assertion_id = row.assertion_id" in query


def test_related_chunk_search_uses_full_text_indexes_with_literal_phrase():
    # D-7: toLower(c.text) CONTAINS scanned every chunk of the graph.
    from lctrend.core.config import resource_path

    store, session = store_with_rows([])
    asyncio.run(
        store.read_related_chunks('Say "hi" \\ OR x*', "current", 1, 3, 30)
    )
    query, parameters = session.queries[0]
    assert "db.index.fulltext.queryNodes('chunk_text', $phrase)" in query
    assert "db.index.fulltext.queryNodes('document_title', $phrase)" in query
    assert "CONTAINS $search_text" in query
    # Model text is one quoted phrase, never Lucene operators.
    assert parameters["phrase"] == '"say \\"hi\\" \\\\ or x*"'
    schema = resource_path("schema", ".cypher").read_text(encoding="utf-8")
    assert "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS" in schema
    assert "CREATE FULLTEXT INDEX document_title IF NOT EXISTS" in schema


def test_related_context_adds_evidence_vector_search_when_given_a_vector():
    rows = [
        {
            "chunk_id": "ru",
            "text": "Большие языковые модели",
            "document_id": "d1",
            "document_version_id": "v1",
            "score": 0.81234,
        }
    ]
    store, session = store_with_rows(rows)
    result = asyncio.run(
        store.read_related_chunks(
            "LLM", "current", 3, 3, 1000, query_vector=[1.0, 0.0]
        )
    )
    assert result[0]["similarity"] == 0.8123
    query, parameters = session.queries[0]
    assert "db.index.vector.queryNodes" in query
    assert "'chunk_embedding'" in query
    assert parameters["vector"] == [1.0, 0.0]
    assert parameters["min_score"] == 0.75
    # Without a vector the query stays literal.
    asyncio.run(store.read_related_chunks("LLM", "current", 3, 3, 1000))
    assert "vector.queryNodes" not in session.queries[-1][0]


def test_related_context_falls_back_to_literal_search_without_index():
    class FailingVectorSession(Session):
        def run(self, query, **parameters):
            if "vector.queryNodes" in query:
                raise RuntimeError("no such index")
            return super().run(query, **parameters)

    session = FailingVectorSession(
        [
            {
                "chunk_id": "c",
                "text": "LLM",
                "document_id": "d",
                "document_version_id": "v",
            }
        ]
    )

    class Driver:
        def session(self, **kwargs):
            return session

    store = object.__new__(GraphStore)
    store._driver, store._database = Driver(), "offline"
    result = asyncio.run(
        store.read_related_chunks(
            "LLM", "current", 3, 3, 1000, query_vector=[1.0]
        )
    )
    assert [row["chunk_id"] for row in result] == ["c"]
    assert "similarity" not in result[0]
