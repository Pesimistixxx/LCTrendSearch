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
    assert "c.name = $preferred_label" in queries
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
