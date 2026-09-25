from lctrend.adapters import parse_openalex
from lctrend.graph import GraphStore
from lctrend.models import (
    Chunk,
    Concept,
    ConceptKind,
    EconomicEvidence,
    ExtractionResult,
    Mention,
    ProcessingRun,
    ResolutionDecision,
)


class Result:
    def consume(self):
        return None


class Transaction:
    def __init__(self):
        self.queries = []

    def run(self, query, **parameters):
        self.queries.append((query, parameters))
        return Result()


def test_document_projection_builds_queries_without_dynamic_cypher_errors():
    document = parse_openalex(
        {
            "id": "https://openalex.org/W1",
            "title": "Example",
            "authorships": [
                {
                    "author": {"id": "https://openalex.org/A1", "display_name": "Ada"},
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
    GraphStore._write_document(tx, document)
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
    document_query = next(parameters for query, parameters in tx.queries if "MERGE (d:Document" in query)
    assert document_query["external_ids"] == ["openalex:w1"]


def test_extraction_stores_aliases_and_resolution_on_relationships():
    tx = Transaction()
    result = ExtractionResult(
        document_version_id="v1",
        run=ProcessingRun(run_id="run1", parser="test", config_hash="x", started_at="now"),
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
                concept_id="c1", kind=ConceptKind.TECHNOLOGY, preferred_label="NLP"
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
    GraphStore._write_extraction(
        tx,
        type(
            "Doc", (), {"document_version_id": "v1", "published_at": None, "chunks": []}
        )(),
        result,
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


def test_solution_link_requires_a_technology_task_sentence_with_a_solution_cue():
    result = ExtractionResult(
        document_version_id="v1",
        run=ProcessingRun(run_id="run1", parser="test", config_hash="x", started_at="now"),
        mentions=[
            Mention(
                mention_id="m1", chunk_id="ch1", surface_text="NER", start=0, end=3,
                type_candidates=[ConceptKind.TECHNOLOGY],
            ),
            Mention(
                mention_id="m2", chunk_id="ch1", surface_text="text classification",
                start=11, end=30, type_candidates=[ConceptKind.TASK],
            ),
        ],
        concepts=[
            Concept(concept_id="tech1", kind=ConceptKind.TECHNOLOGY, preferred_label="NER"),
            Concept(concept_id="task1", kind=ConceptKind.TASK, preferred_label="text classification"),
        ],
        resolutions=[
            ResolutionDecision(resolution_id="r1", mention_id="m1", status="accepted", concept_id="tech1"),
            ResolutionDecision(resolution_id="r2", mention_id="m2", status="accepted", concept_id="task1"),
        ],
    )
    document = type(
        "Doc", (), {"chunks": [Chunk(chunk_id="ch1", kind="abstract", text="NER is for text classification.", order=0)]}
    )()

    links = GraphStore._solution_links(document, result)

    assert links[0][:2] == ("tech1", "task1")
