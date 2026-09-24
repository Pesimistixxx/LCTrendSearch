from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .models import DocumentEnvelope, ExtractionResult, json_value, validate_extraction


CONSTRAINTS = [
    "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.source_id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.document_id IS UNIQUE",
    "CREATE CONSTRAINT document_version_id IF NOT EXISTS FOR (n:DocumentVersion) REQUIRE n.document_version_id IS UNIQUE",
    "CREATE CONSTRAINT external_id IF NOT EXISTS FOR (n:ExternalId) REQUIRE n.external_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT contributor_id IF NOT EXISTS FOR (n:Contributor) REQUIRE n.contributor_id IS UNIQUE",
    "CREATE CONSTRAINT mention_id IF NOT EXISTS FOR (n:Mention) REQUIRE n.mention_id IS UNIQUE",
    "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (n:Concept) REQUIRE n.concept_id IS UNIQUE",
    "CREATE CONSTRAINT concept_name_id IF NOT EXISTS FOR (n:ConceptName) REQUIRE n.name_id IS UNIQUE",
    "CREATE CONSTRAINT assertion_id IF NOT EXISTS FOR (n:Assertion) REQUIRE n.assertion_id IS UNIQUE",
    "CREATE CONSTRAINT resolution_id IF NOT EXISTS FOR (n:ResolutionDecision) REQUIRE n.resolution_id IS UNIQUE",
    "CREATE CONSTRAINT run_id IF NOT EXISTS FOR (n:ProcessingRun) REQUIRE n.run_id IS UNIQUE",
    "CREATE CONSTRAINT claim_group_id IF NOT EXISTS FOR (n:ClaimGroup) REQUIRE n.claim_group_id IS UNIQUE",
    "CREATE CONSTRAINT evidence_family_id IF NOT EXISTS FOR (n:EvidenceFamily) REQUIRE n.family_id IS UNIQUE",
]

ROLE_RELATIONSHIPS = {
    "subject": "SUBJECT",
    "problem": "PROBLEM",
    "task": "TASK",
    "method": "METHOD",
    "baseline": "BASELINE",
    "application": "APPLICATION",
    "metric": "METRIC",
    "material": "MATERIAL",
}


class GraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Install the project first: pip install -e .") from exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        with self._driver.session(database=self._database) as session:
            for query in CONSTRAINTS:
                session.run(query).consume()

    def write_document(self, document: DocumentEnvelope) -> None:
        with self._driver.session(database=self._database) as session:
            session.execute_write(self._write_document, document)

    @staticmethod
    def _write_document(tx: Any, document: DocumentEnvelope) -> None:
        tx.run(
            """
            MERGE (s:Source {source_id: $source.source_id})
            SET s.name = $source.name, s.source_type = $source.source_type
            MERGE (d:Document {document_id: $document_id})
            SET d.document_type = $document_type, d.title = $title,
                d.language = $language, d.published_at = $published_at,
                d.canonical_url = $source.canonical_url
            MERGE (v:DocumentVersion {document_version_id: $version_id})
            SET v.raw_sha256 = $artifact.sha256, v.raw_uri = $artifact.uri,
                v.media_type = $artifact.media_type, v.byte_length = $artifact.byte_length,
                v.access_status = $artifact.access_status, v.coverage = $coverage,
                v.quality_status = $quality_status, v.metadata_json = $metadata_json
            MERGE (d)-[:HAS_VERSION]->(v)
            MERGE (v)-[:FROM_SOURCE {record_id: $source.record_id}]->(s)
            """,
            source=document.source.model_dump(),
            document_id=document.document_id,
            document_type=document.document_type.value,
            title=document.title,
            language=document.language,
            published_at=document.published_at,
            version_id=document.document_version_id,
            artifact=document.artifact.model_dump(),
            coverage=document.coverage,
            quality_status=document.quality_status,
            metadata_json=json_value(document.metadata),
        ).consume()

        for identifier in document.identifiers:
            tx.run(
                """
                MATCH (d:Document {document_id: $document_id})
                MERGE (i:ExternalId {external_id: $external_id})
                SET i.scheme = $scheme, i.value = $value
                MERGE (d)-[:IDENTIFIED_BY]->(i)
                """,
                document_id=document.document_id,
                external_id=identifier.external_id,
                scheme=identifier.scheme,
                value=identifier.value,
            ).consume()

        for contributor in document.contributors:
            label = "Organization" if contributor.kind == "organization" else "Person"
            query = """
                MATCH (d:Document {document_id: $document_id})
                MERGE (c:Contributor:__LABEL__ {contributor_id: $contributor_id})
                SET c.name = $name, c.kind = $kind
                MERGE (d)-[:CONTRIBUTED_BY {role: $role}]->(c)
                """.replace("__LABEL__", label)
            tx.run(
                query,
                document_id=document.document_id,
                **contributor.model_dump(exclude={"external_ids"}),
            ).consume()
            for identifier in contributor.external_ids:
                tx.run(
                    """
                    MATCH (c:Contributor {contributor_id: $contributor_id})
                    MERGE (i:ExternalId {external_id: $external_id})
                    SET i.scheme = $scheme, i.value = $value
                    MERGE (c)-[:IDENTIFIED_BY]->(i)
                    """,
                    contributor_id=contributor.contributor_id,
                    external_id=identifier.external_id,
                    scheme=identifier.scheme,
                    value=identifier.value,
                ).consume()

        for chunk in document.chunks:
            tx.run(
                """
                MATCH (v:DocumentVersion {document_version_id: $version_id})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.kind = $kind, c.text = $text, c.order = $order,
                    c.section_path = $section_path, c.locator_json = $locator_json,
                    c.content_hash = $content_hash, c.parse_status = $parse_status
                MERGE (v)-[:HAS_CHUNK]->(c)
                """,
                version_id=document.document_version_id,
                chunk_id=chunk.chunk_id,
                kind=chunk.kind,
                text=chunk.text,
                order=chunk.order,
                section_path=chunk.section_path,
                locator_json=json_value(chunk.locator),
                content_hash=chunk.content_hash,
                parse_status=chunk.parse_status,
            ).consume()

    def write_extraction(self, document: DocumentEnvelope, result: ExtractionResult) -> None:
        validate_extraction(document, result)
        with self._driver.session(database=self._database) as session:
            session.execute_write(self._write_extraction, document, result)

    @staticmethod
    def _write_extraction(tx: Any, document: DocumentEnvelope, result: ExtractionResult) -> None:
        run = result.run
        tx.run(
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
            MERGE (r:ProcessingRun {run_id: $run_id})
            SET r.pipeline_version = $pipeline_version, r.parser = $parser,
                r.model_revision = $model_revision, r.prompt_hash = $prompt_hash,
                r.config_hash = $config_hash, r.started_at = $started_at, r.status = $status
            MERGE (r)-[:PROCESSED]->(v)
            """,
            version_id=document.document_version_id,
            **run.model_dump(),
        ).consume()

        for concept in result.concepts:
            tx.run(
                """
                MERGE (c:Concept {concept_id: $concept_id})
                SET c.kind = $kind, c.preferred_label = $preferred_label,
                    c.definition = $definition, c.language = $language, c.status = $status
                """,
                **concept.model_dump(exclude={"names"}, mode="json"),
            ).consume()
            for name in concept.names:
                tx.run(
                    """
                    MATCH (c:Concept {concept_id: $concept_id})
                    MERGE (n:ConceptName {name_id: $name_id})
                    SET n.text = $text, n.normalized_text = $normalized_text,
                        n.language = $language, n.name_kind = $name_kind, n.status = $status
                    MERGE (c)-[:HAS_NAME]->(n)
                    """,
                    concept_id=concept.concept_id,
                    **name.model_dump(),
                ).consume()

        for mention in result.mentions:
            tx.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MATCH (r:ProcessingRun {run_id: $run_id})
                MERGE (m:Mention {mention_id: $mention_id})
                SET m.surface_text = $surface_text, m.start = $start, m.end = $end,
                    m.type_candidates = $type_candidates, m.mention_role = $mention_role,
                    m.discourse_role = $discourse_role, m.confidence = $confidence, m.status = $status
                MERGE (c)-[:HAS_MENTION]->(m)
                MERGE (r)-[:CREATED]->(m)
                """,
                run_id=run.run_id,
                **mention.model_dump(mode="json"),
            ).consume()

        for decision in result.resolutions:
            tx.run(
                """
                MATCH (m:Mention {mention_id: $mention_id})
                MERGE (r:ResolutionDecision {resolution_id: $resolution_id})
                SET r.status = $status, r.method = $method, r.score = $score,
                    r.basis = $basis, r.candidates_json = $candidates_json,
                    r.resolver_version = $resolver_version,
                    r.taxonomy_version = $taxonomy_version, r.review_status = $review_status
                MERGE (m)-[:HAS_RESOLUTION]->(r)
                """,
                candidates_json=json_value(decision.candidates),
                **decision.model_dump(exclude={"candidates", "concept_id"}),
            ).consume()
            if decision.concept_id:
                tx.run(
                    """
                    MATCH (r:ResolutionDecision {resolution_id: $resolution_id})
                    MATCH (c:Concept {concept_id: $concept_id})
                    MERGE (r)-[:RESOLVED_AS]->(c)
                    """,
                    resolution_id=decision.resolution_id,
                    concept_id=decision.concept_id,
                ).consume()

        for assertion in result.assertions:
            tx.run(
                """
                MATCH (v:DocumentVersion {document_version_id: $version_id})
                MATCH (r:ProcessingRun {run_id: $run_id})
                MERGE (a:Assertion {assertion_id: $assertion_id})
                SET a.predicate = $predicate, a.qualifiers_json = $qualifiers_json,
                    a.values_json = $values_json, a.polarity = $polarity,
                    a.modality = $modality, a.attribution_kind = $attribution_kind,
                    a.evidence_kind = $evidence_kind,
                    a.extraction_confidence = $extraction_confidence,
                    a.verification_status = $verification_status, a.status = $status
                MERGE (v)-[:HAS_ASSERTION]->(a)
                MERGE (r)-[:CREATED]->(a)
                """,
                version_id=document.document_version_id,
                run_id=run.run_id,
                qualifiers_json=json_value(assertion.qualifiers),
                values_json=json_value(assertion.values),
                **assertion.model_dump(
                    exclude={"roles", "evidence", "qualifiers", "values", "claim_group_id", "evidence_family_id"}
                ),
            ).consume()

            for role, concept_id in assertion.roles.items():
                relation = ROLE_RELATIONSHIPS.get(role)
                if relation is None:
                    raise ValueError(f"unsupported assertion role: {role}")
                tx.run(
                    f"""
                    MATCH (a:Assertion {{assertion_id: $assertion_id}})
                    MATCH (c:Concept {{concept_id: $concept_id}})
                    MERGE (a)-[:{relation}]->(c)
                    """,
                    assertion_id=assertion.assertion_id,
                    concept_id=concept_id,
                ).consume()

            for evidence in assertion.evidence:
                tx.run(
                    """
                    MATCH (a:Assertion {assertion_id: $assertion_id})
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (a)-[e:SUPPORTED_BY {start: $start, end: $end}]->(c)
                    SET e.quote = $quote, e.supports_fields = $supports_fields
                    """,
                    assertion_id=assertion.assertion_id,
                    **evidence.model_dump(),
                ).consume()

            if assertion.claim_group_id:
                tx.run(
                    """
                    MATCH (a:Assertion {assertion_id: $assertion_id})
                    MERGE (g:ClaimGroup {claim_group_id: $group_id})
                    MERGE (a)-[:IN_CLAIM_GROUP]->(g)
                    """,
                    assertion_id=assertion.assertion_id,
                    group_id=assertion.claim_group_id,
                ).consume()
            if assertion.evidence_family_id:
                tx.run(
                    """
                    MATCH (a:Assertion {assertion_id: $assertion_id})
                    MERGE (f:EvidenceFamily {family_id: $family_id})
                    MERGE (a)-[:FROM_EVIDENCE_FAMILY]->(f)
                    """,
                    assertion_id=assertion.assertion_id,
                    family_id=assertion.evidence_family_id,
                ).consume()
