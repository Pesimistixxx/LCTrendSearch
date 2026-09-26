from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from ..core.config import cypher_identifier, load_catalog, resource_path
from ..core.models import (
    Concept,
    ConceptKind,
    ConceptName,
    DocumentEnvelope,
    ExtractionResult,
    json_value,
    stable_id,
    validate_extraction,
)

logger = logging.getLogger(__name__)


def _version_date(document: DocumentEnvelope) -> Any:
    return getattr(document, "version_published_at", None) or getattr(
        document, "retrieved_at", None
    )


def _chunk_date(document: DocumentEnvelope, chunk_id: str) -> Any:
    chunk = next(
        (item for item in document.chunks if item.chunk_id == chunk_id), None
    )
    return (
        (chunk.locator.get("observed_at") or chunk.locator.get("published_at"))
        if chunk
        else None
    ) or _version_date(document)


def _concept_from_properties(properties: Dict[str, Any]) -> Concept:
    if properties.get("names_json"):
        names = [
            ConceptName.model_validate(value)
            for value in json.loads(properties["names_json"])
        ]
    else:
        names = [
            ConceptName(
                name_id=stable_id("name", properties["concept_id"], alias),
                text=alias,
                normalized_text=alias.casefold(),
                status="accepted"
                if alias == properties["preferred_label"]
                else "provisional",
            )
            for alias in dict.fromkeys(
                properties.get("aliases") or [properties["preferred_label"]]
            )
        ]
    return Concept(
        concept_id=properties["concept_id"],
        kind=ConceptKind(properties["kind"]),
        preferred_label=properties["preferred_label"],
        definition=properties.get("definition"),
        language=properties.get("language"),
        status=properties.get("status", "provisional"),
        names=names,
    )


class GraphStore:
    def __init__(
        self, uri: str, user: str, password: str, database: str = "neo4j"
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError(
                "Install the project first: pip install -e ."
            ) from exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        logger.debug("Neo4j driver for %s, database %s", uri, database)

    def close(self) -> None:
        self._driver.close()

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        with self._driver.session(database=self._database) as session:
            for query in (
                resource_path("schema", ".cypher")
                .read_text(encoding="utf-8")
                .split(";")
            ):
                if not query.strip():
                    continue
                session.run(query).consume()
        logger.debug("Neo4j schema ensured")

    def processed_materials(self):
        """Stream prior extractions into the crawler's deduplication ledger.

        A document-only import is not a completed extraction. Article,
        repository and package identities remain separate.
        """
        from ..ingest.discovery import material_identity

        query = """
            MATCH (run:ProcessingRun)-[:PROCESSED]->(v:DocumentVersion)
                  <-[:HAS_VERSION]-(d:Document)
            MATCH (v)-[origin:FROM_SOURCE]->(s:Source)
            WHERE run.status IN ['succeeded', 'partial']
              AND run.parser <> 'metadata'
              AND s.source_id IN
                  ['source:openalex', 'source:github', 'source:pypi']
            RETURN s.source_id AS source, origin.record_id AS source_id,
                   d.title AS title, d.canonical_url AS url,
                   d.external_ids AS external_ids,
                   collect(DISTINCT run.status) AS statuses
        """
        with self._driver.session(database=self._database) as session:
            labels = {
                row["label"]
                for row in session.run(
                    "CALL db.labels() YIELD label RETURN label"
                )
            }
            if not {"ProcessingRun", "Source", "Document"} <= labels:
                return
            for record in session.run(query):
                row = record.data()
                source = row["source"].removeprefix("source:")
                source_id = row["source_id"]
                payload = {}
                if source == "openalex":
                    doi = next(
                        (
                            value[4:]
                            for value in row.get("external_ids") or []
                            if value.startswith("doi:")
                        ),
                        None,
                    )
                    if doi:
                        payload["doi"] = doi
                identity_input = (
                    row["url"] if source == "github" else source_id
                )
                try:
                    canonical = material_identity(
                        source, identity_input, payload
                    )
                except (ValueError, TypeError):
                    logger.debug(
                        "Cannot import material identity: %s/%s",
                        source,
                        source_id,
                    )
                    continue
                yield {
                    "source": source,
                    "source_id": (
                        canonical.removeprefix("github:")
                        if source == "github"
                        else source_id
                    ),
                    "canonical_id": canonical,
                    "title": row["title"] or str(source_id),
                    "url": row["url"],
                    "status": (
                        "parsed"
                        if "succeeded" in row["statuses"]
                        else "partial"
                    ),
                }

    def write_document(self, document: DocumentEnvelope) -> None:
        with self._driver.session(database=self._database) as session:
            session.execute_write(self._write_document, document)
        logger.debug("Document %s written", document.document_version_id)

    @staticmethod
    def _write_document(tx: Any, document: DocumentEnvelope) -> None:
        tx.run(
            """
            MERGE (s:Source {source_id: $source.source_id})
            SET s.name = $source.name, s.source_type = $source.source_type,
                s.source_family = $source.source_family,
                s.independence_group = $source.independence_group,
                s.reliability_tier = $source.reliability_tier
            MERGE (d:Document {document_id: $document_id})
            SET d.document_type = $document_type, d.title = $title,
                d.language = $language, d.published_at = $published_at,
                d.created_at = $published_at,
                d.canonical_url = $source.canonical_url,
                d.external_ids = $external_ids,
                d.metrics_json = $metrics_json
            MERGE (v:DocumentVersion {document_version_id: $version_id})
            SET v.raw_sha256 = $artifact.sha256, v.raw_uri = $artifact.uri,
                v.media_type = $artifact.media_type,
                v.byte_length = $artifact.byte_length,
                v.access_status = $artifact.access_status,
                v.coverage = $coverage,
                v.quality_status = $quality_status,
                v.metadata_json = $metadata_json,
                v.created_at = $version_published_at,
                v.version_published_at = $version_published_at,
                v.retrieved_at = CASE
                    WHEN v.retrieved_at IS NULL
                        OR $retrieved_at < v.retrieved_at
                    THEN $retrieved_at ELSE v.retrieved_at END,
                v.metrics_observed_at = CASE
                    WHEN v.metrics_observed_at IS NULL
                        OR $metrics_observed_at < v.metrics_observed_at
                    THEN $metrics_observed_at
                    ELSE v.metrics_observed_at END,
                v.metrics_json = $metrics_json,
                v.country_codes = $country_codes, v.company_ids = $company_ids,
                v.university_ids = $university_ids, v.domain_ids = $domain_ids,
                v.independence_group = $source.independence_group
            MERGE (d)-[:HAS_VERSION]->(v)
            MERGE (v)-[:FROM_SOURCE {record_id: $source.record_id}]->(s)
            """,
            source=document.source.model_dump(),
            document_id=document.document_id,
            document_type=document.document_type.value,
            title=document.title,
            language=document.language,
            published_at=document.published_at,
            version_published_at=document.version_published_at,
            retrieved_at=document.retrieved_at,
            metrics_observed_at=document.metrics_observed_at,
            country_codes=[item.code for item in document.countries],
            company_ids=[
                item.organization_id
                for item in document.organizations
                if item.organization_type == "company"
            ],
            university_ids=[
                item.organization_id
                for item in document.organizations
                if item.organization_type == "university"
            ],
            domain_ids=[item.domain_id for item in document.domains],
            external_ids=[
                identifier.external_id for identifier in document.identifiers
            ],
            version_id=document.document_version_id,
            artifact=document.artifact.model_dump(),
            coverage=document.coverage,
            quality_status=document.quality_status,
            metadata_json=json_value(document.metadata),
            metrics_json=json_value(document.metrics),
        ).consume()

        tx.run(
            """
            MATCH (d:Document {document_id: $document_id})
                  -[r:ASSOCIATED_WITH_COUNTRY|WRITTEN_IN|JURISDICTION]->()
            DELETE r
            """,
            document_id=document.document_id,
        ).consume()
        tx.run(
            """
            MATCH (d:Document {document_id: $document_id})-[r:ABOUT_DOMAIN]->()
            DELETE r
            """,
            document_id=document.document_id,
        ).consume()
        tx.run(
            """
            MATCH (d:Document {document_id: $document_id})
                  -[r:ASSOCIATED_WITH_ORGANIZATION|HAS_AFFILIATION
                      |OWNED_BY|APPLIED_BY]->()
            DELETE r
            """,
            document_id=document.document_id,
        ).consume()

        for country in document.countries:
            relationship = (
                "JURISDICTION"
                if country.role == "jurisdiction"
                else "WRITTEN_IN"
            )
            tx.run(
                f"""
                MATCH (d:Document {{document_id: $document_id}})
                MERGE (c:Country {{country_id: $country_id}})
                SET c.code = $code, c.name = $code,
                    c.first_seen_at = CASE
                        WHEN $observed_at IS NULL THEN c.first_seen_at
                        WHEN c.first_seen_at IS NULL
                            OR $observed_at < c.first_seen_at
                        THEN $observed_at
                        ELSE c.first_seen_at END
                MERGE (d)-[r:{relationship}]->(c)
                SET r.source_role = $role, r.country_code = $code,
                    r.observed_at = $observed_at
                """,
                document_id=document.document_id,
                observed_at=document.published_at,
                **country.model_dump(),
            ).consume()

        for domain in document.domains:
            tx.run(
                """
                MATCH (d:Document {document_id: $document_id})
                MERGE (x:Domain {domain_id: $domain_id})
                SET x.name = $name, x.external_ids = $external_ids,
                    x.first_seen_at = CASE
                        WHEN $observed_at IS NULL THEN x.first_seen_at
                        WHEN x.first_seen_at IS NULL
                            OR $observed_at < x.first_seen_at
                        THEN $observed_at
                        ELSE x.first_seen_at END
                MERGE (d)-[r:ABOUT_DOMAIN]->(x)
                SET r.observed_at = $observed_at
                """,
                document_id=document.document_id,
                observed_at=document.published_at,
                domain_id=domain.domain_id,
                name=domain.name,
                external_ids=[
                    item.external_id for item in domain.external_ids
                ],
            ).consume()
            if domain.parent_name:
                tx.run(
                    """
                    MATCH (child:Domain {domain_id: $domain_id})
                    MERGE (parent:Domain {domain_id: $parent_id})
                    SET parent.name = $parent_name
                    MERGE (child)-[:SUBDOMAIN_OF]->(parent)
                    """,
                    domain_id=domain.domain_id,
                    parent_id=stable_id("domain", domain.parent_name),
                    parent_name=domain.parent_name,
                ).consume()

        for organization in document.organizations:
            schema = load_catalog("graph")
            relationship = cypher_identifier(
                schema["organization_relationships"].get(
                    organization.role, "HAS_AFFILIATION"
                )
            )
            organization_label = schema["organization_labels"].get(
                organization.organization_type
            )
            if organization_label:
                organization_label = cypher_identifier(organization_label)
            tx.run(
                f"""
                MATCH (d:Document {{document_id: $document_id}})
                MERGE (o:Organization {{organization_id: $organization_id}})
                SET o.name = $name, o.organization_type = $organization_type,
                    o.external_ids = $external_ids,
                    o.first_seen_at = CASE
                        WHEN $observed_at IS NULL THEN o.first_seen_at
                        WHEN o.first_seen_at IS NULL
                            OR $observed_at < o.first_seen_at
                        THEN $observed_at
                        ELSE o.first_seen_at END
                MERGE (d)-[r:{relationship}]->(o)
                SET r.source_role = $role, r.observed_at = $observed_at
                """,
                document_id=document.document_id,
                observed_at=document.published_at,
                organization_id=organization.organization_id,
                name=organization.name,
                organization_type=organization.organization_type,
                role=organization.role,
                external_ids=[
                    item.external_id for item in organization.external_ids
                ],
            ).consume()
            if organization_label:
                tx.run(
                    f"""
                    MATCH (o:Organization
                           {{organization_id: $organization_id}})
                    SET o:{organization_label}
                    """,
                    organization_id=organization.organization_id,
                ).consume()
            if organization.country_code:
                tx.run(
                    """
                    MATCH (o:Organization {organization_id: $organization_id})
                    MATCH (c:Country {code: $country_code})
                    MERGE (o)-[:LOCATED_IN]->(c)
                    """,
                    organization_id=organization.organization_id,
                    country_code=organization.country_code,
                ).consume()

        tx.run(
            """
            MATCH (d:Document {document_id: $document_id})
                  -[r:CONTRIBUTED_BY]->()
            DELETE r
            """,
            document_id=document.document_id,
        ).consume()

        for contributor in document.contributors:
            query = """
                MATCH (d:Document {document_id: $document_id})
                MERGE (c:Contributor {contributor_id: $contributor_id})
                SET c.name = $name, c.kind = $kind,
                    c.external_ids = $external_ids,
                    c.first_seen_at = CASE
                        WHEN $observed_at IS NULL THEN c.first_seen_at
                        WHEN c.first_seen_at IS NULL
                            OR $observed_at < c.first_seen_at
                        THEN $observed_at
                        ELSE c.first_seen_at END
                MERGE (d)-[r:CONTRIBUTED_BY]->(c)
                SET r.roles = CASE
                    WHEN $role IN coalesce(r.roles, []) THEN r.roles
                    ELSE coalesce(r.roles, []) + $role END,
                    r.observed_at = $observed_at
                """
            tx.run(
                query,
                document_id=document.document_id,
                observed_at=document.published_at,
                external_ids=[
                    item.external_id for item in contributor.external_ids
                ],
                **contributor.model_dump(
                    exclude={"external_ids", "affiliation_ids"}
                ),
            ).consume()
            for organization_id in contributor.affiliation_ids:
                tx.run(
                    """
                    MATCH (c:Contributor {contributor_id: $contributor_id})
                    MATCH (o:Organization {organization_id: $organization_id})
                    MERGE (c)-[r:AFFILIATED_WITH]->(o)
                    SET r.observed_at = $observed_at
                    """,
                    contributor_id=contributor.contributor_id,
                    organization_id=organization_id,
                    observed_at=document.published_at,
                ).consume()

        for chunk in document.chunks:
            tx.run(
                """
                MATCH (v:DocumentVersion {document_version_id: $version_id})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.kind = $kind, c.text = $text, c.order = $order,
                    c.section_path = $section_path,
                    c.locator_json = $locator_json,
                    c.content_hash = $content_hash,
                    c.parse_status = $parse_status,
                    c.created_at = $observed_at
                MERGE (v)-[:HAS_CHUNK]->(c)
                """,
                version_id=document.document_version_id,
                observed_at=_chunk_date(document, chunk.chunk_id),
                chunk_id=chunk.chunk_id,
                kind=chunk.kind,
                text=chunk.text,
                order=chunk.order,
                section_path=chunk.section_path,
                locator_json=json_value(chunk.locator),
                content_hash=chunk.content_hash,
                parse_status=chunk.parse_status,
            ).consume()

        tx.run(
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
                  -[active:HAS_CHUNK]->(c:Chunk)
            WHERE NOT c.chunk_id IN $chunk_ids
            DELETE active
            """,
            version_id=document.document_version_id,
            chunk_ids=[chunk.chunk_id for chunk in document.chunks],
        ).consume()

    def write_extraction(
        self, document: DocumentEnvelope, result: ExtractionResult
    ) -> None:
        validate_extraction(document, result)
        with self._driver.session(database=self._database) as session:
            session.execute_write(self._write_extraction, document, result)
        logger.debug(
            "Extraction %s written for %s",
            result.run.run_id,
            document.document_version_id,
        )

    def write_processed(
        self, document: DocumentEnvelope, result: ExtractionResult
    ) -> None:
        """Publish document and extraction atomically.

        Incomplete runs preserve earlier chunks.
        """
        validate_extraction(document, result)
        with self._driver.session(database=self._database) as session:
            session.execute_write(self._write_processed, document, result)
        logger.debug(
            "Document %s and run %s (%s) published",
            document.document_version_id,
            result.run.run_id,
            result.run.status,
        )

    @staticmethod
    def _write_processed(
        tx: Any, document: DocumentEnvelope, result: ExtractionResult
    ) -> None:
        existing = tx.run(
            (
                "MATCH (v:DocumentVersion {document_version_id: $version_id}) "
                "RETURN count(v) AS count"
            ),
            version_id=document.document_version_id,
        ).single()
        if result.run.status == "succeeded" or not existing["count"]:
            GraphStore._write_document(tx, document)
        GraphStore._write_extraction(tx, document, result)

    def read_concepts(self) -> List[Concept]:
        with self._driver.session(database=self._database) as session:
            records = session.run(
                """
                MATCH (c)
                WHERE c.concept_id IS NOT NULL AND c.kind IS NOT NULL
                RETURN properties(c) AS properties
                """
            )
            concepts = [
                _concept_from_properties(record["properties"])
                for record in records
            ]
        logger.debug("Read %d concepts from Neo4j", len(concepts))
        return concepts

    def read_training_data(
        self,
    ) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]
    ]:
        with self._driver.session(database=self._database) as session:
            mentions = [
                record.data()
                for record in session.run(
                    """
                MATCH (t:Technology)<-[m:MENTIONS]-(c:Chunk)
                      <-[:HAS_CHUNK]-(:DocumentVersion)
                      <-[:HAS_VERSION]-(d:Document)
                RETURN t.concept_id AS technology_id,
                       t.preferred_label AS technology,
                       d.document_id AS document_id, count(m) AS mentions
                """
                )
            ]
            documents = [
                record.data()
                for record in session.run(
                    """
                MATCH (d:Document)-[:HAS_VERSION]->(:DocumentVersion)
                      -[:FROM_SOURCE]->(s:Source)
                OPTIONAL MATCH (d)-[country_rel]->(country:Country)
                WHERE type(country_rel) IN ['WRITTEN_IN', 'JURISDICTION']
                WITH d, s, collect(DISTINCT country.code) AS countries
                OPTIONAL MATCH (d)-[company_rel]->(company:Company)
                WHERE type(company_rel) IN
                      ['HAS_AFFILIATION', 'OWNED_BY', 'APPLIED_BY']
                WITH d, s, countries,
                     collect(DISTINCT company.organization_id) AS companies
                OPTIONAL MATCH (d)-[university_rel]->(university:University)
                WHERE type(university_rel) IN
                      ['HAS_AFFILIATION', 'OWNED_BY', 'APPLIED_BY']
                WITH d, s, countries, companies,
                     collect(DISTINCT university.organization_id)
                         AS universities
                OPTIONAL MATCH (d)-[:ABOUT_DOMAIN]->(domain:Domain)
                RETURN d.document_id AS document_id,
                       d.created_at AS created_at,
                       s.source_id AS source_id,
                       s.source_family AS source_family,
                       s.independence_group AS independence_group,
                       d.metrics_json AS metrics_json,
                       countries, companies, universities,
                       collect(DISTINCT domain.domain_id) AS domains
                """
                )
            ]
            tasks = [
                record.data()
                for record in session.run(
                    """
                MATCH (t:Technology)-[r:SOLVES]->(task:Task)
                RETURN t.concept_id AS technology_id,
                       task.concept_id AS task_id,
                       r.observed_at AS observed_at
                """
                )
            ]
        return mentions, documents, tasks

    @staticmethod
    def _solution_links(
        document: DocumentEnvelope, result: ExtractionResult
    ) -> List[Tuple[str, str, str, str, int, int]]:
        """A SOLVES edge projects a reviewed, affirmative source claim."""
        kinds = {item.concept_id: item.kind for item in result.concepts}
        predicates = load_catalog("graph")["solution_predicates"]
        links = []
        seen = set()
        for assertion in result.assertions:
            if (
                assertion.predicate not in predicates
                or assertion.status != "accepted"
                or assertion.verification_status != "supported"
                or assertion.polarity != "affirmed"
                or assertion.modality not in ("reported", "observed")
            ):
                continue
            subject, task = (
                assertion.roles.get("subject"),
                assertion.roles.get("task"),
            )
            if (
                kinds.get(subject) != ConceptKind.TECHNOLOGY
                or kinds.get(task) != ConceptKind.TASK
            ):
                continue
            for evidence in assertion.evidence:
                key = (
                    subject,
                    task,
                    evidence.chunk_id,
                    evidence.start,
                    evidence.end,
                )
                if key not in seen:
                    seen.add(key)
                    links.append(
                        (
                            subject,
                            task,
                            evidence.chunk_id,
                            evidence.quote,
                            evidence.start,
                            evidence.end,
                        )
                    )
        return links

    @staticmethod
    def _write_extraction(
        tx: Any, document: DocumentEnvelope, result: ExtractionResult
    ) -> None:
        run = result.run
        metadata = dict(run.metadata)
        metadata["publication_status"] = {
            "succeeded": "published",
            "failed": "failed",
        }.get(run.status, "staged")
        if run.status != "succeeded":
            # Preserve reviewable partial results without replacing an earlier
            # good graph.
            metadata["staged_result"] = result.model_dump(
                exclude={"run"}, mode="json"
            )
            metadata["staged_chunks"] = [
                chunk.model_dump(mode="json") for chunk in document.chunks
            ]
        tx.run(
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
            MERGE (r:ProcessingRun {run_id: $run_id})
            SET r.pipeline_version = $pipeline_version, r.parser = $parser,
                r.model_revision = $model_revision,
                r.prompt_hash = $prompt_hash,
                r.config_hash = $config_hash, r.started_at = $started_at,
                r.status = $status,
                r.metadata_json = $metadata_json, r.trace_json = $trace_json
            MERGE (r)-[:PROCESSED]->(v)
            WITH r
            UNWIND $input_chunk_ids AS input_chunk_id
            MATCH (input:Chunk {chunk_id: input_chunk_id})
            MERGE (r)-[:USED_CHUNK]->(input)
            """,
            version_id=document.document_version_id,
            **run.model_dump(exclude={"metadata", "trace"}),
            metadata_json=json.dumps(metadata, ensure_ascii=False),
            trace_json=json.dumps(run.trace, ensure_ascii=False),
            input_chunk_ids=[chunk.chunk_id for chunk in document.chunks],
        ).consume()

        if run.status != "succeeded":
            return

        tx.run(
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
                  <-[:PROCESSED]-(prior:ProcessingRun)
            WHERE prior.run_id <> $run_id
            SET prior.status = 'superseded'
            """,
            version_id=document.document_version_id,
            run_id=run.run_id,
        ).consume()

        for concept in result.concepts:
            label = concept.kind.value
            tx.run(
                """
                MERGE (c:__LABEL__ {concept_id: $concept_id})
                SET c.kind = $kind, c.preferred_label = $preferred_label,
                    c.name = $preferred_label,
                    c.definition = $definition, c.language = $language,
                    c.status = $status,
                    c.aliases = $aliases,
                    c.normalized_aliases = $normalized_aliases,
                    c.names_json = $names_json,
                    c.first_seen_at = CASE
                        WHEN $observed_at IS NULL THEN c.first_seen_at
                        WHEN c.first_seen_at IS NULL
                            OR $observed_at < c.first_seen_at
                        THEN $observed_at
                        ELSE c.first_seen_at END
                """.replace("__LABEL__", label),
                aliases=list(
                    dict.fromkeys(
                        [
                            concept.preferred_label,
                            *(
                                name.text
                                for name in concept.names
                                if name.status == "accepted"
                            ),
                        ]
                    )
                ),
                normalized_aliases=list(
                    dict.fromkeys(
                        name.normalized_text
                        for name in concept.names
                        if name.status == "accepted"
                    )
                ),
                names_json=json_value(
                    [name.model_dump() for name in concept.names]
                ),
                observed_at=_version_date(document),
                **concept.model_dump(exclude={"names"}, mode="json"),
            ).consume()

        tx.run(
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
                  -[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[r:MENTIONS]->()
            DELETE r
            """,
            version_id=document.document_version_id,
        ).consume()

        tx.run(
            """
            MATCH ()-[r:SOLVES {document_version_id: $version_id}]->()
            DELETE r
            """,
            version_id=document.document_version_id,
        ).consume()

        tx.run(
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
                  -[:HAS_CHUNK]->(chunk:Chunk)
            MATCH ()-[r:HAS_ECONOMIC_EVIDENCE]->(chunk)
            DELETE r
            """,
            version_id=document.document_version_id,
        ).consume()

        tx.run(
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
                  -[active:HAS_ASSERTION]->(:Assertion)
            DELETE active
            """,
            version_id=document.document_version_id,
        ).consume()

        decisions = {
            decision.mention_id: decision for decision in result.resolutions
        }
        for mention in result.mentions:
            decision = decisions.get(mention.mention_id)
            if (
                decision is None
                or decision.concept_id is None
                or decision.status not in ("accepted", "provisional")
            ):
                continue
            tx.run(
                """
                MATCH (chunk:Chunk {chunk_id: $chunk_id})
                MATCH (concept {concept_id: $concept_id})
                MERGE (chunk)-[r:MENTIONS {mention_id: $mention_id}]->(concept)
                SET r.surface_text = $surface_text,
                    r.canonical_text = $canonical_text,
                    r.start = $start, r.end = $end,
                    r.type_candidates = $type_candidates,
                    r.confidence = $confidence,
                    r.mention_role = $mention_role,
                    r.status = $status,
                    r.resolution_status = $resolution_status,
                    r.method = $method, r.score = $score,
                    r.basis = $basis, r.run_id = $run_id,
                    r.observed_at = $observed_at
                """,
                run_id=run.run_id,
                observed_at=_chunk_date(document, mention.chunk_id),
                concept_id=decision.concept_id,
                method=decision.method,
                score=decision.score,
                resolution_status=decision.status,
                basis=decision.basis,
                **mention.model_dump(mode="json"),
            ).consume()

        for (
            technology_id,
            task_id,
            chunk_id,
            quote,
            start,
            end,
        ) in GraphStore._solution_links(document, result):
            tx.run(
                """
                MATCH (technology:Technology {concept_id: $technology_id})
                MATCH (task:Task {concept_id: $task_id})
                MERGE (technology)-[r:SOLVES
                      {document_version_id: $version_id}]->(task)
                SET r.chunk_id = $chunk_id, r.quote = $quote,
                    r.start = $start, r.end = $end,
                    r.method = 'reviewed_assertion', r.run_id = $run_id,
                    r.observed_at = $observed_at
                """,
                technology_id=technology_id,
                task_id=task_id,
                chunk_id=chunk_id,
                quote=quote,
                start=start,
                end=end,
                version_id=document.document_version_id,
                run_id=run.run_id,
                observed_at=_chunk_date(document, chunk_id),
            ).consume()

        for evidence in result.economic_evidence:
            tx.run(
                """
                MATCH (technology:Technology
                       {concept_id: $technology_concept_id})
                MATCH (chunk:Chunk {chunk_id: $chunk_id})
                MERGE (technology)-[r:HAS_ECONOMIC_EVIDENCE
                      {evidence_id: $evidence_id}]->(chunk)
                SET r.category = $category, r.quote = $quote,
                    r.start = $start, r.end = $end,
                    r.amount_text = $amount_text, r.currency = $currency,
                    r.confidence = $confidence, r.status = $status,
                    r.run_id = $run_id,
                    r.polarity = $polarity, r.modality = $modality,
                    r.observed_at = $observed_at
                """,
                run_id=run.run_id,
                observed_at=_chunk_date(document, evidence.chunk_id),
                **evidence.model_dump(),
            ).consume()

        for assertion in result.assertions:
            tx.run(
                """
                MATCH (v:DocumentVersion {document_version_id: $version_id})
                MATCH (r:ProcessingRun {run_id: $run_id})
                MERGE (a:Assertion {assertion_id: $assertion_id})
                SET a.predicate = $predicate,
                    a.qualifiers_json = $qualifiers_json,
                    a.values_json = $values_json, a.polarity = $polarity,
                    a.modality = $modality,
                    a.attribution_kind = $attribution_kind,
                    a.evidence_kind = $evidence_kind,
                    a.extraction_confidence = $extraction_confidence,
                    a.verification_status = $verification_status,
                    a.status = $status,
                    a.observed_at = $observed_at
                MERGE (v)-[:HAS_ASSERTION]->(a)
                MERGE (r)-[creation:CREATED]->(a)
                SET creation.status = $status,
                    creation.verification_status = $verification_status
                """,
                version_id=document.document_version_id,
                run_id=run.run_id,
                observed_at=_chunk_date(
                    document, assertion.evidence[0].chunk_id
                )
                if assertion.evidence
                else _version_date(document),
                qualifiers_json=json_value(assertion.qualifiers),
                values_json=json_value(assertion.values),
                **assertion.model_dump(
                    exclude={
                        "roles",
                        "evidence",
                        "qualifiers",
                        "values",
                        "claim_group_id",
                        "evidence_family_id",
                    }
                ),
            ).consume()

            for role, concept_id in assertion.roles.items():
                relation = load_catalog("graph")["assertion_roles"].get(role)
                if relation is None:
                    raise ValueError(f"unsupported assertion role: {role}")
                relation = cypher_identifier(relation)
                tx.run(
                    f"""
                    MATCH (a:Assertion {{assertion_id: $assertion_id}})
                    MATCH (c {{concept_id: $concept_id}})
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
