from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterator, List, Tuple

from ..core.aio import resolve
from ..core.config import cypher_identifier, load_catalog, resource_path
from ..core.models import (
    SEMANTIC_CANDIDATE_METHOD,
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


BATCH_SIZE = 500
COUNTRY_CODE = re.compile(r"[A-Z]{2}")


def _batches(rows: List[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
    for start in range(0, len(rows), BATCH_SIZE):
        yield rows[start : start + BATCH_SIZE]


async def _run(tx: Any, query: str, **parameters: Any) -> None:
    """Run one statement in a (sync or async) transaction and consume it."""
    result = await resolve(tx.run(query, **parameters))
    await resolve(result.consume())


async def _single(tx: Any, query: str, **parameters: Any) -> Any:
    result = await resolve(tx.run(query, **parameters))
    return await resolve(result.single())


async def _records(session: Any, query: str, **parameters: Any) -> list:
    result = await resolve(session.run(query, **parameters))
    if hasattr(result, "__aiter__"):
        return [record async for record in result]
    return list(result)


def _data(record: Any) -> Dict[str, Any]:
    return record.data() if hasattr(record, "data") else dict(record)


# Kinds are node labels; a label scan per kind replaces a scan of all nodes.
CONCEPT_LABELS = [kind.value for kind in ConceptKind]


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
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            raise RuntimeError(
                "Install the project first: pip install -e ."
            ) from exc
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._vector_indexes: set = set()
        logger.debug("Neo4j driver for %s, database %s", uri, database)

    async def close(self) -> None:
        await resolve(self._driver.close())

    async def verify_connectivity(self) -> None:
        await resolve(self._driver.verify_connectivity())

    async def __aenter__(self) -> "GraphStore":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def ensure_schema(self) -> None:
        async with self._driver.session(database=self._database) as session:
            for query in (
                resource_path("schema", ".cypher")
                .read_text(encoding="utf-8")
                .split(";")
            ):
                if not query.strip():
                    continue
                await _run(session, query)
        logger.debug("Neo4j schema ensured")

    async def processed_materials(self):
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
        async with self._driver.session(database=self._database) as session:
            labels = {
                row["label"]
                for row in await _records(
                    session, "CALL db.labels() YIELD label RETURN label"
                )
            }
            if not {"ProcessingRun", "Source", "Document"} <= labels:
                return
            rows = [_data(record) for record in await _records(session, query)]
            for row in rows:
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

    async def write_document(self, document: DocumentEnvelope) -> None:
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(self._write_document, document)
        logger.debug("Document %s written", document.document_version_id)

    @staticmethod
    async def _write_document(tx: Any, document: DocumentEnvelope) -> None:
        await _run(
            tx,
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
        )

        await _run(
            tx,
            """
            MATCH (d:Document {document_id: $document_id})
                  -[r:ASSOCIATED_WITH_COUNTRY|WRITTEN_IN|JURISDICTION]->()
            DELETE r
            """,
            document_id=document.document_id,
        )
        await _run(
            tx,
            """
            MATCH (d:Document {document_id: $document_id})-[r:ABOUT_DOMAIN]->()
            DELETE r
            """,
            document_id=document.document_id,
        )
        await _run(
            tx,
            """
            MATCH (d:Document {document_id: $document_id})
                  -[r:ASSOCIATED_WITH_ORGANIZATION|HAS_AFFILIATION
                      |OWNED_BY|APPLIED_BY|FUNDED_BY]->()
            DELETE r
            """,
            document_id=document.document_id,
        )

        for country in document.countries:
            relationship = (
                "JURISDICTION"
                if country.role == "jurisdiction"
                else "WRITTEN_IN"
            )
            await _run(
                tx,
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
            )

        for domain in document.domains:
            await _run(
                tx,
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
            )
            if domain.parent_name:
                await _run(
                    tx,
                    """
                    MATCH (child:Domain {domain_id: $domain_id})
                    MERGE (parent:Domain {domain_id: $parent_id})
                    SET parent.name = $parent_name
                    MERGE (child)-[:SUBDOMAIN_OF]->(parent)
                    """,
                    domain_id=domain.domain_id,
                    parent_id=stable_id("domain", domain.parent_name),
                    parent_name=domain.parent_name,
                )

        schema = load_catalog("graph")
        for organization in document.organizations:
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
            await _run(
                tx,
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
            )
            if organization_label:
                await _run(
                    tx,
                    f"""
                    MATCH (o:Organization
                           {{organization_id: $organization_id}})
                    SET o:{organization_label}
                    """,
                    organization_id=organization.organization_id,
                )
            if organization.country_code:
                # The country need not be one of the document's countries.
                await _run(
                    tx,
                    """
                    MATCH (o:Organization {organization_id: $organization_id})
                    MERGE (c:Country {country_id: $country_id})
                    ON CREATE SET c.code = $country_code,
                        c.name = $country_code
                    MERGE (o)-[:LOCATED_IN]->(c)
                    """,
                    organization_id=organization.organization_id,
                    country_id=stable_id("country", organization.country_code),
                    country_code=organization.country_code,
                )

        await _run(
            tx,
            """
            MATCH (d:Document {document_id: $document_id})
                  -[r:CONTRIBUTED_BY]->()
            DELETE r
            """,
            document_id=document.document_id,
        )

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
            await _run(
                tx,
                query,
                document_id=document.document_id,
                observed_at=document.published_at,
                external_ids=[
                    item.external_id for item in contributor.external_ids
                ],
                **contributor.model_dump(
                    exclude={"external_ids", "affiliation_ids"}
                ),
            )
            for organization_id in contributor.affiliation_ids:
                await _run(
                    tx,
                    """
                    MATCH (c:Contributor {contributor_id: $contributor_id})
                    MATCH (o:Organization {organization_id: $organization_id})
                    MERGE (c)-[r:AFFILIATED_WITH]->(o)
                    SET r.observed_at = $observed_at
                    """,
                    contributor_id=contributor.contributor_id,
                    organization_id=organization_id,
                    observed_at=document.published_at,
                )

        # One statement per batch: a full text has hundreds of chunks.
        for batch in _batches(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "kind": chunk.kind,
                    "text": chunk.text,
                    "order": chunk.order,
                    "section_path": chunk.section_path,
                    "locator_json": json_value(chunk.locator),
                    "content_hash": chunk.content_hash,
                    "parse_status": chunk.parse_status,
                    "observed_at": _chunk_date(document, chunk.chunk_id),
                }
                for chunk in document.chunks
            ]
        ):
            await _run(
                tx,
                """
                MATCH (v:DocumentVersion {document_version_id: $version_id})
                UNWIND $rows AS row
                MERGE (c:Chunk {chunk_id: row.chunk_id})
                SET c.kind = row.kind, c.text = row.text, c.order = row.order,
                    c.section_path = row.section_path,
                    c.locator_json = row.locator_json,
                    c.content_hash = row.content_hash,
                    c.parse_status = row.parse_status,
                    c.created_at = row.observed_at
                MERGE (v)-[:HAS_CHUNK]->(c)
                """,
                version_id=document.document_version_id,
                rows=batch,
            )

        await _run(
            tx,
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
                  -[active:HAS_CHUNK]->(c:Chunk)
            WHERE NOT c.chunk_id IN $chunk_ids
            DELETE active
            """,
            version_id=document.document_version_id,
            chunk_ids=[chunk.chunk_id for chunk in document.chunks],
        )

    async def ensure_vector_indexes(self, result: ExtractionResult) -> None:
        """Create per-label vector indexes once the dimension is known.

        Schema changes cannot share the write transaction, and a Neo4j
        without vector support must not block publication.
        """
        if not result.concept_embeddings:
            return
        dimensions = len(next(iter(result.concept_embeddings.values())))
        kinds = {
            concept.concept_id: concept.kind.value
            for concept in result.concepts
        }
        labels = {
            kinds[concept_id]
            for concept_id in result.concept_embeddings
            if concept_id in kinds
        }
        async with self._driver.session(database=self._database) as session:
            for label in sorted(labels - self._vector_indexes):
                name = re.sub(r"(?<!^)(?=[A-Z])", "_", label).lower()
                try:
                    await _run(
                        session,
                        f"""
                        CREATE VECTOR INDEX {name}_embedding IF NOT EXISTS
                        FOR (n:{cypher_identifier(label)}) ON (n.embedding)
                        OPTIONS {{indexConfig: {{
                            `vector.dimensions`: {int(dimensions)},
                            `vector.similarity_function`: 'cosine'
                        }}}}
                        """,
                    )
                except Exception as exc:
                    logger.warning(
                        "Vector index for %s not created (%s); embeddings "
                        "are stored without an index",
                        label,
                        type(exc).__name__,
                    )
                self._vector_indexes.add(label)

    async def write_extraction(
        self, document: DocumentEnvelope, result: ExtractionResult
    ) -> None:
        validate_extraction(document, result)
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                self._write_extraction, document, result
            )
        await self.ensure_vector_indexes(result)
        logger.debug(
            "Extraction %s written for %s",
            result.run.run_id,
            document.document_version_id,
        )

    async def write_processed(
        self, document: DocumentEnvelope, result: ExtractionResult
    ) -> None:
        """Publish document and extraction atomically.

        Incomplete runs preserve earlier chunks.
        """
        validate_extraction(document, result)
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                self._write_processed, document, result
            )
        await self.ensure_vector_indexes(result)
        logger.debug(
            "Document %s and run %s (%s) published",
            document.document_version_id,
            result.run.run_id,
            result.run.status,
        )

    @staticmethod
    async def _write_processed(
        tx: Any, document: DocumentEnvelope, result: ExtractionResult
    ) -> None:
        existing = await _single(
            tx,
            (
                "MATCH (v:DocumentVersion {document_version_id: $version_id}) "
                "OPTIONAL MATCH (v)<-[:PROCESSED]-(r:ProcessingRun) "
                "WHERE r.status = 'succeeded' OR r.published = true "
                "RETURN count(v) AS count, count(r) AS published"
            ),
            version_id=document.document_version_id,
        )
        existing = existing or {}
        status = result.run.status
        # A partial run is published when nothing better is active for this
        # version: at scale one failed packet (e.g. HTTP 429) must not hide a
        # whole document. A later complete run replaces it; a partial rerun
        # never replaces an active projection.
        publish = status == "succeeded" or (
            status == "partial" and not existing.get("published")
        )
        if publish or not existing.get("count"):
            await resolve(GraphStore._write_document(tx, document))
        await resolve(
            GraphStore._write_extraction(tx, document, result, publish)
        )

    async def processed_versions(self, version_ids: List[str]) -> set:
        """Versions that already have a complete extraction.

        Re-running a query or an overlapping crawl must not pay for the same
        LLM extraction twice.
        """
        if not version_ids:
            return set()
        async with self._driver.session(database=self._database) as session:
            records = await _records(
                session,
                """
                UNWIND $ids AS id
                MATCH (v:DocumentVersion {document_version_id: id})
                      <-[:PROCESSED]-(r:ProcessingRun)
                WHERE r.status = 'succeeded' AND r.parser <> 'metadata'
                RETURN DISTINCT id
                """,
                ids=list(version_ids),
            )
        return {record["id"] for record in records}

    async def read_concepts(self) -> List[Concept]:
        # Metadata nodes share some labels (Organization, Country, Domain)
        # but carry no concept_id; UNION removes multi-label duplicates.
        query = "\nUNION\n".join(
            f"MATCH (c:{cypher_identifier(label)}) "
            "WHERE c.concept_id IS NOT NULL AND c.kind IS NOT NULL "
            "RETURN properties(c) AS properties"
            for label in CONCEPT_LABELS
        )
        async with self._driver.session(database=self._database) as session:
            records = await _records(session, query)
            concepts = [
                _concept_from_properties(record["properties"])
                for record in records
            ]
        logger.debug("Read %d concepts from Neo4j", len(concepts))
        return concepts

    async def read_training_data(
        self,
    ) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]
    ]:
        async with self._driver.session(database=self._database) as session:
            mentions = [
                _data(record)
                for record in await _records(
                    session,
                    """
                MATCH (t:Technology)<-[m:MENTIONS]-(c:Chunk)
                      <-[:HAS_CHUNK]-(:DocumentVersion)
                      <-[:HAS_VERSION]-(d:Document)
                RETURN t.concept_id AS technology_id,
                       t.preferred_label AS technology,
                       d.document_id AS document_id, count(m) AS mentions
                """,
                )
            ]
            documents = [
                _data(record)
                for record in await _records(
                    session,
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
                """,
                )
            ]
            tasks = [
                _data(record)
                for record in await _records(
                    session,
                    """
                MATCH (t:Technology)-[r:SOLVES]->(task:Task)
                RETURN t.concept_id AS technology_id,
                       task.concept_id AS task_id,
                       r.observed_at AS observed_at
                """,
                )
            ]
        return mentions, documents, tasks

    async def read_signal_data(self) -> List[Dict[str, Any]]:
        """Dated per-technology signals projected from reviewed text claims:
        organizations, countries, taxonomy parents, maturity and economics.
        """
        query = """
            MATCH (t:Technology)-[r:DEVELOPED_BY|USED_BY|FUNDED_BY
                                 |DEVELOPED_IN|SUBTECHNOLOGY_OF]->(x)
            RETURN t.concept_id AS technology_id, type(r) AS signal,
                   x.concept_id AS target_id, null AS value,
                   r.observed_at AS observed_at
            UNION ALL
            MATCH (t:Technology)-[r:HAS_MATURITY_EVIDENCE]->(:Chunk)
            RETURN t.concept_id AS technology_id, 'MATURITY' AS signal,
                   r.trl AS target_id, r.stage_rank AS value,
                   r.observed_at AS observed_at
            UNION ALL
            MATCH (t:Technology)-[r:HAS_ECONOMIC_EVIDENCE]->(:Chunk)
            WHERE r.polarity = 'affirmed'
            RETURN t.concept_id AS technology_id, 'ECONOMIC' AS signal,
                   r.category AS target_id, r.amount_value AS value,
                   r.observed_at AS observed_at
        """
        async with self._driver.session(database=self._database) as session:
            return [_data(record) for record in await _records(session, query)]

    async def read_taxonomy_input(
        self, kinds: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        """Embedded concepts with their document dates, and reviewed
        (child, parent) SUBTECHNOLOGY_OF pairs.
        """
        concepts = """
            MATCH (c:__LABEL__)
            WHERE c.embedding IS NOT NULL AND c.concept_id IS NOT NULL
            OPTIONAL MATCH (c)<-[:MENTIONS]-(:Chunk)<-[:HAS_CHUNK]-
                  (:DocumentVersion)<-[:HAS_VERSION]-(d:Document)
            RETURN c.concept_id AS concept_id, c.preferred_label AS label,
                   c.kind AS kind, c.embedding AS embedding,
                   c.first_seen_at AS first_seen_at,
                   collect(DISTINCT d.created_at) AS document_dates
        """
        parents = """
            MATCH (child)-[:SUBTECHNOLOGY_OF]->(parent)
            RETURN DISTINCT child.concept_id AS child,
                   parent.concept_id AS parent
        """
        async with self._driver.session(database=self._database) as session:
            rows = []
            for kind in kinds:
                # One label scan per kind instead of a scan of all nodes.
                query = concepts.replace("__LABEL__", cypher_identifier(kind))
                rows.extend(
                    _data(record) for record in await _records(session, query)
                )
            pairs = [
                (item["child"], item["parent"])
                for item in map(_data, await _records(session, parents))
            ]
        return rows, pairs

    async def write_taxonomy(self, taxonomy: Any) -> None:
        """Replace the taxonomy of one version: TaxonomyNode tree plus
        concept placements.
        """
        nodes = [
            {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "path": list(node.path),
                "level": node.level,
                "label": node.label,
                "size": len(node.subtree_ids),
                "documents_last_year": node.documents_last_year,
                "documents_previous_year": node.documents_previous_year,
                "new_share": node.new_share,
            }
            for node in taxonomy.nodes.values()
        ]
        placements: Dict[str, List[Dict[str, Any]]] = {}
        for concept_id, node_id in taxonomy.placement.items():
            kind = cypher_identifier(taxonomy.concepts[concept_id].kind)
            placements.setdefault(kind, []).append(
                {
                    "concept_id": concept_id,
                    "node_id": node_id,
                    "general": concept_id in taxonomy.general_terms,
                }
            )

        async def write(tx: Any) -> None:
            await _run(
                tx,
                """
                MATCH (n:TaxonomyNode {taxonomy_version: $version})
                DETACH DELETE n
                """,
                version=taxonomy.version,
            )
            for batch in _batches(nodes):
                await _run(
                    tx,
                    """
                    UNWIND $rows AS row
                    MERGE (n:TaxonomyNode {node_id: row.node_id})
                    SET n.taxonomy_version = $version,
                        n.snapshot_date = $snapshot,
                        n.path = row.path, n.level = row.level,
                        n.label = row.label, n.name = row.label,
                        n.size = row.size,
                        n.documents_last_year = row.documents_last_year,
                        n.documents_previous_year =
                            row.documents_previous_year,
                        n.new_share = row.new_share
                    """,
                    rows=batch,
                    version=taxonomy.version,
                    snapshot=taxonomy.snapshot.isoformat(),
                )
            await _run(
                tx,
                """
                UNWIND $rows AS row
                WITH row WHERE row.parent_id IS NOT NULL
                MATCH (child:TaxonomyNode {node_id: row.node_id})
                MATCH (parent:TaxonomyNode {node_id: row.parent_id})
                MERGE (child)-[:CHILD_OF]->(parent)
                """,
                rows=nodes,
            )
            for label, rows in placements.items():
                for batch in _batches(rows):
                    await _run(
                        tx,
                        f"""
                        UNWIND $rows AS row
                        MATCH (c:{label} {{concept_id: row.concept_id}})
                        MATCH (n:TaxonomyNode {{node_id: row.node_id}})
                        MERGE (c)-[r:IN_TAXONOMY]->(n)
                        SET r.taxonomy_version = $version,
                            r.general_term = row.general
                        """,
                        rows=batch,
                        version=taxonomy.version,
                    )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(write)
        logger.info(
            "Taxonomy %s written: %d nodes, %d concepts",
            taxonomy.version,
            len(nodes),
            len(taxonomy.placement),
        )

    @staticmethod
    def _reviewed(assertion: Any) -> bool:
        """Only a reviewed, affirmative, reported or observed source claim
        becomes a direct graph edge.
        """
        return (
            assertion.status == "accepted"
            and assertion.verification_status == "supported"
            and assertion.polarity == "affirmed"
            and assertion.modality in ("reported", "observed")
        )

    @staticmethod
    def _projection_links(
        document: DocumentEnvelope, result: ExtractionResult
    ) -> List[Dict[str, Any]]:
        """Direct edges (SOLVES, DEVELOPED_BY, ...) from reviewed
        assertions.
        """
        kinds = {item.concept_id: item.kind.value for item in result.concepts}
        projections = load_catalog("graph")["projections"]
        links = []
        seen = set()
        for assertion in result.assertions:
            rule = projections.get(assertion.predicate)
            if rule is None or not GraphStore._reviewed(assertion):
                continue
            source = assertion.roles.get(rule["source_role"])
            target = assertion.roles.get(rule["target_role"])
            if (
                kinds.get(source) not in rule["source_kinds"]
                or kinds.get(target) not in rule["target_kinds"]
            ):
                continue
            for evidence in assertion.evidence:
                key = (
                    rule["relationship"],
                    source,
                    target,
                    evidence.chunk_id,
                    evidence.start,
                    evidence.end,
                )
                if key not in seen:
                    seen.add(key)
                    links.append(
                        {
                            "relationship": rule["relationship"],
                            "source": source,
                            "source_label": kinds[source],
                            "target": target,
                            "target_label": kinds[target],
                            "assertion_id": assertion.assertion_id,
                            "chunk_id": evidence.chunk_id,
                            "quote": evidence.quote,
                            "start": evidence.start,
                            "end": evidence.end,
                        }
                    )
        return links

    @staticmethod
    def _solution_links(
        document: DocumentEnvelope, result: ExtractionResult
    ) -> List[Tuple[str, str, str, str, int, int]]:
        """A SOLVES edge projects a reviewed, affirmative source claim."""
        return [
            (
                link["source"],
                link["target"],
                link["chunk_id"],
                link["quote"],
                link["start"],
                link["end"],
            )
            for link in GraphStore._projection_links(document, result)
            if link["relationship"] == "SOLVES"
        ]

    @staticmethod
    def _maturity_evidence(result: ExtractionResult) -> List[Dict[str, Any]]:
        """Reviewed stage statements; the TRL is kept only when quoted."""
        graph = load_catalog("graph")
        kinds = {item.concept_id: item.kind.value for item in result.concepts}
        rows = []
        for assertion in result.assertions:
            if assertion.predicate != graph[
                "maturity_predicate"
            ] or not GraphStore._reviewed(assertion):
                continue
            subject = assertion.roles.get("subject")
            stage = assertion.qualifiers.get("stage")
            for evidence in assertion.evidence:
                rows.append(
                    {
                        "subject": subject,
                        "label": kinds[subject],
                        "assertion_id": assertion.assertion_id,
                        "chunk_id": evidence.chunk_id,
                        "quote": evidence.quote,
                        "start": evidence.start,
                        "end": evidence.end,
                        "stage": stage,
                        "stage_rank": graph["maturity_stage_rank"].get(stage),
                        "trl": assertion.qualifiers.get("trl"),
                    }
                )
        return rows

    @staticmethod
    async def _write_extraction(
        tx: Any,
        document: DocumentEnvelope,
        result: ExtractionResult,
        publish: bool | None = None,
    ) -> None:
        run = result.run
        if publish is None:
            publish = run.status == "succeeded"
        metadata = dict(run.metadata)
        metadata["publication_status"] = (
            ("published" if run.status == "succeeded" else "published_partial")
            if publish
            else {"failed": "failed"}.get(run.status, "staged")
        )
        if not publish:
            # Preserve reviewable partial results without replacing an earlier
            # good graph.
            metadata["staged_result"] = result.model_dump(
                exclude={"run", "concept_embeddings"}, mode="json"
            )
            metadata["staged_chunks"] = [
                chunk.model_dump(mode="json") for chunk in document.chunks
            ]
        await _run(
            tx,
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
            MERGE (r:ProcessingRun {run_id: $run_id})
            SET r.pipeline_version = $pipeline_version, r.parser = $parser,
                r.model_revision = $model_revision,
                r.prompt_hash = $prompt_hash,
                r.config_hash = $config_hash, r.started_at = $started_at,
                r.status = $status, r.published = $published,
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
            published=publish,
        )

        if not publish:
            return

        await _run(
            tx,
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
                  <-[:PROCESSED]-(prior:ProcessingRun)
            WHERE prior.run_id <> $run_id
            SET prior.status = 'superseded'
            """,
            version_id=document.document_version_id,
            run_id=run.run_id,
        )

        for concept in result.concepts:
            label = concept.kind.value
            await _run(
                tx,
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
            )

        # Every concept a mention, role or projection points to is in
        # result.concepts (validate_extraction), so matches can use a label
        # and its concept_id constraint instead of scanning all nodes.
        labels = {
            concept.concept_id: cypher_identifier(concept.kind.value)
            for concept in result.concepts
        }
        graph = load_catalog("graph")

        for concept in result.concepts:
            if concept.kind == ConceptKind.COUNTRY and COUNTRY_CODE.fullmatch(
                concept.preferred_label
            ):
                # An ISO-coded country from text is the same country as the
                # metadata-level one.
                await _run(
                    tx,
                    """
                    MATCH (c:Country {concept_id: $concept_id})
                    MERGE (x:Country {country_id: $country_id})
                    ON CREATE SET x.code = $code, x.name = $code
                    MERGE (c)-[:SAME_AS]->(x)
                    """,
                    concept_id=concept.concept_id,
                    country_id=stable_id("country", concept.preferred_label),
                    code=concept.preferred_label,
                )

        embedded: Dict[str, List[Dict[str, Any]]] = {}
        for concept_id, vector in result.concept_embeddings.items():
            if concept_id in labels:
                embedded.setdefault(labels[concept_id], []).append(
                    {"concept_id": concept_id, "vector": vector}
                )
        for label, rows in embedded.items():
            for batch in _batches(rows):
                await _run(
                    tx,
                    f"""
                    UNWIND $rows AS row
                    MATCH (c:{label} {{concept_id: row.concept_id}})
                    SET c.embedding = row.vector,
                        c.embedding_model = $model
                    """,
                    rows=batch,
                    model=result.embedding_model,
                )

        # A semantic match is a review candidate, never an identity.
        for decision in result.resolutions:
            if (
                decision.method != SEMANTIC_CANDIDATE_METHOD
                or decision.concept_id not in labels
            ):
                continue
            for candidate in decision.candidates:
                kind = candidate.get("kind")
                if kind not in {item.value for item in ConceptKind}:
                    continue
                await _run(
                    tx,
                    f"""
                    MATCH (a:{labels[decision.concept_id]}
                           {{concept_id: $source}})
                    MATCH (b:{cypher_identifier(kind)} {{concept_id: $target}})
                    MERGE (a)-[r:POSSIBLY_SAME_AS]->(b)
                    SET r.score = $score, r.cosine = $cosine,
                        r.method = $method, r.run_id = $run_id,
                        r.review_status = 'pending'
                    """,
                    source=decision.concept_id,
                    target=candidate["concept_id"],
                    score=candidate.get("score"),
                    cosine=candidate.get("cosine"),
                    method=decision.method,
                    run_id=run.run_id,
                )

        await _run(
            tx,
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
                  -[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[r:MENTIONS]->()
            DELETE r
            """,
            version_id=document.document_version_id,
        )

        for projection in graph["projections"].values():
            relationship = cypher_identifier(projection["relationship"])
            await _run(
                tx,
                f"""
                MATCH ()-[r:{relationship}]->()
                WHERE r.document_version_id = $version_id
                DELETE r
                """,
                version_id=document.document_version_id,
            )

        await _run(
            tx,
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
                  -[:HAS_CHUNK]->(chunk:Chunk)
            MATCH ()-[r:HAS_ECONOMIC_EVIDENCE|HAS_MATURITY_EVIDENCE]->(chunk)
            DELETE r
            """,
            version_id=document.document_version_id,
        )

        await _run(
            tx,
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
                  -[active:HAS_ASSERTION]->(:Assertion)
            DELETE active
            """,
            version_id=document.document_version_id,
        )

        decisions = {
            decision.mention_id: decision for decision in result.resolutions
        }
        mention_rows: Dict[str, List[Dict[str, Any]]] = {}
        for mention in result.mentions:
            decision = decisions.get(mention.mention_id)
            if (
                decision is None
                or decision.concept_id is None
                or decision.status not in ("accepted", "provisional")
            ):
                continue
            mention_rows.setdefault(labels[decision.concept_id], []).append(
                {
                    **mention.model_dump(mode="json"),
                    "concept_id": decision.concept_id,
                    "method": decision.method,
                    "score": decision.score,
                    "resolution_status": decision.status,
                    "basis": decision.basis,
                    "observed_at": _chunk_date(document, mention.chunk_id),
                }
            )
        for label, rows in mention_rows.items():
            for batch in _batches(rows):
                await _run(
                    tx,
                    f"""
                    UNWIND $rows AS row
                    MATCH (chunk:Chunk {{chunk_id: row.chunk_id}})
                    MATCH (concept:{label} {{concept_id: row.concept_id}})
                    MERGE (chunk)-[r:MENTIONS
                          {{mention_id: row.mention_id}}]->(concept)
                    SET r.surface_text = row.surface_text,
                        r.canonical_text = row.canonical_text,
                        r.start = row.start, r.end = row.end,
                        r.type_candidates = row.type_candidates,
                        r.confidence = row.confidence,
                        r.mention_role = row.mention_role,
                        r.status = row.status,
                        r.resolution_status = row.resolution_status,
                        r.method = row.method, r.score = row.score,
                        r.basis = row.basis, r.run_id = $run_id,
                        r.observed_at = row.observed_at
                    """,
                    rows=batch,
                    run_id=run.run_id,
                )

        for link in GraphStore._projection_links(document, result):
            await _run(
                tx,
                f"""
                MATCH (source:{cypher_identifier(link["source_label"])}
                       {{concept_id: $source}})
                MATCH (target:{cypher_identifier(link["target_label"])}
                       {{concept_id: $target}})
                MERGE (source)-[r:{cypher_identifier(link["relationship"])}
                      {{document_version_id: $version_id}}]->(target)
                SET r.chunk_id = $chunk_id, r.quote = $quote,
                    r.start = $start, r.end = $end,
                    r.assertion_id = $assertion_id,
                    r.method = 'reviewed_assertion', r.run_id = $run_id,
                    r.observed_at = $observed_at
                """,
                source=link["source"],
                target=link["target"],
                chunk_id=link["chunk_id"],
                quote=link["quote"],
                start=link["start"],
                end=link["end"],
                assertion_id=link["assertion_id"],
                version_id=document.document_version_id,
                run_id=run.run_id,
                observed_at=_chunk_date(document, link["chunk_id"]),
            )

        for row in GraphStore._maturity_evidence(result):
            await _run(
                tx,
                f"""
                MATCH (subject:{cypher_identifier(row["label"])}
                       {{concept_id: $subject}})
                MATCH (chunk:Chunk {{chunk_id: $chunk_id}})
                MERGE (subject)-[r:HAS_MATURITY_EVIDENCE
                      {{assertion_id: $assertion_id, start: $start}}]->(chunk)
                SET r.stage = $stage, r.stage_rank = $stage_rank,
                    r.trl = $trl, r.quote = $quote, r.end = $end,
                    r.run_id = $run_id, r.observed_at = $observed_at
                """,
                subject=row["subject"],
                chunk_id=row["chunk_id"],
                assertion_id=row["assertion_id"],
                start=row["start"],
                end=row["end"],
                quote=row["quote"],
                stage=row["stage"],
                stage_rank=row["stage_rank"],
                trl=row["trl"],
                run_id=run.run_id,
                observed_at=_chunk_date(document, row["chunk_id"]),
            )

        for evidence in result.economic_evidence:
            await _run(
                tx,
                """
                MATCH (technology:Technology
                       {concept_id: $technology_concept_id})
                MATCH (chunk:Chunk {chunk_id: $chunk_id})
                MERGE (technology)-[r:HAS_ECONOMIC_EVIDENCE
                      {evidence_id: $evidence_id}]->(chunk)
                SET r.category = $category, r.quote = $quote,
                    r.start = $start, r.end = $end,
                    r.amount_text = $amount_text,
                    r.amount_value = $amount_value, r.currency = $currency,
                    r.confidence = $confidence, r.status = $status,
                    r.run_id = $run_id,
                    r.polarity = $polarity, r.modality = $modality,
                    r.observed_at = $observed_at
                """,
                run_id=run.run_id,
                observed_at=_chunk_date(document, evidence.chunk_id),
                **evidence.model_dump(),
            )

        for assertion in result.assertions:
            await _run(
                tx,
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
            )

            for role, concept_id in assertion.roles.items():
                relation = graph["assertion_roles"].get(role)
                if relation is None:
                    raise ValueError(f"unsupported assertion role: {role}")
                relation = cypher_identifier(relation)
                await _run(
                    tx,
                    f"""
                    MATCH (a:Assertion {{assertion_id: $assertion_id}})
                    MATCH (c:{labels[concept_id]} {{concept_id: $concept_id}})
                    MERGE (a)-[:{relation}]->(c)
                    """,
                    assertion_id=assertion.assertion_id,
                    concept_id=concept_id,
                )

            for evidence in assertion.evidence:
                await _run(
                    tx,
                    """
                    MATCH (a:Assertion {assertion_id: $assertion_id})
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (a)-[e:SUPPORTED_BY {start: $start, end: $end}]->(c)
                    SET e.quote = $quote, e.supports_fields = $supports_fields
                    """,
                    assertion_id=assertion.assertion_id,
                    **evidence.model_dump(),
                )

            if assertion.claim_group_id:
                await _run(
                    tx,
                    """
                    MATCH (a:Assertion {assertion_id: $assertion_id})
                    MERGE (g:ClaimGroup {claim_group_id: $group_id})
                    MERGE (a)-[:IN_CLAIM_GROUP]->(g)
                    """,
                    assertion_id=assertion.assertion_id,
                    group_id=assertion.claim_group_id,
                )
            if assertion.evidence_family_id:
                await _run(
                    tx,
                    """
                    MATCH (a:Assertion {assertion_id: $assertion_id})
                    MERGE (f:EvidenceFamily {family_id: $family_id})
                    MERGE (a)-[:FROM_EVIDENCE_FAMILY]->(f)
                    """,
                    assertion_id=assertion.assertion_id,
                    family_id=assertion.evidence_family_id,
                )
