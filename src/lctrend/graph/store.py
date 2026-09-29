from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..core.aio import resolve
from ..core.config import cypher_identifier, load_catalog, resource_path
from ..core.models import (
    AMBIGUOUS_COLLISION_METHOD,
    DECLARED_ALIAS_METHOD,
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
from ..core.organizations import (
    country_names,
    display_rank,
    organization_identity,
)
from ..extraction.lexical import (
    KEY_VERSION,
    KIND_RANK,
    VOTED_FAMILIES,
    kind_family,
    settled_kind,
)
from ..extraction.resolver import concept_text
from .works import (
    FoundWork,
    WorkKey,
    WorkPlan,
    document_work_keys,
    plan_work,
)

logger = logging.getLogger(__name__)


def _version_date(document: DocumentEnvelope) -> Any:
    """Content date of a version. Collection time is not a publication
    date: undated content stays undated (``None``).
    """
    return getattr(document, "version_published_at", None) or getattr(
        document, "published_at", None
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
# Graphs whose constraints and indexes this process already ensured: the
# schema is idempotent, but ~40 statements per job cost ~40 round trips.
_SCHEMA_READY: set = set()
COUNTRY_CODE = re.compile(r"[A-Z]{2}")


def _country_row(code: str) -> Dict[str, Any]:
    """Identity and full names of a country node (name is Russian)."""
    name, name_en = country_names(code)
    return {
        "country_id": stable_id("country", code),
        "code": code,
        "name": name,
        "name_en": name_en,
    }


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


def _lucene_phrase(text: str) -> str:
    """A literal phrase query: the model's text is never Lucene syntax."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _data(record: Any) -> Dict[str, Any]:
    return record.data() if hasattr(record, "data") else dict(record)


# Kinds are node labels; a label scan per kind replaces a scan of all nodes.
CONCEPT_LABELS = [kind.value for kind in ConceptKind]


def evidence_chunk_ids(result: Optional[ExtractionResult]) -> Optional[set]:
    """Chunks the graph keeps as nodes, or None to keep every chunk.

    graph.json ``chunk_nodes``: "evidence" keeps the chunks something
    stands on (a mention, a claim's quote, economic evidence, an evidence
    vector); the rest of the text stays in the raw snapshot and in the
    run's ``input_chunk_ids``. "all" keeps every chunk.
    """
    if load_catalog("graph").get("chunk_nodes", "evidence") == "all":
        return None
    if result is None:
        return set()
    return (
        {mention.chunk_id for mention in result.mentions}
        | {
            span.chunk_id
            for assertion in result.assertions
            for span in assertion.evidence
        }
        | {item.chunk_id for item in result.economic_evidence}
        | set(result.chunk_embeddings)
    )


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
        identity_key=properties.get("identity_key"),
        label_counts=json.loads(properties.get("label_counts_json") or "{}"),
        kind_counts=json.loads(properties.get("kind_counts_json") or "{}"),
        profile=json.loads(
            properties.get("technology_profile_json") or "null"
        ),
    )


# Kinds of one identity family (extraction.lexical.kind_family). An
# organization node is only ever relabeled upwards, so concurrent writers
# cannot downgrade it; a technology node takes the kind its mentions voted.
FAMILY_RANK = KIND_RANK
# The ranked labels each concept id is stored under, and its stored kind
# votes.
FAMILY_LABELS_QUERY = (
    "UNWIND $ids AS id\n"
    + "".join(
        f"OPTIONAL MATCH (n{index}:{label} {{concept_id: id}})\n"
        for index, label in enumerate(FAMILY_RANK)
    )
    + "RETURN id AS concept_id, "
    + ", ".join(
        f"n{index} IS NOT NULL AS {label}"
        for index, label in enumerate(FAMILY_RANK)
    )
    + ", coalesce("
    + ", ".join(
        f"n{index}.kind_counts_json" for index in range(len(FAMILY_RANK))
    )
    + ") AS kind_counts_json"
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
        self._driver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            # A remote server drops connections for seconds at a time;
            # managed transactions retry them for up to two minutes
            # (driver default: 30 s) instead of failing the document.
            max_transaction_retry_time=120.0,
        )
        self._database = database
        self._schema_key = (uri, database)
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
        """Create constraints and indexes once per process and graph."""
        key = getattr(self, "_schema_key", None)
        if key is not None and key in _SCHEMA_READY:
            return
        async with self._driver.session(database=self._database) as session:
            for query in (
                resource_path("schema", ".cypher")
                .read_text(encoding="utf-8")
                .split(";")
            ):
                if not query.strip():
                    continue
                await _run(session, query)
        if key is not None:
            _SCHEMA_READY.add(key)
        logger.debug("Neo4j schema ensured")

    async def processed_materials(self):
        """Stream prior extractions into the crawler's deduplication ledger:
        the graph decides which materials are duplicates.

        A document-only import is not a completed extraction. Article,
        repository and package identities remain separate; any other source
        (grants, vacancies) is identified as ``source:record_id``.
        """
        from ..ingest.discovery import material_identity

        query = """
            MATCH (run:ProcessingRun)-[:PROCESSED]->(v:DocumentVersion)
                  <-[:HAS_VERSION]-(d:Document)
            MATCH (v)-[origin:FROM_SOURCE]->(s:Source)
            WHERE run.status IN ['succeeded', 'partial']
              AND run.parser <> 'metadata'
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

    @staticmethod
    async def _write_metrics(
        tx: Any, document: DocumentEnvelope, observed: str
    ) -> None:
        """One dated metrics observation of the version.

        Counters change without the content changing, so they are kept as
        a history on the unchanged version instead of forcing a new one.
        """
        if not document.metrics:
            return
        observed_at = document.metrics_observed_at or observed
        await _run(
            tx,
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
            MERGE (m:MetricsObservation {observation_id: $observation_id})
            SET m.observed_at = $observed_at, m.metrics_json = $metrics_json,
                m.document_version_id = $version_id
            MERGE (v)-[:HAS_METRICS]->(m)
            """,
            version_id=document.document_version_id,
            observation_id=stable_id(
                "metrics", document.document_version_id, observed_at
            ),
            observed_at=observed_at,
            metrics_json=json_value(document.metrics),
        )

    async def record_metrics(self, document: DocumentEnvelope) -> None:
        """Keep the counters of a re-fetched, already processed version."""
        observed = (
            document.retrieved_at or datetime.now(timezone.utc).isoformat()
        )

        async def write(tx: Any) -> None:
            await GraphStore._write_metrics(tx, document, observed)
            observed_at = document.metrics_observed_at or observed
            await _run(
                tx,
                """
                MATCH (v:DocumentVersion {document_version_id: $version_id})
                SET v.metrics_json = CASE
                        WHEN v.metrics_observed_at IS NULL
                            OR $observed_at >= v.metrics_observed_at
                        THEN $metrics_json ELSE v.metrics_json END,
                    v.metrics_observed_at = CASE
                        WHEN v.metrics_observed_at IS NULL
                            OR $observed_at > v.metrics_observed_at
                        THEN $observed_at ELSE v.metrics_observed_at END
                """,
                version_id=document.document_version_id,
                observed_at=observed_at,
                metrics_json=json_value(document.metrics),
            )

        if not document.metrics:
            return
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(write)

    @staticmethod
    async def _write_parties(tx: Any, document: DocumentEnvelope) -> None:
        """Countries, domains, organizations and people of a document.

        One UNWIND statement per relationship type and label, not one per
        item: each statement costs a network round trip (D-1).
        """
        schema = load_catalog("graph")
        organization_relationships = sorted(
            {
                "ASSOCIATED_WITH_ORGANIZATION",
                "HAS_AFFILIATION",
                *schema["organization_relationships"].values(),
            }
        )
        replaced = "|".join(
            [
                "ASSOCIATED_WITH_COUNTRY",
                "WRITTEN_IN",
                "JURISDICTION",
                "ABOUT_DOMAIN",
                "CONTRIBUTED_BY",
                *map(cypher_identifier, organization_relationships),
            ]
        )
        await _run(
            tx,
            f"""
            MATCH (d:Document {{document_id: $document_id}})-[r:{replaced}]->()
            DELETE r
            """,
            document_id=document.document_id,
        )
        observed_at = document.published_at

        countries: Dict[str, List[Dict[str, Any]]] = {}
        for country in document.countries:
            relationship = (
                "JURISDICTION"
                if country.role == "jurisdiction"
                else "WRITTEN_IN"
            )
            countries.setdefault(relationship, []).append(
                {**country.model_dump(), **_country_row(country.code)}
            )
        for relationship, rows in countries.items():
            await _run(
                tx,
                f"""
                MATCH (d:Document {{document_id: $document_id}})
                UNWIND $rows AS row
                MERGE (c:Country {{country_id: row.country_id}})
                SET c.code = row.code, c.name = row.name,
                    c.name_en = row.name_en,
                    c.first_seen_at = CASE
                        WHEN $observed_at IS NULL THEN c.first_seen_at
                        WHEN c.first_seen_at IS NULL
                            OR $observed_at < c.first_seen_at
                        THEN $observed_at
                        ELSE c.first_seen_at END
                MERGE (d)-[r:{relationship}]->(c)
                SET r.source_role = row.role, r.country_code = row.code,
                    r.observed_at = $observed_at
                """,
                document_id=document.document_id,
                observed_at=observed_at,
                rows=rows,
            )

        if document.domains:
            await _run(
                tx,
                """
                MATCH (d:Document {document_id: $document_id})
                UNWIND $rows AS row
                MERGE (x:Domain {domain_id: row.domain_id})
                SET x.name = row.name, x.external_ids = row.external_ids,
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
                observed_at=observed_at,
                rows=[
                    {
                        "domain_id": domain.domain_id,
                        "name": domain.name,
                        "external_ids": [
                            item.external_id for item in domain.external_ids
                        ],
                    }
                    for domain in document.domains
                ],
            )
        parents = [
            {
                "domain_id": domain.domain_id,
                "parent_id": stable_id("domain", domain.parent_name),
                "parent_name": domain.parent_name,
            }
            for domain in document.domains
            if domain.parent_name
        ]
        if parents:
            await _run(
                tx,
                """
                UNWIND $rows AS row
                MATCH (child:Domain {domain_id: row.domain_id})
                MERGE (parent:Domain {domain_id: row.parent_id})
                SET parent.name = row.parent_name
                MERGE (child)-[:SUBDOMAIN_OF]->(parent)
                """,
                rows=parents,
            )

        organizations: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for organization in document.organizations:
            relationship = cypher_identifier(
                schema["organization_relationships"].get(
                    organization.role, "HAS_AFFILIATION"
                )
            )
            organization_label = schema["organization_labels"].get(
                organization.organization_type
            )
            organizations.setdefault(
                (
                    relationship,
                    cypher_identifier(organization_label)
                    if organization_label
                    else "",
                ),
                [],
            ).append(
                {
                    "organization_id": organization.organization_id,
                    "name": organization.name,
                    "name_rank": display_rank(organization.name),
                    "organization_type": organization.organization_type,
                    "role": organization.role,
                    "external_ids": [
                        item.external_id for item in organization.external_ids
                    ],
                }
            )
        for (relationship, label), rows in organizations.items():
            await _run(
                tx,
                f"""
                MATCH (d:Document {{document_id: $document_id}})
                UNWIND $rows AS row
                MERGE (o:Organization {{organization_id: row.organization_id}})
                // One company reaches this node from several sources and
                // countries: the best-written name stays, identifiers add up.
                WITH d, row, o,
                     o.name IS NULL
                     OR row.name_rank < coalesce(o.name_rank, 1000000)
                     AS renamed
                SET o.name = CASE WHEN renamed THEN row.name ELSE o.name END,
                    o.name_rank = CASE WHEN renamed THEN row.name_rank
                        ELSE o.name_rank END,
                    o.organization_type = row.organization_type,
                    o.external_ids = coalesce(o.external_ids, [])
                        + [item IN row.external_ids
                           WHERE NOT item IN coalesce(o.external_ids, [])],
                    o.first_seen_at = CASE
                        WHEN $observed_at IS NULL THEN o.first_seen_at
                        WHEN o.first_seen_at IS NULL
                            OR $observed_at < o.first_seen_at
                        THEN $observed_at
                        ELSE o.first_seen_at END
                {f"SET o:{label}" if label else ""}
                MERGE (d)-[r:{relationship}]->(o)
                SET r.source_role = row.role, r.observed_at = $observed_at
                """,
                document_id=document.document_id,
                observed_at=observed_at,
                rows=rows,
            )
        located = [
            {
                "organization_id": organization.organization_id,
                **_country_row(organization.country_code),
            }
            for organization in document.organizations
            if organization.country_code
        ]
        if located:
            # The country need not be one of the document's countries.
            await _run(
                tx,
                """
                UNWIND $rows AS row
                MATCH (o:Organization {organization_id: row.organization_id})
                MERGE (c:Country {country_id: row.country_id})
                ON CREATE SET c.code = row.code, c.name = row.name,
                    c.name_en = row.name_en
                MERGE (o)-[:LOCATED_IN]->(c)
                """,
                rows=located,
            )

        if document.contributors:
            await _run(
                tx,
                """
                MATCH (d:Document {document_id: $document_id})
                UNWIND $rows AS row
                MERGE (c:Contributor {contributor_id: row.contributor_id})
                SET c.name = row.name, c.kind = row.kind,
                    c.external_ids = row.external_ids,
                    c.first_seen_at = CASE
                        WHEN $observed_at IS NULL THEN c.first_seen_at
                        WHEN c.first_seen_at IS NULL
                            OR $observed_at < c.first_seen_at
                        THEN $observed_at
                        ELSE c.first_seen_at END
                MERGE (d)-[r:CONTRIBUTED_BY]->(c)
                SET r.roles = CASE
                    WHEN row.role IN coalesce(r.roles, []) THEN r.roles
                    ELSE coalesce(r.roles, []) + row.role END,
                    r.observed_at = $observed_at
                """,
                document_id=document.document_id,
                observed_at=observed_at,
                rows=[
                    {
                        "external_ids": [
                            item.external_id
                            for item in contributor.external_ids
                        ],
                        **contributor.model_dump(
                            exclude={"external_ids", "affiliation_ids"}
                        ),
                    }
                    for contributor in document.contributors
                ],
            )
        affiliations = [
            {
                "contributor_id": contributor.contributor_id,
                "organization_id": organization_id,
            }
            for contributor in document.contributors
            for organization_id in contributor.affiliation_ids
        ]
        if affiliations:
            await _run(
                tx,
                """
                UNWIND $rows AS row
                MATCH (c:Contributor {contributor_id: row.contributor_id})
                MATCH (o:Organization {organization_id: row.organization_id})
                MERGE (c)-[r:AFFILIATED_WITH]->(o)
                SET r.observed_at = $observed_at
                """,
                observed_at=observed_at,
                rows=affiliations,
            )

    @staticmethod
    async def _write_economic_facts(
        tx: Any, document: DocumentEnvelope
    ) -> None:
        """Money of the record (grant award, salary offer) as dated nodes.

        The version also keeps them as JSON, so a snapshot reads them with
        the version's own date and parties. The organizations are the
        document's parties, written just before.
        """
        rows = [
            fact.model_dump(mode="json") for fact in document.economic_facts
        ]
        await _run(
            tx,
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
            SET v.economic_facts_json = $facts_json
            WITH v
            OPTIONAL MATCH (v)-[old:HAS_ECONOMIC_FACT]->(:EconomicFact)
            DELETE old
            """,
            version_id=document.document_version_id,
            facts_json=json_value(rows),
        )
        if not rows:
            return
        await _run(
            tx,
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
            UNWIND $rows AS row
            MERGE (f:EconomicFact {fact_id: row.fact_id})
            SET f.category = row.category, f.amount = row.amount,
                f.amount_max = row.amount_max, f.currency = row.currency,
                f.period = row.period, f.observed_at = row.observed_at,
                f.source_field = row.source_field,
                f.amount_usd_real = row.amount_usd_real,
                f.amount_max_usd_real = row.amount_max_usd_real,
                f.real_base_year = row.real_base_year,
                f.real_status = row.real_status,
                f.document_type = $document_type,
                f.source_family = $source_family
            MERGE (v)-[:HAS_ECONOMIC_FACT]->(f)
            WITH f, row
            OPTIONAL MATCH (f)-[old:RECEIVED_BY|PAID_BY]->(:Organization)
            DELETE old
            WITH DISTINCT f, row
            OPTIONAL MATCH (recipient:Organization
                {organization_id: row.recipient_organization_id})
            OPTIONAL MATCH (payer:Organization
                {organization_id: row.payer_organization_id})
            FOREACH (_ IN CASE WHEN recipient IS NULL THEN [] ELSE [1] END |
                MERGE (f)-[:RECEIVED_BY]->(recipient))
            FOREACH (_ IN CASE WHEN payer IS NULL THEN [] ELSE [1] END |
                MERGE (f)-[:PAID_BY]->(payer))
            """,
            version_id=document.document_version_id,
            document_type=document.document_type.value,
            source_family=document.source.source_family,
            rows=rows,
        )

    @staticmethod
    async def _write_assertions(
        tx: Any,
        document: DocumentEnvelope,
        result: ExtractionResult,
        labels: Dict[str, str],
    ) -> None:
        """Assertions with their roles, evidence and groups, batched."""
        if not result.assertions:
            return
        run = result.run
        graph = load_catalog("graph")
        rows = [
            {
                "observed_at": _chunk_date(
                    document, assertion.evidence[0].chunk_id
                )
                if assertion.evidence
                else _version_date(document),
                "qualifiers_json": json_value(assertion.qualifiers),
                "values_json": json_value(assertion.values),
                **assertion.model_dump(
                    mode="json",
                    exclude={
                        "roles",
                        "evidence",
                        "qualifiers",
                        "values",
                        "claim_group_id",
                        "evidence_family_id",
                    },
                ),
            }
            for assertion in result.assertions
        ]
        for batch in _batches(rows):
            await _run(
                tx,
                """
                MATCH (v:DocumentVersion {document_version_id: $version_id})
                MATCH (r:ProcessingRun {run_id: $run_id})
                UNWIND $rows AS row
                MERGE (a:Assertion {assertion_id: row.assertion_id})
                SET a.predicate = row.predicate,
                    a.qualifiers_json = row.qualifiers_json,
                    a.values_json = row.values_json,
                    a.polarity = row.polarity,
                    a.modality = row.modality,
                    a.attribution_kind = row.attribution_kind,
                    a.evidence_kind = row.evidence_kind,
                    a.extraction_confidence = row.extraction_confidence,
                    a.verification_status = row.verification_status,
                    a.status = row.status,
                    a.observed_at = row.observed_at,
                    a.recorded_at = $recorded_at
                MERGE (v)-[:HAS_ASSERTION]->(a)
                MERGE (r)-[creation:CREATED]->(a)
                SET creation.status = row.status,
                    creation.verification_status = row.verification_status
                """,
                version_id=document.document_version_id,
                run_id=run.run_id,
                recorded_at=run.started_at,
                rows=batch,
            )

        roles: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for assertion in result.assertions:
            for role, concept_id in assertion.roles.items():
                relation = graph["assertion_roles"].get(role)
                if relation is None:
                    raise ValueError(f"unsupported assertion role: {role}")
                roles.setdefault(
                    (cypher_identifier(relation), labels[concept_id]), []
                ).append(
                    {
                        "assertion_id": assertion.assertion_id,
                        "concept_id": concept_id,
                    }
                )
        for (relation, label), rows in roles.items():
            await _run(
                tx,
                f"""
                UNWIND $rows AS row
                MATCH (a:Assertion {{assertion_id: row.assertion_id}})
                MATCH (c:{label} {{concept_id: row.concept_id}})
                MERGE (a)-[:{relation}]->(c)
                """,
                rows=rows,
            )

        evidence = [
            {
                "assertion_id": assertion.assertion_id,
                **span.model_dump(mode="json"),
            }
            for assertion in result.assertions
            for span in assertion.evidence
        ]
        for batch in _batches(evidence):
            await _run(
                tx,
                """
                UNWIND $rows AS row
                MATCH (a:Assertion {assertion_id: row.assertion_id})
                MATCH (c:Chunk {chunk_id: row.chunk_id})
                MERGE (a)-[e:SUPPORTED_BY
                      {start: row.start, end: row.end}]->(c)
                SET e.quote = row.quote,
                    e.supports_fields = row.supports_fields
                """,
                rows=batch,
            )

        groups = [
            {
                "assertion_id": assertion.assertion_id,
                "group_id": assertion.claim_group_id,
            }
            for assertion in result.assertions
            if assertion.claim_group_id
        ]
        if groups:
            await _run(
                tx,
                """
                UNWIND $rows AS row
                MATCH (a:Assertion {assertion_id: row.assertion_id})
                MERGE (g:ClaimGroup {claim_group_id: row.group_id})
                MERGE (a)-[:IN_CLAIM_GROUP]->(g)
                """,
                rows=groups,
            )
        families = [
            {
                "assertion_id": assertion.assertion_id,
                "family_id": assertion.evidence_family_id,
            }
            for assertion in result.assertions
            if assertion.evidence_family_id
        ]
        if families:
            await _run(
                tx,
                """
                UNWIND $rows AS row
                MATCH (a:Assertion {assertion_id: row.assertion_id})
                MERGE (f:EvidenceFamily {family_id: row.family_id})
                MERGE (a)-[:FROM_EVIDENCE_FAMILY]->(f)
                """,
                rows=families,
            )

    async def write_document(self, document: DocumentEnvelope) -> None:
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(self._write_document, document)
        logger.debug("Document %s written", document.document_version_id)

    async def read_document_identities(self) -> List[Dict[str, Any]]:
        """What a stored document's work keys are made of (graph.works)."""
        async with self._driver.session(database=self._database) as session:
            return [
                _data(record)
                for record in await _records(
                    session,
                    """
                    MATCH (d:Document)
                    OPTIONAL MATCH (d)-[:HAS_VERSION]->(v:DocumentVersion)
                    RETURN d.document_id AS document_id,
                           d.document_type AS document_type,
                           d.title AS title,
                           coalesce(d.external_ids, []) AS external_ids,
                           collect(v.metadata_json) AS metadata_json
                    """,
                )
            ]

    async def write_works(
        self, documents: List[Tuple[str, List[WorkKey]]]
    ) -> None:
        """Link stored documents to their works, a batch per transaction."""
        for start in range(0, len(documents), BATCH_SIZE):
            batch = documents[start : start + BATCH_SIZE]

            async def write(tx: Any) -> None:
                for document_id, keys in batch:
                    await GraphStore._write_work(tx, document_id, keys)

            async with self._driver.session(
                database=self._database
            ) as session:
                await session.execute_write(write)
            logger.info(
                "Linked works of %d/%d documents",
                start + len(batch),
                len(documents),
            )

    @staticmethod
    async def _write_work(
        tx: Any, document_id: str, keys: List[WorkKey]
    ) -> WorkPlan:
        """Link a document to its work (graph.works), folding stored works
        its keys show to be the same one.

        Two round trips: the key nodes are merged and the works they reach
        read in one statement (the keys' uniqueness constraint makes a
        concurrent writer of the same work wait for this transaction), the
        decision is written in another; a fold of stored works, rare, adds
        one per folded work.
        """
        reached = [
            _data(record)
            for record in await _records(
                tx,
                """
                UNWIND $rows AS row
                MERGE (k:WorkKey {key: row.key})
                ON CREATE SET k.scheme = row.scheme, k.strong = row.strong
                WITH collect(k) AS mine
                OPTIONAL MATCH (:Document {document_id: $document_id})
                               -[:MANIFESTATION_OF]->(c:Work)
                WITH mine, collect(c) AS currents
                WITH currents, reduce(hits = [], k IN mine |
                    hits + [(k)-[:IDENTIFIES]->(w:Work) |
                            {work: w, key: k.key}]) AS hits
                UNWIND hits + [c IN currents | {work: c, key: null}] AS hit
                WITH hit.work AS w, collect(hit.key) AS via,
                     hit.work IN currents AS current
                MATCH (key:WorkKey)-[:IDENTIFIES]->(w)
                RETURN w.work_id AS work_id, via, current,
                       collect({key: key.key, scheme: key.scheme,
                                strong: coalesce(key.strong, true)}) AS keys
                """,
                rows=[
                    {"key": key.key, "scheme": key.scheme, "strong": key.strong}
                    for key in keys
                ],
                document_id=document_id,
            )
        ]
        found: Dict[str, FoundWork] = {}
        for row in reached:
            work = found.setdefault(
                row["work_id"],
                FoundWork(
                    work_id=row["work_id"],
                    keys=[
                        WorkKey(
                            item["key"], item["scheme"], bool(item["strong"])
                        )
                        for item in row.get("keys") or []
                    ],
                ),
            )
            work.via |= {key for key in row.get("via") or [] if key}
            work.current = work.current or bool(row.get("current"))
        plan = plan_work(document_id, keys, list(found.values()))
        for absorbed in plan.absorbed:
            # The survivor takes the folded work's keys and documents.
            await _run(
                tx,
                """
                MATCH (a:Work {work_id: $absorbed})
                MERGE (w:Work {work_id: $work_id})
                WITH a, w
                OPTIONAL MATCH (k:WorkKey)-[r:IDENTIFIES]->(a)
                FOREACH (_ IN CASE WHEN k IS NULL THEN [] ELSE [1] END |
                    MERGE (k)-[:IDENTIFIES]->(w))
                DELETE r
                WITH DISTINCT a, w
                OPTIONAL MATCH (d:Document)-[m:MANIFESTATION_OF]->(a)
                FOREACH (_ IN CASE WHEN d IS NULL THEN [] ELSE [1] END |
                    MERGE (d)-[:MANIFESTATION_OF]->(w)
                    SET d.work_id = w.work_id)
                DELETE m
                WITH DISTINCT a, w
                SET w.absorbed_ids = coalesce(w.absorbed_ids, [])
                    + coalesce(a.absorbed_ids, []) + a.work_id
                DETACH DELETE a
                """,
                absorbed=absorbed,
                work_id=plan.work_id,
            )
        await _run(
            tx,
            """
            MERGE (w:Work {work_id: $work_id})
            WITH w
            // A strong key names one work; a title may name several, and
            // then it joins none of them (graph.works.plan_work).
            OPTIONAL MATCH (k:WorkKey)
            WHERE k.key IN $keys
              AND (NOT coalesce(k.strong, true)
                   OR NOT (k)-[:IDENTIFIES]->(:Work))
            FOREACH (_ IN CASE WHEN k IS NULL THEN [] ELSE [1] END |
                MERGE (k)-[:IDENTIFIES]->(w))
            WITH DISTINCT w
            MATCH (d:Document {document_id: $document_id})
            OPTIONAL MATCH (d)-[old:MANIFESTATION_OF]->(other:Work)
            WHERE other.work_id <> w.work_id
            DELETE old
            WITH DISTINCT d, w
            MERGE (d)-[:MANIFESTATION_OF]->(w)
            SET d.work_id = w.work_id
            """,
            work_id=plan.work_id,
            keys=[key.key for key in plan.keys],
            document_id=document_id,
        )
        if plan.rejected:
            logger.info(
                "Document %s shares a title with works %s that another "
                "identifier contradicts; kept apart",
                document_id,
                plan.rejected,
            )
        return plan

    @staticmethod
    async def _write_document(
        tx: Any,
        document: DocumentEnvelope,
        publishing: bool = False,
        kept_chunk_ids: Optional[set] = None,
    ) -> None:
        # Mutable metadata overwritten on a content-identical version needs
        # the current observation, never the earliest collection timestamp.
        observed = (
            document.retrieved_at or datetime.now(timezone.utc).isoformat()
        )
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
                v.document_published_at = $published_at,
                v.document_type = $document_type,
                v.source_family = $source.source_family,
                v.reliability_tier = $source.reliability_tier,
                // The first retrieval gates as-known visibility; a later
                // download must not move the version out of past snapshots.
                // Legacy graphs kept only the latest one as retrieved_at.
                v.first_retrieved_at = CASE
                    WHEN coalesce(v.first_retrieved_at, v.retrieved_at)
                        IS NULL
                        OR $retrieved_at
                            < coalesce(v.first_retrieved_at, v.retrieved_at)
                    THEN $retrieved_at
                    ELSE coalesce(v.first_retrieved_at, v.retrieved_at) END,
                v.last_retrieved_at = CASE
                    WHEN coalesce(v.last_retrieved_at, v.retrieved_at)
                        IS NULL
                        OR $retrieved_at
                            > coalesce(v.last_retrieved_at, v.retrieved_at)
                    THEN $retrieved_at
                    ELSE coalesce(v.last_retrieved_at, v.retrieved_at) END,
                // Latest observation for readers of the current state; the
                // dated history lives in MetricsObservation nodes.
                v.metrics_json = CASE
                    WHEN $metrics_observed_at IS NULL THEN $metrics_json
                    WHEN v.metrics_observed_at IS NULL
                        OR $metrics_observed_at >= v.metrics_observed_at
                    THEN $metrics_json
                    ELSE v.metrics_json END,
                v.metrics_observed_at = CASE
                    WHEN $metrics_observed_at IS NULL THEN NULL
                    WHEN v.metrics_observed_at IS NULL
                        OR $metrics_observed_at > v.metrics_observed_at
                    THEN $metrics_observed_at
                    ELSE v.metrics_observed_at END,
                v.country_codes = $country_codes, v.company_ids = $company_ids,
                v.university_ids = $university_ids, v.domain_ids = $domain_ids,
                v.contributor_ids = $contributor_ids,
                v.organization_ids = $organization_ids,
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
            retrieved_at=observed,
            metrics_observed_at=(document.metrics_observed_at or observed)
            if document.metrics
            else None,
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
            # Document-level party edges are replaced on every update; the
            # version keeps its own parties for point-in-time features.
            contributor_ids=list(
                dict.fromkeys(
                    item.contributor_id for item in document.contributors
                )
            ),
            organization_ids=list(
                dict.fromkeys(
                    item.organization_id for item in document.organizations
                )
            ),
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
        await GraphStore._write_metrics(tx, document, observed)
        await GraphStore._write_work(
            tx, document.document_id, document_work_keys(document)
        )

        await GraphStore._write_parties(tx, document)
        await GraphStore._write_economic_facts(tx, document)

        # Only chunks something stands on become nodes (chunk_nodes);
        # a document without extraction keeps its text in the snapshot.
        kept = (
            evidence_chunk_ids(None)
            if kept_chunk_ids is None
            else kept_chunk_ids
        )
        chunks = [
            chunk
            for chunk in document.chunks
            if kept is None or chunk.chunk_id in kept
        ]
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
                for chunk in chunks
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

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if publishing:
            # A new run replaces the chunk set: evidence on chunks that leave
            # the version must go with them, not outlive the run that made
            # it (extraction cleanup only sees the chunks still linked).
            await _run(
                tx,
                """
                MATCH (v:DocumentVersion {document_version_id: $version_id})
                      -[:HAS_CHUNK]->(c:Chunk)
                WHERE NOT c.chunk_id IN $chunk_ids
                MATCH (c)-[r:MENTIONS]->()
                DELETE r
                """,
                version_id=document.document_version_id,
                chunk_ids=chunk_ids,
            )
            await _run(
                tx,
                """
                MATCH (v:DocumentVersion {document_version_id: $version_id})
                      -[:HAS_CHUNK]->(c:Chunk)
                WHERE NOT c.chunk_id IN $chunk_ids
                MATCH ()-[r:HAS_ECONOMIC_EVIDENCE|HAS_MATURITY_EVIDENCE]->(c)
                DELETE r
                """,
                version_id=document.document_version_id,
                chunk_ids=chunk_ids,
            )
        # An import without extraction (e.g. without the PDF this time)
        # must not hide the chunks a published run's evidence stands on.
        await _run(
            tx,
            """
            MATCH (v:DocumentVersion {document_version_id: $version_id})
            WHERE $publishing OR NOT EXISTS {
                MATCH (v)<-[:PROCESSED]-(run:ProcessingRun)
                WHERE run.published = true OR run.status = 'succeeded'
            }
            MATCH (v)-[active:HAS_CHUNK]->(c:Chunk)
            WHERE NOT c.chunk_id IN $chunk_ids
            DELETE active
            """,
            version_id=document.document_version_id,
            chunk_ids=chunk_ids,
            publishing=publishing,
        )

    async def ensure_vector_indexes(self, result: ExtractionResult) -> None:
        """Create per-label vector indexes once the dimension is known.

        Schema changes cannot share the write transaction, and a Neo4j
        without vector support must not block publication.
        """
        vectors = [
            *result.concept_embeddings.values(),
            *result.chunk_embeddings.values(),
        ]
        if not vectors:
            return
        dimensions = len(vectors[0])
        kinds = {
            concept.concept_id: concept.kind.value
            for concept in result.concepts
        }
        labels = {
            kinds[concept_id]
            for concept_id in result.concept_embeddings
            if concept_id in kinds
        }
        if result.chunk_embeddings:
            # Evidence chunks: semantic retrieval of related context.
            labels.add("Chunk")
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
                "WHERE (r.status = 'succeeded' OR r.published = true) "
                # Publishing the same run again (a resumed crawl) is not
                # "something better is already active" (D-3).
                "AND r.run_id <> $run_id "
                "RETURN count(v) AS count, count(r) AS published"
            ),
            version_id=document.document_version_id,
            run_id=result.run.run_id,
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
            await resolve(
                GraphStore._write_document(
                    tx,
                    document,
                    publishing=publish,
                    kept_chunk_ids=evidence_chunk_ids(result),
                )
            )
        await resolve(
            GraphStore._write_extraction(tx, document, result, publish)
        )

    _TEXT_ONLY_CHUNKS = """
        MATCH (c:Chunk)
        WHERE NOT EXISTS { (c)-[:MENTIONS]->() }
          AND NOT EXISTS {
              (c)<-[:SUPPORTED_BY|HAS_MATURITY_EVIDENCE
                    |HAS_ECONOMIC_EVIDENCE]-()
          }
          AND c.embedding IS NULL
    """

    async def prune_text_chunks(
        self, apply: bool = False, batch: int = 1000
    ) -> Dict[str, int]:
        """Chunk nodes nothing stands on (graph.json chunk_nodes).

        Their text stays in the raw snapshots; each run first records the
        chunks it read (``input_chunk_ids``), so dropping USED_CHUNK loses
        no audit. Without ``apply`` only counts.
        """
        async with self._driver.session(database=self._database) as session:
            found = await _records(
                session,
                self._TEXT_ONLY_CHUNKS
                + "RETURN count(c) AS chunks, "
                "sum(size(coalesce(c.text, ''))) AS chars",
            )
            summary = {
                "chunks": int(found[0]["chunks"] or 0) if found else 0,
                "chars": int(found[0]["chars"] or 0) if found else 0,
                "deleted": 0,
                "warning": (
                    "Irreversible: the text of these chunks then exists "
                    "only in the raw snapshots (artifacts/raw of the "
                    "machine that ingested them). Back up Neo4j first."
                ),
            }
            if not apply or not summary["chunks"]:
                return summary

            async def record_inputs(tx):
                await _run(
                    tx,
                    """
                    MATCH (r:ProcessingRun) WHERE r.input_chunk_ids IS NULL
                    OPTIONAL MATCH (r)-[:USED_CHUNK]->(c:Chunk)
                    WITH r, collect(c.chunk_id) AS ids
                    SET r.input_chunk_ids = ids
                    """,
                )

            await session.execute_write(record_inputs)

            # The candidates are found once; each batch then deletes by
            # id instead of scanning every chunk again.
            ids = [
                record["chunk_id"]
                for record in await _records(
                    session,
                    self._TEXT_ONLY_CHUNKS + "RETURN c.chunk_id AS chunk_id",
                )
            ]

            async def delete(tx, chunk_ids):
                row = await _single(
                    tx,
                    "UNWIND $ids AS id MATCH (c:Chunk {chunk_id: id}) "
                    "DETACH DELETE c RETURN count(*) AS deleted",
                    ids=chunk_ids,
                )
                return int(row["deleted"]) if row else 0

            for start in range(0, len(ids), batch):
                summary["deleted"] += await session.execute_write(
                    delete, ids[start : start + batch]
                )
        logger.info("Text-only chunks pruned: %s", summary)
        return summary

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
                // Runs of the removed GLiNER extractor never close a version
                // for LLM processing (D-4).
                WHERE r.status = 'succeeded'
                  AND NOT r.parser IN ['metadata', 'gliner']
                RETURN DISTINCT id
                """,
                ids=list(version_ids),
            )
        return {record["id"] for record in records}

    async def processed_inputs(
        self, version_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """What each complete extraction of these versions read.

        The version does not change when a full text becomes available, so
        "already processed" also compares coverage and the PDF hash.
        """
        from ..ingest.processed import run_input

        if not version_ids:
            return {}
        async with self._driver.session(database=self._database) as session:
            records = await _records(
                session,
                """
                UNWIND $ids AS id
                MATCH (v:DocumentVersion {document_version_id: id})
                      <-[:PROCESSED]-(r:ProcessingRun)
                // Runs of the removed GLiNER extractor never close a version
                // for LLM processing (D-4).
                WHERE r.status = 'succeeded'
                  AND NOT r.parser IN ['metadata', 'gliner']
                RETURN id, r.metadata_json AS metadata_json
                """,
                ids=list(version_ids),
            )
        found: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            row = _data(record)
            found.setdefault(row["id"], []).append(
                run_input(row.get("metadata_json"))
            )
        return found

    async def read_concepts(self) -> List[Concept]:
        # Metadata nodes share some labels (Organization, Country, Domain)
        # but carry no concept_id; UNION removes multi-label duplicates.
        # A merged concept lives on in its MERGED_INTO target.
        query = "\nUNION\n".join(
            f"MATCH (c:{cypher_identifier(label)}) "
            "WHERE c.concept_id IS NOT NULL AND c.kind IS NOT NULL "
            "AND coalesce(c.status, '') <> 'merged' "
            # Only the fields the registry reads: properties(c) also sent
            # every embedding (~0.5 GB at 20k concepts x 2560) (D-6).
            "RETURN c {.concept_id, .kind, .preferred_label, .definition, "
            ".language, .status, .names_json, .aliases, .identity_key, "
            ".label_counts_json, .kind_counts_json, "
            ".technology_profile_json} AS properties"
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

    @staticmethod
    async def _write_embeddings(
        tx: Any,
        embedded: Dict[str, List[Dict[str, Any]]],
        model: str | None,
        recorded_at: str,
    ) -> None:
        """Store concept vectors, grouped by node label, with the text
        each was computed from (resolver.concept_text)."""
        for label, rows in embedded.items():
            for batch in _batches(rows):
                await _run(
                    tx,
                    f"""
                    UNWIND $rows AS row
                    MATCH (c:{cypher_identifier(label)}
                           {{concept_id: row.concept_id}})
                    // The same vector of the same model keeps the date on
                    // which past snapshots already saw it.
                    WITH c, row,
                         c.embedding = row.vector
                         AND c.embedding_model = $model
                         AND c.embedding_observed_at IS NOT NULL
                         AS unchanged
                    SET c.embedding_observed_at = CASE WHEN unchanged
                            THEN c.embedding_observed_at
                            ELSE $recorded_at END,
                        c.embedding = row.vector,
                        c.embedding_model = $model,
                        c.embedding_text = row.text
                    """,
                    rows=batch,
                    model=model,
                    recorded_at=recorded_at,
                )

    async def read_label_vectors(
        self, kinds: List[str], model: str
    ) -> List[Tuple[str, List[float]]]:
        """Stored concept vectors of ``model`` for the semantic cache, keyed
        by the text each was computed from."""
        query = "\nUNION\n".join(
            f"MATCH (c:{cypher_identifier(label)}) "
            "WHERE c.embedding IS NOT NULL AND c.embedding_model = $model "
            "AND c.preferred_label IS NOT NULL "
            "RETURN coalesce(c.embedding_text, c.preferred_label) AS label, "
            "c.embedding AS vector"
            for label in kinds
        )
        async with self._driver.session(database=self._database) as session:
            records = await _records(session, query, model=model)
        return [
            (record["label"], list(record["vector"])) for record in records
        ]

    async def read_concepts_to_embed(
        self, kinds: List[str], model: str, force: bool = False
    ) -> List[Dict[str, Any]]:
        """Concepts of ``kinds`` without a current vector of ``model``: none,
        another model's, or one computed before the concept got its
        definition (resolver.concept_text)."""
        query = "\nUNION\n".join(
            f"MATCH (c:{cypher_identifier(label)}) "
            f"WHERE c.concept_id IS NOT NULL AND c.kind = $kinds[{index}] "
            "AND coalesce(c.status, '') <> 'merged' "
            "AND ($force OR c.embedding IS NULL "
            "     OR coalesce(c.embedding_model, '') <> $model "
            "     OR coalesce(c.embedding_text, c.preferred_label) <> "
            "        CASE WHEN trim(coalesce(c.definition, '')) = '' "
            "             THEN c.preferred_label "
            "             ELSE c.preferred_label + ': ' + c.definition END) "
            "RETURN c.concept_id AS concept_id, c.kind AS node_label, "
            "c.preferred_label AS label, c.definition AS definition"
            for index, label in enumerate(kinds)
        )
        async with self._driver.session(database=self._database) as session:
            records = await _records(
                session, query, kinds=list(kinds), model=model, force=force
            )
        unique = {row["concept_id"]: row for row in map(_data, records)}
        return sorted(unique.values(), key=lambda row: row["concept_id"])

    async def write_concept_embeddings(
        self, rows: List[Dict[str, Any]], model: str
    ) -> None:
        """Backfill vectors for concepts stored without them."""
        embedded: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            embedded.setdefault(row["node_label"], []).append(
                {
                    "concept_id": row["concept_id"],
                    "vector": row["vector"],
                    "text": row.get("text"),
                }
            )
        recorded_at = datetime.now(timezone.utc).isoformat()
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                self._write_embeddings, embedded, model, recorded_at
            )

    async def set_concept_kind(
        self, concept_id: str, kind: str
    ) -> Dict[str, Any]:
        """A reviewed kind: relabel the concept within its kind family and
        mark it accepted, so later mentions' votes do not relabel it back
        (resolver._observe and the key migration keep reviewed kinds)."""
        from .merge import read_concepts_by_id

        found = await read_concepts_by_id(self, [concept_id])
        if concept_id not in found:
            raise ValueError(f"concept {concept_id} not found")
        concept, status = found[concept_id]
        if status == "merged":
            raise ValueError(f"concept {concept_id} is merged")
        target = ConceptKind(kind)
        if kind_family(target) != kind_family(concept.kind):
            raise ValueError(
                f"{concept.kind.value} and {target.value} are not one "
                "identity family"
            )
        source, label = (
            cypher_identifier(concept.kind.value),
            cypher_identifier(target.value),
        )

        async def write(tx: Any) -> None:
            await _run(
                tx,
                f"""
                MATCH (c:{source} {{concept_id: $id}})
                REMOVE c:{source}
                SET c:{label}, c.kind = $kind, c.status = 'accepted',
                    c.kind_reviewed_at = $now
                """,
                id=concept_id,
                kind=target.value,
                now=datetime.now(timezone.utc).isoformat(),
            )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(write)
        return {
            "concept_id": concept_id,
            "label": concept.preferred_label,
            "from": concept.kind.value,
            "to": target.value,
        }

    async def read_merge_candidates(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Pending POSSIBLY_SAME_AS pairs with what the graph knows of both
        concepts: definition, names, mentions, domains, parents
        (graph.review)."""
        concept = """{{
            concept_id: {c}.concept_id, label: {c}.preferred_label,
            kind: {c}.kind, status: {c}.status,
            definition: {c}.definition, aliases: {c}.aliases,
            mentions: size([({c})<-[m:MENTIONS]-(:Chunk)
                WHERE coalesce(m.resolution_status, '') <> 'ambiguous'
                | 1]),
            domains: [({c})-[:BELONGS_TO_DOMAIN]->(d)
                | coalesce(d.preferred_label, d.name)],
            parents: [({c})-[:SUBTECHNOLOGY_OF]->(p)
                | coalesce(p.preferred_label, p.name)]
        }}"""
        query = f"""
            MATCH (a)-[r:POSSIBLY_SAME_AS]->(b)
            WHERE coalesce(r.review_status, 'pending') = 'pending'
              AND coalesce(a.status, '') <> 'merged'
              AND coalesce(b.status, '') <> 'merged'
            RETURN {concept.format(c="a")} AS source,
                   {concept.format(c="b")} AS target,
                   r.method AS method, r.score AS score,
                   r.cosine AS cosine, r.alias AS alias
            ORDER BY coalesce(r.score, 0) DESC, a.concept_id, b.concept_id
            {"LIMIT $limit" if limit else ""}
        """
        async with self._driver.session(database=self._database) as session:
            return [
                _data(record)
                for record in await _records(session, query, limit=limit)
            ]

    async def merge_concepts(
        self, source_id: str, target_id: str, reason: str | None = None
    ) -> Dict[str, Any]:
        """Merge a duplicate concept into another (see graph.merge)."""
        from .merge import merge_concepts

        return await merge_concepts(self, source_id, target_id, reason)

    async def read_concept_forms(
        self,
    ) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
        """Resolved mentions per concept and per canonical form."""
        query = "\nUNION ALL\n".join(
            f"MATCH (c:{cypher_identifier(label)})<-[m:MENTIONS]-(:Chunk) "
            "WHERE c.concept_id IS NOT NULL "
            "AND coalesce(m.resolution_status, '') <> 'ambiguous' "
            "RETURN c.concept_id AS concept_id, "
            "coalesce(m.canonical_text, m.surface_text) AS form, "
            "count(m) AS mentions"
            for label in CONCEPT_LABELS
        )
        mentions: Dict[str, int] = {}
        forms: Dict[str, Dict[str, int]] = {}
        async with self._driver.session(database=self._database) as session:
            for record in await _records(session, query):
                row = _data(record)
                if row.get("form") is None:
                    continue
                concept_id, count = row["concept_id"], int(row["mentions"])
                mentions[concept_id] = mentions.get(concept_id, 0) + count
                counts = forms.setdefault(concept_id, {})
                counts[row["form"]] = counts.get(row["form"], 0) + count
        return mentions, forms

    async def read_concept_kinds(self) -> Dict[str, Dict[str, int]]:
        """Resolved mentions per concept and per kind the model reported."""
        query = "\nUNION ALL\n".join(
            f"MATCH (c:{cypher_identifier(label)})<-[m:MENTIONS]-(:Chunk) "
            "WHERE c.concept_id IS NOT NULL "
            "AND coalesce(m.resolution_status, '') <> 'ambiguous' "
            "AND size(coalesce(m.type_candidates, [])) > 0 "
            "RETURN c.concept_id AS concept_id, "
            "m.type_candidates[0] AS kind, count(m) AS mentions"
            for label in CONCEPT_LABELS
        )
        kinds: Dict[str, Dict[str, int]] = {}
        async with self._driver.session(database=self._database) as session:
            for record in await _records(session, query):
                row = _data(record)
                if row.get("kind") is None:
                    continue
                counts = kinds.setdefault(row["concept_id"], {})
                counts[row["kind"]] = counts.get(row["kind"], 0) + int(
                    row["mentions"]
                )
        return kinds

    async def write_concept_identities(self, updates: List[Any]) -> None:
        """Store key v2 identity keys, form and kind counts, preferred
        labels, cleaned names and settled kinds."""
        rows: Dict[str, List[Dict[str, Any]]] = {}
        moves: Dict[Tuple[str, str], List[str]] = {}
        for update in updates:
            names = getattr(update, "names", None)
            kind_counts = getattr(update, "kind_counts", None)
            rows.setdefault(cypher_identifier(update.kind), []).append(
                {
                    "concept_id": update.concept_id,
                    "identity_key": update.identity_key,
                    "label_counts_json": json_value(update.label_counts),
                    "kind_counts_json": json_value(kind_counts)
                    if kind_counts
                    else None,
                    "preferred_label": update.preferred_label,
                    "names_json": json_value(
                        [name.model_dump() for name in names]
                    )
                    if names is not None
                    else None,
                }
            )
            new_kind = getattr(update, "new_kind", None)
            if new_kind:
                moves.setdefault(
                    (
                        cypher_identifier(update.kind),
                        cypher_identifier(new_kind),
                    ),
                    [],
                ).append(update.concept_id)

        async def write(tx: Any) -> None:
            for label, items in rows.items():
                for batch in _batches(items):
                    await _run(
                        tx,
                        f"""
                        UNWIND $rows AS row
                        MATCH (c:{label} {{concept_id: row.concept_id}})
                        SET c.identity_key = row.identity_key,
                            c.key_version = $key_version,
                            c.label_counts_json = row.label_counts_json,
                            c.kind_counts_json = coalesce(
                                row.kind_counts_json, c.kind_counts_json),
                            c.names_json = coalesce(
                                row.names_json, c.names_json),
                            c.preferred_label = row.preferred_label,
                            c.name = row.preferred_label
                        """,
                        rows=batch,
                        key_version=KEY_VERSION,
                    )
            for (source, target), ids in sorted(moves.items()):
                for batch in _batches(ids):
                    await _run(
                        tx,
                        f"""
                        UNWIND $ids AS id
                        MATCH (c:{source} {{concept_id: id}})
                        REMOVE c:{source}
                        SET c:{target}, c.kind = $kind
                        """,
                        ids=batch,
                        kind=target,
                    )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(write)

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

    async def write_crawl_run(self, run: Dict[str, Any]) -> None:
        """Record the bounds of one crawl: source, query, period, records
        seen and checkpoint. Feature building separates "the source was
        searched and has nothing" from "the source was never searched".
        """
        async with self._driver.session(database=self._database) as session:
            await _run(
                session,
                """
                MERGE (s:Source {source_id: $source_id})
                ON CREATE SET s.source_family = $source_family
                MERGE (r:CrawlRun {crawl_id: $crawl_id})
                SET r.source_id = $source_id,
                    r.source_family = $source_family,
                    r.query = $query, r.filter = $filter,
                    r.period_start = $period_start,
                    r.period_end = $period_end,
                    r.records_seen = $records_seen,
                    r.records_ingested = $records_ingested,
                    r.failures = $failures,
                    r.search_failures = $search_failures,
                    r.checkpoint_json = $checkpoint_json,
                    r.started_at = $started_at,
                    r.finished_at = $finished_at,
                    r.status = $status, r.exhaustive = $exhaustive,
                    r.technology_ids = $technology_ids,
                    r.observed_at = $observed_at,
                    r.retrieved_at = $retrieved_at
                MERGE (r)-[:SEARCHED]->(s)
                """,
                **{
                    "query": None,
                    "filter": None,
                    "period_start": None,
                    "period_end": None,
                    "records_seen": 0,
                    "records_ingested": 0,
                    "failures": 0,
                    "search_failures": 0,
                    "checkpoint_json": None,
                    "finished_at": None,
                    "status": "running",
                    "exhaustive": False,
                    "technology_ids": [],
                    "observed_at": None,
                    "retrieved_at": None,
                    **run,
                },
            )

    async def read_temporal_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Everything the temporal dataset needs, with its dates.

        Rows are per document version (not per document), per mention day,
        per dated relation and per assertion, so a snapshot can select only
        what was published and observed by its date. The current Document
        node supplies identity only; its mutable parties and metrics cannot
        reconstruct historical version properties.
        """
        versions = """
            MATCH (d:Document)-[:HAS_VERSION]->(v:DocumentVersion)
                  -[:FROM_SOURCE]->(s:Source)
            WITH d, v, s,
                 [(v)-[:HAS_METRICS]->(obs:MetricsObservation)
                  | {observed_at: obs.observed_at,
                     metrics_json: obs.metrics_json}] AS metric_observations
            OPTIONAL MATCH (v)<-[:PROCESSED]-(run:ProcessingRun)
            WHERE run.published = true OR run.status = 'succeeded'
            RETURN d.document_id AS document_id,
                   // Copies of one work in several sources count once.
                   coalesce(d.work_id, d.document_id) AS work_id,
                   coalesce(v.document_type, d.document_type) AS document_type,
                   coalesce(v.document_published_at, v.version_published_at)
                       AS document_published_at,
                   v.document_version_id AS version_id,
                   v.version_published_at AS version_published_at,
                   coalesce(v.first_retrieved_at, v.retrieved_at)
                       AS retrieved_at,
                   coalesce(v.last_retrieved_at, v.retrieved_at)
                       AS last_retrieved_at,
                   v.metrics_observed_at AS metrics_observed_at,
                   v.metrics_json AS metrics_json,
                   metric_observations,
                   v.metadata_json AS metadata_json,
                   v.economic_facts_json AS economic_facts_json,
                   v.coverage AS coverage,
                   v.quality_status AS quality_status,
                   coalesce(v.country_codes, []) AS countries,
                   coalesce(v.company_ids, []) AS companies,
                   coalesce(v.university_ids, []) AS universities,
                   coalesce(v.domain_ids, []) AS domains,
                   coalesce(v.contributor_ids, []) AS contributors,
                   coalesce(v.organization_ids, []) AS organizations,
                   s.source_id AS source_id,
                   d.title AS title, d.canonical_url AS url,
                   v.source_family AS source_family,
                   v.independence_group AS independence_group,
                   v.reliability_tier AS reliability_tier,
                   count(run) > 0 AS extracted,
                   min(run.started_at) AS extracted_at
        """
        mentions = """
            MATCH (t:Technology)<-[m:MENTIONS]-(c:Chunk)
                  <-[:HAS_CHUNK]-(v:DocumentVersion)
            WHERE coalesce(m.resolution_status, '') <> 'ambiguous'
            OPTIONAL MATCH (run:ProcessingRun {run_id: m.run_id})
            RETURN t.concept_id AS technology_id,
                   v.document_version_id AS version_id,
                   substring(coalesce(m.observed_at, ''), 0, 10)
                       AS observed_at,
                   coalesce(m.recorded_at, run.started_at) AS recorded_at,
                   count(m) AS mentions,
                   sum(coalesce(m.confidence, 0.0)) AS confidence_sum,
                   count(m.confidence) AS confidence_count,
                   sum(CASE WHEN m.resolution_status = 'accepted'
                       THEN 1 ELSE 0 END) AS accepted,
                   sum(CASE WHEN m.resolution_status = 'provisional'
                       THEN 1 ELSE 0 END) AS provisional,
                   sum(CASE WHEN m.method = $semantic
                       THEN 1 ELSE 0 END) AS ambiguous,
                   collect(DISTINCT c.content_hash) AS content_hashes
        """
        technologies = """
            MATCH (t:Technology)
            WHERE t.concept_id IS NOT NULL
              AND coalesce(t.status, '') <> 'merged'
            RETURN t.concept_id AS technology_id,
                   t.preferred_label AS technology,
                   t.kind AS kind, t.definition AS definition,
                   t.first_seen_at AS first_seen_at,
                   t.status AS status, t.embedding AS embedding,
                   t.embedding_model AS embedding_model,
                   t.embedding_observed_at AS embedding_observed_at
        """
        relations = """
            MATCH (t:Technology)-[r:SOLVES|DEVELOPED_BY|USED_BY|FUNDED_BY
                                 |DEVELOPED_IN|SUBTECHNOLOGY_OF]->(x)
            OPTIONAL MATCH (run:ProcessingRun {run_id: r.run_id})
            RETURN t.concept_id AS technology_id, type(r) AS relation,
                   coalesce(x.concept_id, x.organization_id, x.domain_id)
                       AS target_id,
                   coalesce(x.preferred_label, x.name) AS target_label,
                   x.kind AS target_kind,
                   labels(x) AS target_labels,
                   r.document_version_id AS version_id,
                   r.observed_at AS observed_at,
                   coalesce(r.recorded_at, run.started_at) AS recorded_at
        """
        maturity = """
            MATCH (t:Technology)-[r:HAS_MATURITY_EVIDENCE]->(c:Chunk)
            OPTIONAL MATCH (run:ProcessingRun {run_id: r.run_id})
                  -[:PROCESSED]->(v:DocumentVersion)
            RETURN t.concept_id AS technology_id, r.stage AS stage,
                   r.stage_rank AS stage_rank, r.trl AS trl,
                   r.observed_at AS observed_at,
                   coalesce(r.recorded_at, run.started_at) AS recorded_at,
                   head(collect(v.document_version_id)) AS version_id
        """
        economics = """
            MATCH (t:Technology)-[r:HAS_ECONOMIC_EVIDENCE]->(c:Chunk)
            OPTIONAL MATCH (run:ProcessingRun {run_id: r.run_id})
                  -[:PROCESSED]->(v:DocumentVersion)
                  -[:FROM_SOURCE]->(s:Source)
            RETURN t.concept_id AS technology_id,
                   r.evidence_id AS evidence_id, r.category AS category,
                   r.amount_value AS amount_value, r.currency AS currency,
                   r.amount_text AS amount_text, r.unit AS unit,
                   r.period AS period, r.assertion_id AS assertion_id,
                   r.confidence AS confidence, r.status AS status,
                   r.polarity AS polarity, r.modality AS modality,
                   r.observed_at AS observed_at,
                   coalesce(r.recorded_at, run.started_at) AS recorded_at,
                   head(collect(v.document_version_id)) AS version_id,
                   max(v.reliability_tier) AS reliability_tier
        """
        assertions = """
            MATCH (v:DocumentVersion)-[:HAS_ASSERTION]->(a:Assertion)
                  -[:SUBJECT]->(t:Technology)
            OPTIONAL MATCH (a)-[:IN_CLAIM_GROUP]->(g:ClaimGroup)
            OPTIONAL MATCH (a)-[:FROM_EVIDENCE_FAMILY]->(f:EvidenceFamily)
            OPTIONAL MATCH (run:ProcessingRun)-[:CREATED]->(a)
            OPTIONAL MATCH (a)-[e:SUPPORTED_BY]->(:Chunk)
            WITH v, a, t, g, f, max(run.started_at) AS run_recorded_at,
                 head(collect(e.quote)) AS quote
            RETURN t.concept_id AS technology_id,
                   a.assertion_id AS assertion_id, a.predicate AS predicate,
                   a.status AS status,
                   a.verification_status AS verification_status,
                   a.polarity AS polarity, a.modality AS modality,
                   a.evidence_kind AS evidence_kind,
                   a.extraction_confidence AS confidence,
                   a.observed_at AS observed_at,
                   coalesce(a.recorded_at, run_recorded_at) AS recorded_at,
                   v.document_version_id AS version_id,
                   g.claim_group_id AS claim_group_id,
                   f.family_id AS evidence_family_id,
                   a.qualifiers_json AS qualifiers_json,
                   // Roles are read, not the stored claim_key: a merge moves
                   // role edges, so the slot follows the concepts it names.
                   [(a)-[role]->(concept)
                    WHERE type(role) IN $role_types
                      AND concept.concept_id IS NOT NULL
                    | [type(role), concept.concept_id]] AS roles,
                   // Names for reports (signal cards); roles stay ids.
                   [(a)-[role]->(concept)
                    WHERE type(role) IN $role_types
                      AND concept.concept_id IS NOT NULL
                    | [type(role), concept.preferred_label, concept.kind]]
                       AS role_labels,
                   quote
        """
        organizations = """
            MATCH (o:Organization)
            WHERE o.organization_id IS NOT NULL AND o.name IS NOT NULL
            RETURN o.organization_id AS organization_id, o.name AS name
        """
        crawls = """
            MATCH (r:CrawlRun)
            RETURN r.crawl_id AS crawl_id, r.source_id AS source_id,
                   r.source_family AS source_family, r.query AS query,
                   r.period_start AS period_start,
                   r.period_end AS period_end,
                   r.records_seen AS records_seen,
                   r.started_at AS started_at, r.finished_at AS finished_at,
                   r.observed_at AS observed_at,
                   r.retrieved_at AS retrieved_at,
                   r.status AS status, r.failures AS failures,
                   r.search_failures AS search_failures,
                   r.exhaustive AS exhaustive,
                   r.technology_ids AS technology_ids
        """
        result: Dict[str, List[Dict[str, Any]]] = {}
        async with self._driver.session(database=self._database) as session:
            labels = {
                row["label"]
                for row in await _records(
                    session, "CALL db.labels() YIELD label RETURN label"
                )
            }
            for name, query, needed in (
                ("versions", versions, "DocumentVersion"),
                ("mentions", mentions, "Technology"),
                ("technologies", technologies, "Technology"),
                ("relations", relations, "Technology"),
                ("maturity", maturity, "Technology"),
                ("economics", economics, "Technology"),
                ("assertions", assertions, "Assertion"),
                ("crawls", crawls, "CrawlRun"),
                ("organizations", organizations, "Organization"),
            ):
                result[name] = (
                    [
                        _data(record)
                        for record in await _records(
                            session,
                            query,
                            semantic=SEMANTIC_CANDIDATE_METHOD,
                            role_types=list(
                                load_catalog("graph")["assertion_roles"].values()
                            ),
                        )
                    ]
                    if needed in labels
                    else []
                )
        logger.info(
            "Temporal data: %s",
            ", ".join(f"{name}={len(rows)}" for name, rows in result.items()),
        )
        return result

    async def read_signal_data(self) -> List[Dict[str, Any]]:
        """Dated per-technology signals projected from reviewed text claims:
        organizations, countries, taxonomy parents, maturity and economics.
        """
        query = """
            MATCH (t:Technology)-[r:DEVELOPED_BY|USED_BY|FUNDED_BY
                                 |DEVELOPED_IN|MANUFACTURED_IN|TESTED_IN
                                 |DEPLOYED_IN|BELONGS_TO_DOMAIN|TARGETS_MARKET
                                 |SUBTECHNOLOGY_OF]->(x)
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
            WHERE r.status = 'accepted' AND r.polarity = 'affirmed'
              AND r.modality IN ['reported', 'observed']
            RETURN t.concept_id AS technology_id, 'ECONOMIC' AS signal,
                   r.category AS target_id, r.amount_value AS value,
                   r.observed_at AS observed_at
        """
        async with self._driver.session(database=self._database) as session:
            return [_data(record) for record in await _records(session, query)]

    async def read_technology_labels(self) -> Dict[str, Dict[str, Any]]:
        """Model scores and LLM labels written by
        ``modeling.labeling.graph_labels``, by technology id."""
        query = """
            MATCH (t:Technology)
            WHERE coalesce(t.status, '') <> 'merged'
              AND (t.signal_probability IS NOT NULL
                   OR t.llm_verdict IS NOT NULL)
            RETURN t.concept_id AS technology_id,
                   t.signal_probability AS probability,
                   t.signal_flag AS flag, t.signal_model AS model,
                   t.signal_snapshot AS snapshot,
                   t.llm_verdict AS verdict,
                   t.llm_is_technology AS is_technology,
                   t.llm_score AS llm_score, t.llm_hype AS hype,
                   t.llm_maturity AS maturity,
                   t.llm_rationale AS rationale
        """
        async with self._driver.session(database=self._database) as session:
            return {
                row["technology_id"]: row
                for row in map(_data, await _records(session, query))
            }

    async def read_related_chunks(
        self,
        query: str,
        exclude_version_id: str,
        limit_documents: int,
        limit_chunks: int,
        max_chars: int,
        query_vector: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Whole original chunks from published successes.

        A literal full-text match comes first; with ``query_vector`` the
        evidence chunks nearest to the query (another language, a synonym)
        follow, best first. Related sources are context, not evidence for
        the current document. Oversized chunks are skipped rather than
        truncated or summarized.
        """
        query = str(query).strip().casefold()
        if not query or len(query) > 1000:
            return []
        limit_documents = min(20, max(0, int(limit_documents)))
        limit_chunks = min(100, max(0, int(limit_chunks)))
        max_chars = min(200000, max(0, int(max_chars)))
        if not limit_documents or not limit_chunks or not max_chars:
            return []
        # Full-text indexes find candidates (D-7: CONTAINS alone scanned
        # every chunk); the literal CONTAINS check keeps the old meaning.
        lexical = """
                CALL db.index.fulltext.queryNodes('chunk_text', $phrase)
                YIELD node
                MATCH (d:Document)-[:HAS_VERSION]->(v:DocumentVersion)
                      -[:HAS_CHUNK]->(node)
                RETURN node AS c, v, d, 2.0 AS score
                UNION
                CALL db.index.fulltext.queryNodes('document_title', $phrase)
                YIELD node
                MATCH (node)-[:HAS_VERSION]->(v:DocumentVersion)
                      -[:HAS_CHUNK]->(c:Chunk)
                RETURN c, v, node AS d, 2.0 AS score
        """
        # Vector hits carry a cosine < 2, so literal matches stay first.
        semantic = """
                UNION
                CALL db.index.vector.queryNodes(
                    'chunk_embedding', $vector_limit, $vector)
                YIELD node, score
                WITH node, score WHERE score >= $min_score
                MATCH (d:Document)-[:HAS_VERSION]->(v:DocumentVersion)
                      -[:HAS_CHUNK]->(node)
                RETURN node AS c, v, d, score
        """
        body = """
            MATCH (run:ProcessingRun)-[:PROCESSED]->(v)
            WHERE run.status = 'succeeded'
              AND (run.published = true OR run.published IS NULL)
              AND run.parser <> 'metadata'
              AND v.document_version_id <> $exclude_version_id
              AND c.parse_status = 'accepted'
              AND (score < 2.0
                   OR toLower(c.text) CONTAINS $search_text
                   OR toLower(d.title) CONTAINS $search_text)
            WITH c, v, d, max(score) AS score
            RETURN c.chunk_id AS chunk_id, c.text AS text,
                   c.kind AS kind, c.locator_json AS locator_json,
                   d.document_id AS document_id, d.title AS title,
                   v.document_version_id AS document_version_id,
                   c.order AS chunk_order, score
            ORDER BY score DESC, document_id, document_version_id,
                     chunk_order, chunk_id
            LIMIT $candidate_limit
        """
        settings = load_catalog("pipeline").get("graph_context", {})
        parameters = {
            "search_text": query,
            "phrase": _lucene_phrase(query),
            "exclude_version_id": exclude_version_id,
            "candidate_limit": min(2000, limit_documents * limit_chunks * 4),
            "vector": query_vector,
            "vector_limit": min(200, limit_documents * limit_chunks * 4),
            "min_score": float(settings.get("semantic_min_score", 0.75)),
        }
        async with self._driver.session(database=self._database) as session:
            records = None
            if query_vector:
                try:
                    records = await _records(
                        session,
                        f"CALL {{{lexical}{semantic}}}{body}",
                        **parameters,
                    )
                except Exception as exc:
                    # No vector index yet (no evidence embedded) or a Neo4j
                    # without vector search: the literal match still works.
                    logger.debug(
                        "Vector context search unavailable (%s)",
                        type(exc).__name__,
                    )
            if records is None:
                records = await _records(
                    session, f"CALL {{{lexical}}}{body}", **parameters
                )
        output, documents, seen = [], set(), set()
        used = 0
        for record in records:
            row = _data(record)
            text = row.get("text")
            identity = (row.get("document_version_id"), row.get("chunk_id"))
            document_id = row.get("document_id")
            if (
                not isinstance(text, str)
                or not text
                or identity in seen
                or identity[0] == exclude_version_id
                or not document_id
                or used + len(text) > max_chars
                or (
                    document_id not in documents
                    and len(documents) >= limit_documents
                )
            ):
                continue
            seen.add(identity)
            documents.add(document_id)
            used += len(text)
            locator = row.pop("locator_json", None)
            if locator:
                try:
                    row["locator"] = json.loads(locator)
                except (TypeError, ValueError):
                    pass
            row.pop("chunk_order", None)
            score = row.pop("score", None)
            if isinstance(score, (int, float)) and score < 2.0:
                row["similarity"] = round(float(score), 4)
            output.append(row)
            if len(output) >= limit_chunks:
                break
        return output

    async def read_taxonomy_input(
        self, kinds: List[str], snapshot: str | None = None
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        """Embedded concepts with their document dates, and reviewed
        (child, parent) SUBTECHNOLOGY_OF pairs.
        """
        concepts = """
            MATCH (c:__LABEL__)
            WHERE c.embedding IS NOT NULL AND c.concept_id IS NOT NULL
              AND coalesce(c.status, '') <> 'merged'
            OPTIONAL MATCH (c)<-[m:MENTIONS]-(chunk:Chunk)<-[:HAS_CHUNK]-
                  (v:DocumentVersion)<-[:HAS_VERSION]-(d:Document)
            WHERE coalesce(m.resolution_status, '') <> 'ambiguous'
            WITH c, collect(DISTINCT CASE WHEN d.document_id IS NOT NULL
                THEN {document_id: d.document_id,
                      date: coalesce(chunk.created_at, v.version_published_at,
                                     d.created_at)} END) AS document_evidence
            RETURN c.concept_id AS concept_id, c.preferred_label AS label,
                   c.kind AS kind, c.embedding AS embedding,
                   c.embedding_model AS embedding_model,
                   c.embedding_observed_at AS embedding_observed_at,
                   size(c.embedding) AS embedding_dimensions,
                   c.first_seen_at AS first_seen_at, document_evidence,
                   [item IN document_evidence | item.date] AS document_dates
        """
        parents = """
            MATCH (child)-[r:SUBTECHNOLOGY_OF]->(parent)
            WHERE $snapshot IS NULL OR (r.observed_at IS NOT NULL
                AND substring(toString(r.observed_at), 0, 10) <= $snapshot)
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
                for item in map(
                    _data,
                    await _records(
                        session,
                        parents,
                        snapshot=snapshot[:10] if snapshot else None,
                    ),
                )
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
    def _domain_identity_links(
        document: DocumentEnvelope, result: ExtractionResult
    ) -> List[Dict[str, Any]]:
        """Exact labels and unique curated aliases bridge existing domains.

        Case and whitespace normalization is safe here; stemming, punctuation
        removal and embedding proximity are deliberately not identity proof.
        Duplicate metadata labels and shared curated aliases remain separate.
        """

        def canonical(value: str) -> str:
            return " ".join(
                unicodedata.normalize("NFKC", value).casefold().split()
            )

        domains: Dict[str, set] = {}
        for domain in document.domains:
            domains.setdefault(canonical(domain.name), set()).add(
                domain.domain_id
            )
        curated: Dict[str, set] = {}
        for entry in load_catalog("sources").get("domains", []):
            name = canonical(entry["name"])
            for label in [entry["name"], *entry.get("aliases", [])]:
                curated.setdefault(canonical(label), set()).add(name)
        reviewed: Dict[str, List[str]] = {}
        for assertion in result.assertions:
            if GraphStore._reviewed(assertion):
                for concept_id in assertion.roles.values():
                    reviewed.setdefault(concept_id, []).append(
                        assertion.assertion_id
                    )
        links = []
        for concept in result.concepts:
            label = canonical(concept.preferred_label)
            direct_matches = domains.get(label, set())
            matches = set(direct_matches)
            canonical_names = curated.get(label, set())
            if len(canonical_names) == 1:
                matches.update(domains.get(next(iter(canonical_names)), set()))
            if (
                concept.kind == ConceptKind.DOMAIN
                and (
                    concept.status == "accepted"
                    or concept.concept_id in reviewed
                )
                and len(matches) == 1
            ):
                links.append(
                    {
                        "concept_id": concept.concept_id,
                        "domain_id": next(iter(matches)),
                        "assertion_ids": sorted(
                            set(reviewed.get(concept.concept_id, []))
                        ),
                    }
                )
                if not direct_matches:
                    links[-1]["method"] = "exact_curated_alias"
        return links

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
    async def _settle_family_kinds(
        tx: Any, result: ExtractionResult
    ) -> Tuple[ExtractionResult, Dict[str, str]]:
        """Write each concept of a kind family under one settled kind.

        An organization takes its highest kind: a stored node of a lower
        kind is relabeled, one of a higher kind keeps its label, so a stale
        registry copy cannot downgrade it. A technology, method or material
        takes the kind its mentions voted (Concept.kind_counts), up or
        down. Either way the stored node is relabeled, never duplicated.
        Also returns the stored label of every ambiguous candidate of the
        family.
        """
        candidates = {
            item["concept_id"]: item["kind"]
            for decision in result.resolutions
            if decision.method == AMBIGUOUS_COLLISION_METHOD
            for item in decision.candidates
            if item.get("kind") in FAMILY_RANK
        }
        ids = list(
            dict.fromkeys(
                [
                    *(
                        concept.concept_id
                        for concept in result.concepts
                        if concept.kind.value in FAMILY_RANK
                    ),
                    *candidates,
                ]
            )
        )
        if not ids:
            return result, {}
        rows = [
            _data(record)
            for record in await _records(tx, FAMILY_LABELS_QUERY, ids=ids)
        ]
        stored = {
            row["concept_id"]: [
                label for label in FAMILY_RANK if row.get(label)
            ]
            for row in rows
        }
        stored_votes = {
            row["concept_id"]: json.loads(row.get("kind_counts_json") or "{}")
            for row in rows
        }
        moves: Dict[Tuple[str, str], List[str]] = {}
        concepts = []
        for concept in result.concepts:
            labels = stored.get(concept.concept_id) or []
            if concept.kind.value not in FAMILY_RANK or not labels:
                concepts.append(concept)
                continue
            update: Dict[str, Any] = {}
            if kind_family(concept.kind) in VOTED_FAMILIES:
                # Votes another job stored are kept: per kind, the larger
                # count of the stored node and of this (maybe stale) copy.
                votes = stored_votes.get(concept.concept_id) or {
                    max(labels, key=FAMILY_RANK.get): 1
                }
                own = concept.kind_counts or {concept.kind.value: 1}
                votes = {
                    kind: max(votes.get(kind, 0), own.get(kind, 0))
                    for kind in {*votes, *own}
                }
                # A reviewed kind (set-concept-kind) is not re-voted.
                kind = (
                    concept.kind.value
                    if concept.status == "accepted"
                    else settled_kind(votes, concept.kind)
                )
                update["kind_counts"] = votes
            else:
                kind = max([concept.kind.value, *labels], key=FAMILY_RANK.get)
            if kind not in labels:
                source = max(labels, key=FAMILY_RANK.get)
                moves.setdefault((source, kind), []).append(concept.concept_id)
            if len(labels) > 1:
                logger.warning(
                    "Concept %s is stored under several kinds %s; merge them",
                    concept.concept_id,
                    labels,
                )
            concepts.append(
                concept.model_copy(
                    update={"kind": ConceptKind(kind), **update}
                )
            )
        for (source, target), rows in sorted(moves.items()):
            await _run(
                tx,
                f"""
                UNWIND $ids AS id
                MATCH (c:{source} {{concept_id: id}})
                REMOVE c:{source}
                SET c:{target}, c.kind = $kind
                """,
                ids=rows,
                kind=target,
            )
        candidate_labels = {
            concept_id: max(
                stored.get(concept_id) or [kind], key=FAMILY_RANK.get
            )
            for concept_id, kind in candidates.items()
        }
        return result.model_copy(
            update={"concepts": concepts}
        ), candidate_labels

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
                exclude={"run", "concept_embeddings", "chunk_embeddings"},
                mode="json",
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
                r.metadata_json = $metadata_json, r.trace_json = $trace_json,
                // Every chunk the run read, also those kept only as text.
                r.input_chunk_ids = $input_chunk_ids
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

        result, candidate_labels = await GraphStore._settle_family_kinds(
            tx, result
        )
        concept_rows: Dict[str, List[Dict[str, Any]]] = {}
        for concept in result.concepts:
            concept_rows.setdefault(
                cypher_identifier(concept.kind.value), []
            ).append(
                {
                    "aliases": list(
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
                    "normalized_aliases": list(
                        dict.fromkeys(
                            name.normalized_text
                            for name in concept.names
                            if name.status == "accepted"
                        )
                    ),
                    "names_json": json_value(
                        [name.model_dump() for name in concept.names]
                    ),
                    "key_version": (
                        KEY_VERSION if concept.identity_key else None
                    ),
                    "label_counts_json": json_value(concept.label_counts),
                    "kind_counts_json": json_value(concept.kind_counts),
                    "profile_json": (
                        json_value(concept.profile)
                        if concept.profile
                        else None
                    ),
                    **concept.model_dump(
                        exclude={"names", "label_counts", "kind_counts"},
                        mode="json",
                    ),
                }
            )
        for label, rows in concept_rows.items():
            for batch in _batches(rows):
                await _run(
                    tx,
                    f"""
                    UNWIND $rows AS row
                    MERGE (c:{label} {{concept_id: row.concept_id}})
                    SET c.kind = row.kind,
                        c.preferred_label = row.preferred_label,
                        c.name = row.preferred_label,
                        c.definition = coalesce(row.definition,
                                                c.definition),
                        c.language = row.language,
                        c.status = CASE WHEN c.status = 'merged'
                            THEN c.status ELSE row.status END,
                        c.aliases = row.aliases,
                        c.normalized_aliases = row.normalized_aliases,
                        c.names_json = row.names_json,
                        c.identity_key = row.identity_key,
                        c.key_version = row.key_version,
                        c.label_counts_json = row.label_counts_json,
                        c.kind_counts_json = row.kind_counts_json,
                        // Technology contract: a validated profile is
                        // never replaced by a proposed one.
                        c.technology_profile_json = CASE
                            WHEN row.profile_json IS NULL
                                OR (c.classification_status = 'validated'
                                    AND row.profile.classification_status
                                        <> 'validated')
                            THEN c.technology_profile_json
                            ELSE row.profile_json END,
                        c.classification_status = CASE
                            WHEN c.classification_status = 'validated'
                            THEN c.classification_status
                            ELSE coalesce(row.profile.classification_status,
                                          c.classification_status) END,
                        c.technical_mechanism = coalesce(
                            c.technical_mechanism,
                            row.profile.technical_mechanism),
                        c.technical_function = coalesce(
                            c.technical_function,
                            row.profile.technical_function),
                        c.technology_type = coalesce(
                            c.technology_type, row.profile.technology_type),
                        c.boundary = coalesce(
                            c.boundary, row.profile.boundary),
                        c.application_context = coalesce(
                            c.application_context,
                            row.profile.application_context),
                        c.first_seen_at = CASE
                            WHEN $observed_at IS NULL THEN c.first_seen_at
                            WHEN c.first_seen_at IS NULL
                                OR $observed_at < c.first_seen_at
                            THEN $observed_at
                            ELSE c.first_seen_at END
                    """,
                    rows=batch,
                    observed_at=_version_date(document),
                )

        # Every concept a mention, role or projection points to is in
        # result.concepts (validate_extraction), so matches can use a label
        # and its concept_id constraint instead of scanning all nodes.
        labels = {
            concept.concept_id: cypher_identifier(concept.kind.value)
            for concept in result.concepts
        }
        graph = load_catalog("graph")

        # An ISO-coded country from text is the same country as the
        # metadata-level one.
        same_countries = [
            {
                "concept_id": concept.concept_id,
                **_country_row(concept.preferred_label),
            }
            for concept in result.concepts
            if concept.kind == ConceptKind.COUNTRY
            and COUNTRY_CODE.fullmatch(concept.preferred_label)
        ]
        if same_countries:
            await _run(
                tx,
                """
                UNWIND $rows AS row
                MATCH (c:Country {concept_id: row.concept_id})
                SET c.name = row.name, c.name_en = row.name_en,
                    c.code = row.code
                MERGE (x:Country {country_id: row.country_id})
                ON CREATE SET x.code = row.code, x.name = row.name,
                    x.name_en = row.name_en
                MERGE (c)-[:SAME_AS]->(x)
                """,
                rows=same_countries,
            )

        # A company named in the text is the company of the metadata
        # (OpenAlex institution, patent applicant, GitHub owner).
        same_companies = [
            {
                "concept_id": concept.concept_id,
                "organization_id": organization_identity(
                    concept.preferred_label, "company", "", ""
                )[0],
            }
            for concept in result.concepts
            if concept.kind == ConceptKind.COMPANY
        ]
        if same_companies:
            await _run(
                tx,
                """
                UNWIND $rows AS row
                MATCH (c:Company {concept_id: row.concept_id})
                MATCH (x:Organization {organization_id: row.organization_id})
                MERGE (c)-[:SAME_AS]->(x)
                """,
                rows=same_companies,
            )

        domain_links = [
            {
                "method": link.get("method", "exact_canonical_label"),
                **{
                    key: value
                    for key, value in link.items()
                    if key != "method"
                },
            }
            for link in GraphStore._domain_identity_links(document, result)
        ]
        if domain_links:
            await _run(
                tx,
                """
                UNWIND $rows AS row
                MATCH (c:Domain {concept_id: row.concept_id})
                MATCH (x:Domain {domain_id: row.domain_id})
                MERGE (c)-[r:SAME_AS {document_version_id: $version_id}]->(x)
                SET r.status = 'accepted', r.method = row.method,
                    r.assertion_ids = row.assertion_ids,
                    r.observed_at = $observed_at
                """,
                version_id=document.document_version_id,
                observed_at=_version_date(document),
                rows=domain_links,
            )

        embedded: Dict[str, List[Dict[str, Any]]] = {}
        texts = {
            concept.concept_id: concept_text(concept)
            for concept in result.concepts
        }
        for concept_id, vector in result.concept_embeddings.items():
            if concept_id in labels:
                embedded.setdefault(labels[concept_id], []).append(
                    {
                        "concept_id": concept_id,
                        "vector": vector,
                        "text": texts.get(concept_id),
                    }
                )
        await GraphStore._write_embeddings(
            tx, embedded, result.embedding_model, run.started_at
        )
        for batch in _batches(
            [
                {"chunk_id": chunk_id, "vector": vector}
                for chunk_id, vector in result.chunk_embeddings.items()
            ]
        ):
            await _run(
                tx,
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {chunk_id: row.chunk_id})
                SET c.embedding = row.vector, c.embedding_model = $model
                """,
                rows=batch,
                model=result.embedding_model,
            )

        # A semantic match is a review candidate, never an identity; so is
        # an alias the source declared that already names another concept.
        candidates: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        kinds = {item.value for item in ConceptKind}
        for decision in result.resolutions:
            if (
                decision.method == AMBIGUOUS_COLLISION_METHOD
                or decision.concept_id not in labels
            ):
                continue
            for candidate in decision.candidates:
                kind = candidate.get("kind")
                method = candidate.get("method") or decision.method
                if kind not in kinds or method not in (
                    SEMANTIC_CANDIDATE_METHOD,
                    DECLARED_ALIAS_METHOD,
                ):
                    continue
                if candidate["concept_id"] == decision.concept_id:
                    continue
                candidates.setdefault(
                    (labels[decision.concept_id], cypher_identifier(kind)), []
                ).append(
                    {
                        "source": decision.concept_id,
                        "target": candidate["concept_id"],
                        "score": candidate.get("score"),
                        "cosine": candidate.get("cosine"),
                        "method": method,
                        "alias": candidate.get("alias"),
                    }
                )
        for (source_label, target_label), rows in candidates.items():
            await _run(
                tx,
                f"""
                UNWIND $rows AS row
                MATCH (a:{source_label} {{concept_id: row.source}})
                MATCH (b:{target_label} {{concept_id: row.target}})
                MERGE (a)-[r:POSSIBLY_SAME_AS]->(b)
                SET r.score = row.score, r.cosine = row.cosine,
                    r.method = row.method, r.alias = row.alias,
                    r.run_id = $run_id,
                    r.review_status = coalesce(r.review_status, 'pending')
                """,
                rows=rows,
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

        relationships = "|".join(
            sorted(
                {
                    cypher_identifier(projection["relationship"])
                    for projection in graph["projections"].values()
                }
            )
        )
        await _run(
            tx,
            f"""
            MATCH ()-[r:{relationships}]->()
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
            if decision is None:
                continue
            if decision.method == AMBIGUOUS_COLLISION_METHOD:
                # An identity-key collision links the mention to every
                # candidate; readers of mention counts skip these links.
                targets = [
                    (
                        item["concept_id"],
                        candidate_labels.get(item["concept_id"])
                        or cypher_identifier(item["kind"]),
                    )
                    for item in decision.candidates
                    if item.get("kind") in CONCEPT_LABELS
                ]
            elif decision.concept_id is not None and decision.status in (
                "accepted",
                "provisional",
            ):
                targets = [(decision.concept_id, labels[decision.concept_id])]
            else:
                continue
            for concept_id, label in targets:
                mention_rows.setdefault(label, []).append(
                    {
                        # The profile goes to the concept node.
                        **mention.model_dump(
                            mode="json", exclude={"profile"}
                        ),
                        "concept_id": concept_id,
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
                        r.observed_at = row.observed_at,
                        r.recorded_at = $recorded_at
                    """,
                    rows=batch,
                    run_id=run.run_id,
                    recorded_at=run.started_at,
                )

        projections: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for link in GraphStore._projection_links(document, result):
            projections.setdefault(
                (
                    cypher_identifier(link["source_label"]),
                    cypher_identifier(link["target_label"]),
                    cypher_identifier(link["relationship"]),
                ),
                [],
            ).append(
                {
                    key: link[key]
                    for key in (
                        "source",
                        "target",
                        "chunk_id",
                        "quote",
                        "start",
                        "end",
                        "assertion_id",
                    )
                }
                | {"observed_at": _chunk_date(document, link["chunk_id"])}
            )
        for (source, target, relationship), rows in projections.items():
            await _run(
                tx,
                f"""
                UNWIND $rows AS row
                MATCH (source:{source} {{concept_id: row.source}})
                MATCH (target:{target} {{concept_id: row.target}})
                MERGE (source)-[r:{relationship}
                      {{document_version_id: $version_id}}]->(target)
                SET r.chunk_id = row.chunk_id, r.quote = row.quote,
                    r.start = row.start, r.end = row.end,
                    r.assertion_id = row.assertion_id,
                    r.method = 'reviewed_assertion', r.run_id = $run_id,
                    r.observed_at = row.observed_at,
                    r.recorded_at = $recorded_at
                """,
                rows=rows,
                version_id=document.document_version_id,
                run_id=run.run_id,
                recorded_at=run.started_at,
            )

        maturity: Dict[str, List[Dict[str, Any]]] = {}
        for row in GraphStore._maturity_evidence(result):
            maturity.setdefault(cypher_identifier(row["label"]), []).append(
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "label"
                    },
                    "observed_at": _chunk_date(document, row["chunk_id"]),
                }
            )
        for label, rows in maturity.items():
            await _run(
                tx,
                f"""
                UNWIND $rows AS row
                MATCH (subject:{label} {{concept_id: row.subject}})
                MATCH (chunk:Chunk {{chunk_id: row.chunk_id}})
                MERGE (subject)-[r:HAS_MATURITY_EVIDENCE
                      {{assertion_id: row.assertion_id, start: row.start}}
                      ]->(chunk)
                SET r.stage = row.stage, r.stage_rank = row.stage_rank,
                    r.trl = row.trl, r.quote = row.quote, r.end = row.end,
                    r.run_id = $run_id, r.observed_at = row.observed_at,
                    r.recorded_at = $recorded_at
                """,
                rows=rows,
                run_id=run.run_id,
                recorded_at=run.started_at,
            )

        if result.economic_evidence:
            await _run(
                tx,
                """
                UNWIND $rows AS row
                MATCH (technology:Technology
                       {concept_id: row.technology_concept_id})
                MATCH (chunk:Chunk {chunk_id: row.chunk_id})
                MERGE (technology)-[r:HAS_ECONOMIC_EVIDENCE
                      {evidence_id: row.evidence_id}]->(chunk)
                SET r.category = row.category, r.quote = row.quote,
                    r.start = row.start, r.end = row.end,
                    r.amount_text = row.amount_text,
                    r.amount_value = row.amount_value,
                    r.currency = row.currency,
                    r.unit = row.unit, r.period = row.period,
                    r.assertion_id = row.assertion_id,
                    r.confidence = row.confidence, r.status = row.status,
                    r.run_id = $run_id,
                    r.polarity = row.polarity, r.modality = row.modality,
                    r.observed_at = row.observed_at,
                    r.recorded_at = $recorded_at
                """,
                rows=[
                    {
                        **evidence.model_dump(mode="json"),
                        "observed_at": _chunk_date(
                            document, evidence.chunk_id
                        ),
                    }
                    for evidence in result.economic_evidence
                ],
                run_id=run.run_id,
                recorded_at=run.started_at,
            )

        await GraphStore._write_assertions(tx, document, result, labels)
