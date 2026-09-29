from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


def stable_id(namespace: str, *parts: object) -> str:
    value = "\x1f".join(
        str(part).strip() for part in parts if part is not None
    )
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{namespace}:{digest}"


# A semantic match proposes a review candidate; the mention keeps its own
# provisional concept (see resolver.json semantic.use_in_llm).
SEMANTIC_CANDIDATE_METHOD = "new_provisional_semantic_candidate"
# Several concepts share the identity key of a mention; it is linked to all
# of them as ambiguous until they are merged.
AMBIGUOUS_COLLISION_METHOD = "deterministic_alias_collision"
# A name the source itself equates with the mention's ("compactin
# (ML-236B)"): it resolves the mention, and when it already names another
# concept the pair is a merge candidate.
DECLARED_ALIAS_METHOD = "declared_alias"


class DocumentType(str, Enum):
    ARTICLE = "article"
    PATENT = "patent"
    REPOSITORY = "repository"
    PACKAGE = "package"
    REPORT = "report"
    TRANSCRIPT = "transcript"
    STANDARD = "standard"
    JOB_POSTING = "job_posting"
    GRANT = "grant"
    REGULATORY = "regulatory"
    NEWS = "news"


class ConceptKind(str, Enum):
    TECHNOLOGY = "Technology"
    METHOD = "Method"
    TASK = "Task"
    PROBLEM = "Problem"
    APPLICATION_CONTEXT = "ApplicationContext"
    METRIC = "Metric"
    MATERIAL = "Material"
    DOMAIN = "Domain"
    MARKET_SEGMENT = "MarketSegment"
    ORGANIZATION = "Organization"
    COMPANY = "Company"
    UNIVERSITY = "University"
    COUNTRY = "Country"
    CANDIDATE = "ConceptCandidate"


class ExternalId(BaseModel):
    scheme: str
    value: str

    @property
    def external_id(self) -> str:
        return f"{self.scheme.lower()}:{self.value.strip().lower()}"


class SourceRef(BaseModel):
    source_id: str
    name: str
    source_type: str
    record_id: str
    canonical_url: Optional[str] = None
    source_family: Optional[str] = None
    independence_group: Optional[str] = None
    reliability_tier: int = 1


class Artifact(BaseModel):
    uri: str
    sha256: str
    media_type: str
    byte_length: int = 0
    access_status: str = "available"


class Contributor(BaseModel):
    contributor_id: str
    name: str
    kind: str = "person"
    role: str = "author"
    external_ids: List[ExternalId] = Field(default_factory=list)
    affiliation_ids: List[str] = Field(default_factory=list)


class Organization(BaseModel):
    organization_id: str
    name: str
    organization_type: str = "other"
    country_code: Optional[str] = None
    role: str = "associated"
    external_ids: List[ExternalId] = Field(default_factory=list)


class Country(BaseModel):
    country_id: str
    code: str
    name: Optional[str] = None
    role: str = "associated"


class Domain(BaseModel):
    domain_id: str
    name: str
    parent_name: Optional[str] = None
    external_ids: List[ExternalId] = Field(default_factory=list)


class EconomicFact(BaseModel):
    """A structured money fact of a source record, not of text: a grant
    award, a salary offer. It belongs to the whole document, so every
    technology the document mentions is touched by it; the organizations
    are the document's own (recipient, employer, funder).
    """

    fact_id: str
    # grant_award, salary_offer, ...
    category: str
    amount: Optional[float] = None
    # Upper bound of a range ("from 100 000 to 150 000"); amount is the
    # lower bound, or the only value.
    amount_max: Optional[float] = None
    currency: Optional[str] = None
    # total, per_year, per_month
    period: str = "total"
    # The date the money refers to: award, fiscal year, vacancy posting.
    observed_at: Optional[str] = None
    recipient_organization_id: Optional[str] = None
    payer_organization_id: Optional[str] = None
    # The record field the amount was read from.
    source_field: Optional[str] = None
    # Constant dollars of real_base_year (core.money); None when the
    # currency or the year is unknown.
    amount_usd_real: Optional[float] = None
    amount_max_usd_real: Optional[float] = None
    real_base_year: Optional[int] = None
    real_status: Optional[str] = None


class Chunk(BaseModel):
    chunk_id: str
    kind: str
    text: str
    order: int
    section_path: List[str] = Field(default_factory=list)
    locator: Dict[str, Any] = Field(default_factory=dict)
    content_hash: Optional[str] = None
    parse_status: str = "accepted"

    @model_validator(mode="after")
    def set_content_hash(self) -> "Chunk":
        if self.content_hash is None:
            self.content_hash = hashlib.sha256(
                self.text.encode("utf-8")
            ).hexdigest()
        return self


class DocumentEnvelope(BaseModel):
    schema_version: str = "material-pipeline/0.1"
    document_id: str
    document_version_id: str
    document_type: DocumentType
    title: str
    language: Optional[str] = None
    published_at: Optional[str] = None
    version_published_at: Optional[str] = None
    retrieved_at: Optional[str] = None
    metrics_observed_at: Optional[str] = None
    source: SourceRef
    artifact: Artifact
    identifiers: List[ExternalId] = Field(default_factory=list)
    contributors: List[Contributor] = Field(default_factory=list)
    organizations: List[Organization] = Field(default_factory=list)
    countries: List[Country] = Field(default_factory=list)
    domains: List[Domain] = Field(default_factory=list)
    chunks: List[Chunk] = Field(default_factory=list)
    economic_facts: List[EconomicFact] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    coverage: str = "metadata_only"
    quality_status: str = "accepted"

    @model_validator(mode="after")
    def unique_chunk_ids(self) -> "DocumentEnvelope":
        ids = [chunk.chunk_id for chunk in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "chunk_id must be unique within a document version"
            )
        # Several domains are what makes a cross-domain (bridge) signal
        # visible; they only need to be distinct.
        domain_ids = [domain.domain_id for domain in self.domains]
        if len(domain_ids) != len(set(domain_ids)):
            raise ValueError("domain_id must be unique within a document")
        return self


class ConceptName(BaseModel):
    name_id: str
    text: str
    normalized_text: str
    language: Optional[str] = None
    name_kind: str = "preferred"
    status: str = "accepted"


class Concept(BaseModel):
    concept_id: str
    kind: ConceptKind
    preferred_label: str
    definition: Optional[str] = None
    language: Optional[str] = None
    status: str = "provisional"
    names: List[ConceptName] = Field(default_factory=list)
    # Lexical identity key (extraction.lexical) the concept was created
    # under; None for concepts created before key v2.
    identity_key: Optional[str] = None
    # Resolved mentions per canonical form; the most frequent form is the
    # preferred label of a concept that has not been reviewed.
    label_counts: Dict[str, int] = Field(default_factory=dict)
    # Resolved mentions per reported kind; the kind of the concept is
    # settled from them (extraction.lexical.settled_kind).
    kind_counts: Dict[str, int] = Field(default_factory=dict)
    # Technology contract profile (docs/technology-contract.md): mechanism,
    # function, type, boundary, verbatim names, field evidence and
    # classification_status. A validated profile replaces a proposed one.
    profile: Optional[Dict[str, Any]] = None


class Mention(BaseModel):
    mention_id: str
    chunk_id: str
    surface_text: str
    canonical_text: Optional[str] = None
    start: int
    end: int
    type_candidates: List[ConceptKind]
    mention_role: str = "candidate"
    discourse_role: str = "unknown"
    confidence: Optional[float] = None
    status: str = "candidate"
    # What the source says the entity is; context for semantic matching and
    # review, never identity evidence.
    definition: Optional[str] = None
    # Other names the source itself equates with this one ("compactin
    # (ML-236B)"), checked against the chunk; they are identity evidence.
    declared_aliases: List[str] = Field(default_factory=list)
    # Technology contract profile of the entity in this document.
    profile: Optional[Dict[str, Any]] = None


class EvidenceSpan(BaseModel):
    chunk_id: str
    quote: str
    start: int
    end: int
    supports_fields: List[str] = Field(default_factory=list)


class Assertion(BaseModel):
    assertion_id: str
    predicate: str
    roles: Dict[str, str]
    evidence: List[EvidenceSpan]
    qualifiers: Dict[str, Any] = Field(default_factory=dict)
    values: List[Dict[str, Any]] = Field(default_factory=list)
    polarity: str = "affirmed"
    modality: str = "reported"
    attribution_kind: str = "author_reported"
    evidence_kind: str = "author_statement"
    extraction_confidence: Optional[float] = None
    verification_status: str = "unverified"
    status: str = "needs_review"
    claim_group_id: Optional[str] = None
    evidence_family_id: Optional[str] = None


class ResolutionDecision(BaseModel):
    resolution_id: str
    mention_id: str
    status: str
    concept_id: Optional[str] = None
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    method: str = "unresolved"
    score: Optional[float] = None
    basis: List[str] = Field(default_factory=list)
    resolver_version: str = "resolver/0.1"
    taxonomy_version: Optional[str] = None
    review_status: str = "pending"


class ProcessingRun(BaseModel):
    run_id: str
    pipeline_version: str = "0.1.0"
    parser: str
    model_revision: Optional[str] = None
    prompt_hash: Optional[str] = None
    config_hash: str
    started_at: str
    status: str = "succeeded"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trace: List[Dict[str, Any]] = Field(default_factory=list)


class EconomicEvidence(BaseModel):
    evidence_id: str
    technology_concept_id: str
    chunk_id: str
    category: str
    quote: str
    start: int
    end: int
    amount_text: Optional[str] = None
    # Numeric reading of amount_text (scale words applied), same currency.
    amount_value: Optional[float] = None
    currency: Optional[str] = None
    assertion_id: Optional[str] = None
    unit: Optional[str] = None
    period: Optional[str] = None
    confidence: Optional[float] = None
    polarity: str = "affirmed"
    modality: str = "reported"
    status: str = "candidate"


class ExtractionResult(BaseModel):
    document_version_id: str
    run: ProcessingRun
    mentions: List[Mention] = Field(default_factory=list)
    concepts: List[Concept] = Field(default_factory=list)
    assertions: List[Assertion] = Field(default_factory=list)
    resolutions: List[ResolutionDecision] = Field(default_factory=list)
    economic_evidence: List[EconomicEvidence] = Field(default_factory=list)
    # Unit vectors of concept labels (semantic layer), keyed by concept_id;
    # stored on concept nodes for similarity search and taxonomy building.
    concept_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    # Unit vectors of evidence chunks (text of accepted claims), keyed by
    # chunk_id; semantic retrieval of related context across documents.
    chunk_embeddings: Dict[str, List[float]] = Field(default_factory=dict)
    embedding_model: Optional[str] = None


def json_value(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def validate_extraction(
    document: DocumentEnvelope, result: ExtractionResult
) -> None:
    if result.document_version_id != document.document_version_id:
        raise ValueError("extraction belongs to another document version")

    chunks = {chunk.chunk_id: chunk for chunk in document.chunks}
    concept_ids = {concept.concept_id for concept in result.concepts}
    mention_ids = {mention.mention_id for mention in result.mentions}

    for chunk_id in result.chunk_embeddings:
        if chunk_id not in chunks:
            raise ValueError(
                f"embedded chunk {chunk_id} is not in the document"
            )

    for mention in result.mentions:
        chunk = chunks.get(mention.chunk_id)
        if (
            chunk is None
            or chunk.text[mention.start : mention.end] != mention.surface_text
        ):
            raise ValueError(
                f"mention {mention.mention_id} is not anchored to its chunk"
            )

    for decision in result.resolutions:
        if decision.mention_id not in mention_ids:
            raise ValueError(
                f"resolution {decision.resolution_id} "
                "references missing mention"
            )
        if decision.concept_id and decision.concept_id not in concept_ids:
            raise ValueError(
                f"resolution {decision.resolution_id} "
                "references missing concept"
            )

    for assertion in result.assertions:
        missing = set(assertion.roles.values()) - concept_ids
        if missing:
            raise ValueError(
                f"assertion {assertion.assertion_id} "
                f"references missing concepts: {missing}"
            )
        if not assertion.evidence:
            raise ValueError(
                f"assertion {assertion.assertion_id} has no evidence"
            )
        for evidence in assertion.evidence:
            chunk = chunks.get(evidence.chunk_id)
            if (
                chunk is None
                or chunk.text[evidence.start : evidence.end] != evidence.quote
            ):
                raise ValueError(
                    f"assertion {assertion.assertion_id} "
                    "has invalid evidence anchor"
                )

    concept_kinds = {
        concept.concept_id: concept.kind for concept in result.concepts
    }
    for evidence in result.economic_evidence:
        if (
            concept_kinds.get(evidence.technology_concept_id)
            != ConceptKind.TECHNOLOGY
        ):
            raise ValueError(
                f"economic evidence {evidence.evidence_id} "
                "must reference a technology"
            )
        chunk = chunks.get(evidence.chunk_id)
        if (
            chunk is None
            or chunk.text[evidence.start : evidence.end] != evidence.quote
        ):
            raise ValueError(
                f"economic evidence {evidence.evidence_id} "
                "has invalid evidence anchor"
            )
