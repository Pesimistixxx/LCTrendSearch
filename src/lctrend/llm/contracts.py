"""Document-local LLM contracts; models cannot assign acceptance or global
IDs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..core.models import ConceptKind


class LocalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceSpan(LocalModel):
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    start: Optional[int] = None
    end: Optional[int] = None


class LocalEntity(LocalModel):
    local_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: ConceptKind
    definition: Optional[str] = None
    # ISO 3166-1 alpha-2 for Country entities only; it canonicalizes
    # "Germany"/"Германия"/"German" to one concept across documents.
    country_code: Optional[str] = None
    evidence: List[SourceSpan] = Field(min_length=1)


class LocalClaim(LocalModel):
    claim_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    roles: Dict[str, str]
    qualifiers: Dict[str, Any] = Field(default_factory=dict)
    values: List[Dict[str, Any]] = Field(default_factory=list)
    polarity: Literal["affirmed", "negated", "unknown"] = "unknown"
    modality: Literal[
        "reported", "observed", "planned", "hypothetical", "unknown"
    ] = "unknown"
    attribution_kind: str = "author_reported"
    evidence: List[SourceSpan] = Field(min_length=1)


class ContextRequest(LocalModel):
    tool: Literal["read_chunk", "search_chunks"]
    argument: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class Extraction(LocalModel):
    entities: List[LocalEntity] = Field(default_factory=list)
    claims: List[LocalClaim] = Field(default_factory=list)
    context_requests: List[ContextRequest] = Field(default_factory=list)


class ReviewItem(LocalModel):
    claim_id: str = Field(min_length=1)
    decision: Literal["supported", "unsupported", "unclear"]
    reason: str = Field(min_length=1)


class Review(LocalModel):
    items: List[ReviewItem] = Field(default_factory=list)
