"""Document-local LLM contracts; models cannot assign acceptance or global
IDs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from ..core.models import ConceptKind


class LocalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Elements dropped by validate_items: (field, index, reason).
    _dropped_items: List[Dict[str, Any]] = PrivateAttr(default_factory=list)


def _item_model(annotation: Any) -> Optional[Type[BaseModel]]:
    """The model of a ``List[Model]`` field, if any."""
    for argument in get_args(annotation):
        if isinstance(argument, type) and issubclass(argument, BaseModel):
            return argument
    return None


def _known_fields(model: Type[BaseModel], value: Any) -> Any:
    """Drop keys the model does not define, at every nesting level.

    Unknown keys carry nothing the pipeline reads (acceptance and global IDs
    are never read from the model), so they must not cost the element.
    """
    if not isinstance(value, dict):
        return value
    cleaned = {}
    for name, field in model.model_fields.items():
        if name not in value:
            continue
        item = value[name]
        nested = _item_model(field.annotation)
        if nested is not None and isinstance(item, list):
            item = [_known_fields(nested, element) for element in item]
        elif (
            isinstance(field.annotation, type)
            and issubclass(field.annotation, BaseModel)
        ):
            item = _known_fields(field.annotation, item)
        cleaned[name] = item
    return cleaned


def validate_items(model: Type[LocalModel], data: Any) -> LocalModel:
    """Validate a response element by element.

    One malformed entity, claim or review item is dropped (and reported)
    instead of failing the whole packet. A response whose top level is not
    an object of the expected shape still fails.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{model.__name__} response must be an object")
    cleaned: Dict[str, Any] = {}
    dropped: List[Dict[str, Any]] = []
    for name, field in model.model_fields.items():
        if name not in data:
            continue
        value = data[name]
        element = _item_model(field.annotation)
        if element is None or not isinstance(value, list):
            cleaned[name] = value
            continue
        kept = []
        for index, item in enumerate(value):
            try:
                kept.append(
                    element.model_validate(_known_fields(element, item))
                )
            except ValidationError as exc:
                dropped.append(
                    {
                        "field": name,
                        "index": index,
                        # Error types only: messages can echo source text.
                        "reasons": sorted(
                            {error["type"] for error in exc.errors()}
                        ),
                    }
                )
        cleaned[name] = kept
    result = model.model_validate(cleaned)
    result._dropped_items = dropped
    return result


class SourceSpan(LocalModel):
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    start: Optional[int] = None
    end: Optional[int] = None


class SourceName(LocalModel):
    """A name of the technology written in the source, verbatim."""

    name: str = Field(min_length=1)
    # Where and how the source uses it ("abstract, abbreviation of ...").
    context: Optional[str] = None


class SupportSpan(SourceSpan):
    """A quote that supports fields of a technology definition, not a
    mention of its name (docs/technology-contract.md)."""

    supports: List[
        Literal[
            "definition",
            "mechanism",
            "function",
            "boundary",
            "application",
            "name",
        ]
    ] = Field(min_length=1)


class LocalEntity(LocalModel):
    local_id: str = Field(min_length=1)
    # For Technology and Method: the canonical name, which may summarize
    # the source; source_names keep the verbatim names.
    label: str = Field(min_length=1)
    kind: ConceptKind
    # What the source says the entity is (Technology, Method, Material).
    definition: Optional[str] = None
    # Technology contract (Technology, Method): distinguishing principle,
    # performed operation, form of the technology, what it is not.
    technical_mechanism: Optional[str] = None
    technical_function: Optional[str] = None
    technology_type: Optional[str] = None
    boundary: Optional[str] = None
    application_context: Optional[str] = None
    source_names: List[SourceName] = Field(default_factory=list)
    support: List[SupportSpan] = Field(default_factory=list)
    # Why the contract's conditions are not met, in the model's words.
    uncertainty: Optional[str] = None
    # Set by validation only (hidden from the model's schema): validated,
    # or proposed with the unmet conditions.
    classification_status: SkipJsonSchema[Optional[str]] = None
    contract_issues: SkipJsonSchema[List[str]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _name_evidence(cls, value: Any) -> Any:
        # Live (2026-09-29): with source_names and support given, the model
        # often omits the name quote, and the whole technology was lost.
        # The first source name becomes it; validation finds its chunk.
        if (
            isinstance(value, dict)
            and not value.get("evidence")
            and value.get("source_names")
        ):
            names = value["source_names"]
            first = names[0] if isinstance(names, list) and names else None
            name = first.get("name") if isinstance(first, dict) else first
            support = value.get("support") or [{}]
            chunk = support[0].get("chunk_id") if support[0] else None
            if isinstance(name, str) and name.strip():
                value = {
                    **value,
                    "evidence": [{"chunk_id": chunk or "?", "quote": name}],
                }
        return value
    # Other names the source itself equates with the label: an abbreviation
    # or code in parentheses, "also known as".
    aliases: List[str] = Field(default_factory=list)
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
    tool: Literal["read_chunk", "search_chunks", "search_graph"]
    argument: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    # Claims that wait for this context; empty means the whole packet.
    claim_ids: List[str] = Field(default_factory=list)


class Extraction(LocalModel):
    entities: List[LocalEntity] = Field(default_factory=list)
    claims: List[LocalClaim] = Field(default_factory=list)
    context_requests: List[ContextRequest] = Field(default_factory=list)


TRIAGE_VERDICTS = (
    "technology",
    "product",
    "software_component",
    "feature_or_metric",
    "title",
    "dataset_or_resource",
    "domain",
    "other",
)


class TriageItem(LocalModel):
    """Whether one extracted technology is a technology of the radar."""

    id: str = Field(min_length=1)
    verdict: Literal[TRIAGE_VERDICTS]
    reason: str = ""


class Triage(LocalModel):
    items: List[TriageItem] = Field(default_factory=list)


class ReviewItem(LocalModel):
    claim_id: str = Field(min_length=1)
    decision: Literal["supported", "unsupported", "unclear"]
    reason: str = Field(min_length=1)


class Review(LocalModel):
    items: List[ReviewItem] = Field(default_factory=list)
