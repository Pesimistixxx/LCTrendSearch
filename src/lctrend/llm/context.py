"""Document-local original-text packets with explicit, bounded context
expansion.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.config import load_catalog
from ..core.models import Chunk, DocumentEnvelope, stable_id


class ContextBudgetError(ValueError):
    pass


class ContextContractError(ValueError):
    pass


class PipelineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primary_chunks: int = Field(default=6, ge=1)
    max_context_chunks: int = Field(default=12, ge=1)
    max_source_chars: int = Field(default=24000, ge=1)
    max_payload_chars: int = Field(default=30000, ge=1)
    max_model_calls: int = Field(default=8, ge=1)
    max_context_rounds: int = Field(default=2, ge=0)
    max_requests_per_round: int = Field(default=4, ge=1)
    search_limit: int = Field(default=3, ge=1)
    max_map_entries: int = Field(default=60, ge=0)
    max_retries: int = Field(default=1, ge=0)
    retry_delay_seconds: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    max_retry_delay_seconds: float = Field(
        default=2.0, ge=0, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def valid_limits(self) -> "PipelineSettings":
        if self.primary_chunks > self.max_context_chunks:
            raise ValueError(
                "primary_chunks must not exceed max_context_chunks"
            )
        if self.retry_delay_seconds > self.max_retry_delay_seconds:
            raise ValueError(
                "retry_delay_seconds must not exceed max_retry_delay_seconds"
            )
        return self

    @classmethod
    def from_catalog(cls) -> "PipelineSettings":
        return cls.model_validate(load_catalog("pipeline")["settings"])


class ContextPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")
    packet_id: str
    focus_chunk_ids: List[str] = Field(default_factory=list)
    support_chunk_ids: List[str] = Field(default_factory=list)
    selection_reasons: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_ids(self) -> "ContextPacket":
        ids = [*self.focus_chunk_ids, *self.support_chunk_ids]
        if len(ids) != len(set(ids)):
            raise ValueError("A chunk must occur only once in a packet")
        return self


class PacketPlan(BaseModel):
    packets: List[ContextPacket] = Field(default_factory=list)
    omitted_chunk_ids: List[str] = Field(default_factory=list)
    omitted_reasons: Dict[str, str] = Field(default_factory=dict)
    support_omissions: Dict[str, List[str]] = Field(default_factory=dict)


def _ordered(document: DocumentEnvelope) -> List[Chunk]:
    return sorted(document.chunks, key=lambda item: item.order)


def _index(document: DocumentEnvelope) -> Dict[str, Chunk]:
    by_id = {chunk.chunk_id: chunk for chunk in document.chunks}
    if len(by_id) != len(document.chunks):
        raise ContextContractError("Duplicate chunk IDs in document")
    return by_id


def _selected(document: DocumentEnvelope, ids: Iterable[str]) -> List[Chunk]:
    requested = list(ids)
    if len(requested) != len(set(requested)):
        raise ContextContractError("Duplicate selected chunk IDs")
    by_id = _index(document)
    missing = set(requested) - by_id.keys()
    if missing:
        raise ContextContractError(
            "Unknown chunks for this document version: "
            + ", ".join(sorted(missing))
        )
    if any(
        by_id[chunk_id].parse_status == "rejected" for chunk_id in requested
    ):
        raise ContextContractError(
            "A rejected source chunk cannot be used as context evidence"
        )
    return [
        chunk
        for chunk in _ordered(document)
        if chunk.chunk_id in set(requested)
    ]


def _metadata(document: DocumentEnvelope) -> Dict[str, Any]:
    allowed = load_catalog("pipeline")["metadata_fields"]
    return {
        "document_id": document.document_id,
        "document_version_id": document.document_version_id,
        "title": document.title,
        "language": document.language,
        "document_type": document.document_type.value,
        "coverage": document.coverage,
        "quality_status": document.quality_status,
        "published_at": document.published_at,
        "version_published_at": document.version_published_at,
        "retrieved_at": document.retrieved_at,
        "artifact_sha256": document.artifact.sha256,
        "media_type": document.artifact.media_type,
        "domains": [domain.name for domain in document.domains],
        "identifiers": [
            identifier.scheme + ":" + identifier.value
            for identifier in document.identifiers
        ],
        "source": {
            "name": document.source.name,
            "source_type": document.source.source_type,
            "source_family": document.source.source_family,
            "record_id": document.source.record_id,
        },
        "metadata": {
            key: document.metadata[key]
            for key in allowed
            if key in document.metadata
        },
    }


def _payload_size(payload: Dict[str, Any]) -> int:
    # Matches the provider's normal JSON serialization, including whitespace
    # and metadata.
    return len(json.dumps(payload, ensure_ascii=False))


def _source_limits(chunks: List[Chunk], settings: PipelineSettings) -> None:
    if len(chunks) > settings.max_context_chunks:
        raise ContextBudgetError("context_chunks_limit")
    if sum(len(chunk.text) for chunk in chunks) > settings.max_source_chars:
        raise ContextBudgetError("source_chars_limit")


def build_payload(
    document: DocumentEnvelope,
    packet: ContextPacket,
    settings: PipelineSettings,
    feedback: Optional[List[str]] = None,
    hints: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    chunks = _selected(
        document, [*packet.focus_chunk_ids, *packet.support_chunk_ids]
    )
    _source_limits(chunks, settings)
    order = _ordered(document)
    positions = {
        chunk.chunk_id: position for position, chunk in enumerate(order)
    }
    selected_positions = [positions[chunk.chunk_id] for chunk in chunks]
    selected_ids = {chunk.chunk_id for chunk in chunks}
    navigation = sorted(
        order,
        key=lambda item: (
            item.chunk_id not in selected_ids,
            min(
                (
                    abs(positions[item.chunk_id] - position)
                    for position in selected_positions
                ),
                default=positions[item.chunk_id],
            ),
            positions[item.chunk_id],
        ),
    )[: settings.max_map_entries]
    locator_fields = load_catalog("pipeline")["navigation_locator_fields"]
    entries = [
        {
            "chunk_id": item.chunk_id,
            "order": item.order,
            "kind": item.kind,
            "section_path": item.section_path,
            "text_chars": len(item.text),
            "locator": {
                key: item.locator[key]
                for key in locator_fields
                if key in item.locator
            },
        }
        for item in navigation
    ]
    payload = {
        "predicate_contract": load_catalog("llm_schema")["predicates"],
        "document": _metadata(document),
        "packet": packet.model_dump(mode="json"),
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        "document_map": {
            "entries": entries,
            "total_chunks": len(order),
            "omitted_entries": len(order) - len(entries),
            "limit_reason": "max_map_entries"
            if len(entries) < len(order)
            else None,
        },
        "feedback": feedback or [],
    }
    if hints is not None:
        # Machine NER spans are optional navigation: the first thing dropped.
        visible = [hint for hint in hints if hint["chunk_id"] in selected_ids]
        payload["ner_hints"] = visible
        while visible and _payload_size(payload) > settings.max_payload_chars:
            visible.pop()
    # The index is navigation, not source evidence. Its bounded reduction is
    # visible; original source chunks, qualifiers and user-supplied metadata
    # are never sliced.
    while entries and _payload_size(payload) > settings.max_payload_chars:
        entries.pop()
        payload["document_map"]["omitted_entries"] = len(order) - len(entries)
        payload["document_map"]["limit_reason"] = "payload_budget"
    entries.sort(key=lambda item: item["order"])
    if _payload_size(payload) > settings.max_payload_chars:
        raise ContextBudgetError("payload_chars_limit")
    return payload


def _stream(chunk: Chunk) -> tuple:
    # Neighboring sections of a file can help; unrelated releases/files cannot.
    return (
        chunk.kind,
        chunk.locator.get("path"),
        chunk.locator.get("release_id"),
        chunk.locator.get("json_pointer"),
        chunk.locator.get("part"),
    )


def _packet(document: DocumentEnvelope, focus: List[str]) -> ContextPacket:
    return ContextPacket(
        packet_id=stable_id("packet", document.document_version_id, *focus),
        focus_chunk_ids=focus,
        selection_reasons={chunk_id: "primary_source" for chunk_id in focus},
    )


def _neighbor_ids(document: DocumentEnvelope, ids: Iterable[str]) -> List[str]:
    ordered = _ordered(document)
    selected = set(ids)
    found = set()
    for index, chunk in enumerate(ordered):
        if chunk.chunk_id in selected:
            for neighbor in ordered[max(0, index - 1) : index + 2]:
                if neighbor.chunk_id not in selected and _stream(
                    neighbor
                ) == _stream(chunk):
                    found.add(neighbor.chunk_id)
    return [chunk.chunk_id for chunk in ordered if chunk.chunk_id in found]


def _with_neighbors(
    document: DocumentEnvelope,
    packet: ContextPacket,
    settings: PipelineSettings,
) -> Tuple[ContextPacket, List[str]]:
    result = packet.model_copy(deep=True)
    omitted = []
    by_id = _index(document)
    for chunk_id in _neighbor_ids(document, packet.focus_chunk_ids):
        if by_id[chunk_id].parse_status == "rejected":
            omitted.append(chunk_id)
            continue
        candidate = result.model_copy(deep=True)
        candidate.support_chunk_ids.append(chunk_id)
        candidate.selection_reasons[chunk_id] = "adjacent_source_context"
        try:
            build_payload(document, candidate, settings)
        except ContextBudgetError:
            omitted.append(chunk_id)
        else:
            result = candidate
    return result, omitted


def plan_packets(
    document: DocumentEnvelope, settings: PipelineSettings
) -> PacketPlan:
    packets, focus, omitted, reasons, support_omissions = [], [], [], {}, {}
    by_id = _index(document)

    def finish() -> None:
        if focus:
            packet, omitted_support = _with_neighbors(
                document, _packet(document, list(focus)), settings
            )
            packets.append(packet)
            if omitted_support:
                support_omissions[packet.packet_id] = omitted_support
            focus.clear()

    for chunk in _ordered(document):
        if not chunk.text.strip() or chunk.parse_status == "rejected":
            omitted.append(chunk.chunk_id)
            reasons[chunk.chunk_id] = (
                "empty_text" if not chunk.text.strip() else "parse_rejected"
            )
            continue
        if focus and (
            len(focus) >= settings.primary_chunks
            or by_id[focus[-1]].section_path != chunk.section_path
            or _stream(by_id[focus[-1]]) != _stream(chunk)
        ):
            finish()
        candidate = _packet(document, [*focus, chunk.chunk_id])
        try:
            build_payload(document, candidate, settings)
        except ContextBudgetError:
            finish()
            try:
                build_payload(
                    document, _packet(document, [chunk.chunk_id]), settings
                )
            except ContextBudgetError as error:
                omitted.append(chunk.chunk_id)
                reasons[chunk.chunk_id] = str(error)
                continue
        focus.append(chunk.chunk_id)
    finish()
    return PacketPlan(
        packets=packets,
        omitted_chunk_ids=omitted,
        omitted_reasons=reasons,
        support_omissions=support_omissions,
    )


def expand_packet(
    document: DocumentEnvelope,
    packet: ContextPacket,
    requests: Iterable[Any],
    settings: PipelineSettings,
) -> Tuple[ContextPacket, List[Dict[str, Any]]]:
    # Validate the existing packet before resolving any request.
    build_payload(document, packet, settings)
    result = packet.model_copy(deep=True)
    by_id = _index(document)
    outcomes = []
    for index, value in enumerate(requests):
        request = (
            value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else value
        )
        if not isinstance(request, dict):
            raise ContextContractError("A context request must be an object")
        tool, argument = (
            request.get("tool"),
            str(request.get("argument") or ""),
        )
        outcome = {
            "tool": tool,
            "argument": argument,
            "reason": request.get("reason"),
            "added_chunk_ids": [],
        }
        if index >= settings.max_requests_per_round:
            outcome["status"] = "request_limit"
            outcomes.append(outcome)
            continue
        if tool == "read_chunk":
            candidates = [by_id[argument]] if argument in by_id else []
            outcome["status"] = (
                "unknown_chunk" if not candidates else "already_visible"
            )
        elif tool == "search_chunks":
            query = argument.strip().casefold()
            candidates = [
                chunk
                for chunk in _ordered(document)
                if chunk.parse_status != "rejected"
                and query
                and query in chunk.text.casefold()
            ][: settings.search_limit]
            outcome["status"] = (
                "no_match" if not candidates else "already_visible"
            )
        else:
            outcome["status"] = "unsupported_tool"
            outcomes.append(outcome)
            continue
        for chunk in candidates:
            if chunk.parse_status == "rejected":
                outcome["status"] = "rejected_chunk"
                outcome.setdefault("omitted_chunk_ids", []).append(
                    chunk.chunk_id
                )
                continue
            if chunk.chunk_id in [
                *result.focus_chunk_ids,
                *result.support_chunk_ids,
            ]:
                continue
            candidate = result.model_copy(deep=True)
            candidate.support_chunk_ids.append(chunk.chunk_id)
            candidate.selection_reasons[chunk.chunk_id] = "requested:" + tool
            try:
                build_payload(document, candidate, settings)
            except ContextBudgetError as error:
                outcome["status"] = "context_budget"
                outcome.setdefault("omitted_chunk_ids", []).append(
                    chunk.chunk_id
                )
                outcome["budget_reason"] = str(error)
            else:
                result = candidate
                outcome["added_chunk_ids"].append(chunk.chunk_id)
                outcome["status"] = "added"
        outcomes.append(outcome)
    return result, outcomes


def _cited_ids(value: Any) -> set:
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evidence" and isinstance(item, list):
                for span in item:
                    if isinstance(span, dict) and span.get("chunk_id"):
                        found.add(str(span["chunk_id"]))
            else:
                found.update(_cited_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_cited_ids(item))
    return found


def review_payload(
    document: DocumentEnvelope,
    extractiondict: Dict[str, Any],
    visible_ids: Iterable[str],
    settings: PipelineSettings,
) -> Dict[str, Any]:
    visible = set(visible_ids)
    _selected(document, visible)
    required = _cited_ids(extractiondict)
    if required - visible:
        raise ContextContractError(
            "Reviewer evidence references an unseen chunk"
        )
    chunks = _selected(document, required)
    _source_limits(chunks, settings)
    payload = {
        "document": _metadata(document),
        "extraction": extractiondict,
        "predicate_contract": load_catalog("llm_schema")["predicates"],
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        "review_context": {
            "required_chunk_ids": [chunk.chunk_id for chunk in chunks],
            "neighbor_chunk_ids": [],
            "omitted_neighbor_chunk_ids": [],
        },
    }
    if _payload_size(payload) > settings.max_payload_chars:
        raise ContextBudgetError("review_payload_chars_limit")
    by_id = _index(document)
    for chunk_id in _neighbor_ids(document, required):
        if by_id[chunk_id].parse_status == "rejected":
            payload["review_context"]["omitted_neighbor_chunk_ids"].append(
                chunk_id
            )
            continue
        neighbors = payload["review_context"]["neighbor_chunk_ids"]
        candidate_chunks = _selected(
            document, [*required, *neighbors, chunk_id]
        )
        candidate = {
            **payload,
            "chunks": [
                chunk.model_dump(mode="json") for chunk in candidate_chunks
            ],
            "review_context": {
                **payload["review_context"],
                "neighbor_chunk_ids": [*neighbors, chunk_id],
            },
        }
        try:
            _source_limits(candidate_chunks, settings)
            if _payload_size(candidate) > settings.max_payload_chars:
                raise ContextBudgetError("review_payload_chars_limit")
        except ContextBudgetError:
            payload["review_context"]["omitted_neighbor_chunk_ids"].append(
                chunk_id
            )
        else:
            payload = candidate
    if _payload_size(payload) > settings.max_payload_chars:
        raise ContextBudgetError("review_payload_chars_limit")
    return payload
