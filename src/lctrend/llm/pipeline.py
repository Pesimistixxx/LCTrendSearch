"""Bounded document processing: packets, extraction, review, resolution.

The coordinator reads bounded original context through an injected reader.
The caller publishes its validated result to Neo4j.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
from uuid import uuid4

from ..core.aio import resolve
from ..core.config import RESOURCE_DIR, load_catalog
from ..core.models import (
    SEMANTIC_CANDIDATE_METHOD,
    Assertion,
    Concept,
    ConceptKind,
    DocumentEnvelope,
    EvidenceSpan,
    ExtractionResult,
    Mention,
    ProcessingRun,
    json_value,
    stable_id,
    validate_extraction,
)
from ..extraction.economics import (
    economic_evidence_from_assertions,
    extract_economic_evidence,
)
from ..extraction.resolver import (
    ConceptRegistry,
    concept_text,
    resolve_mentions,
)
from ..ingest.processed import fulltext_sha256
from ..linking import known as known_layer
from ..linking.sections import SKIPPED_REASON
from .client import CALL_LOG, LLMError, Provider, rate_limited
from .context import (
    ContextBudgetError,
    PipelineSettings,
    build_payload,
    expand_context,
    plan_packets,
    review_batches,
    split_packet,
)
from .contracts import Extraction, Review
from .validation import validate_local_extraction, validate_review

logger = logging.getLogger(__name__)
# Indirection lets offline tests skip real backoff pauses.
_sleep = asyncio.sleep


def _emit(event, **value):
    if event is not None:
        try:
            event({"branch": "llm", **value})
        except Exception:
            logger.debug("Progress event callback failed", exc_info=True)


def _prompt(stage: str) -> str:
    override = os.getenv("LCTREND_CONFIG_DIR")
    candidate = (
        Path(override) / "prompts" / f"{stage}.txt" if override else None
    )
    path = (
        candidate
        if candidate and candidate.is_file()
        else RESOURCE_DIR / "prompts" / f"{stage}.txt"
    )
    return path.read_text(encoding="utf-8-sig")


def _refs(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [_refs(item, mapping) for item in value]
    if isinstance(value, dict):
        return {
            key: mapping[item] if key == "entity_ref" else _refs(item, mapping)
            for key, item in value.items()
        }
    return value


ITEM_ISSUE_CODES = {"review_unclear", "conflicting_reviews"}


def _item_issue(issue: dict) -> bool:
    """One entity or claim was gated; the packet itself was processed."""
    return "item" in issue or issue.get("code") in ITEM_ISSUE_CODES


def _source_key(chunks: dict, span: Any) -> list:
    """Recognize the same evidence in overlap by original text coordinates."""
    value = span.model_dump() if hasattr(span, "model_dump") else span
    chunk = chunks[value["chunk_id"]]
    stream = chunk.locator.get("source_stream_id")
    segments = chunk.locator.get("source_segments", [])
    if stream and segments:
        first = next(
            (
                s
                for s in segments
                if s["chunk_start"] <= value["start"] < s["chunk_end"]
            ),
            None,
        )
        last = next(
            (
                s
                for s in segments
                if s["chunk_start"] < value["end"] <= s["chunk_end"]
            ),
            None,
        )
        if first and last:
            return [
                "source",
                stream,
                first["source_start"] + value["start"] - first["chunk_start"],
                last["source_start"] + value["end"] - last["chunk_start"],
            ]
    return ["chunk", value["chunk_id"], value["start"], value["end"]]


async def _concept_embeddings(
    concepts: list[Concept],
    resolutions: list,
    semantic: Any,
    metadata: dict,
) -> tuple[dict[str, list[float]], str | None]:
    """Label vectors for graph storage; mostly cached by resolution.

    An unavailable semantic layer is recorded, never a document failure.
    """
    candidates = sum(
        item.method == SEMANTIC_CANDIDATE_METHOD for item in resolutions
    )
    embed = getattr(semantic, "embed", None)
    if embed is None:
        metadata["semantic"] = {"status": "disabled"}
        return {}, None
    kinds = set(load_catalog("resolver")["semantic"]["embedded_kinds"])
    chosen = [concept for concept in concepts if concept.kind.value in kinds]
    vectors = (
        await asyncio.to_thread(
            embed, [concept_text(concept) for concept in chosen]
        )
        if chosen
        else []
    )
    if vectors is None or getattr(semantic, "failure", None):
        metadata["semantic"] = {
            "status": "unavailable",
            "error": getattr(semantic, "failure", None),
            "candidates": candidates,
        }
        return {}, None
    model = getattr(semantic, "embedding_model_name", None)
    metadata["semantic"] = {
        "status": "ok",
        "model": model,
        "embedded_concepts": len(chosen),
        "candidates": candidates,
    }
    if getattr(semantic, "decision_failure", None):
        # Vectors are stored; only the cross-encoder candidates are missing.
        metadata["semantic"]["decision_error"] = semantic.decision_failure
    return {
        concept.concept_id: vector for concept, vector in zip(chosen, vectors)
    }, model


async def _evidence_embeddings(
    document: DocumentEnvelope,
    assertions: list,
    economic: list,
    semantic: Any,
    metadata: dict,
) -> dict[str, list[float]]:
    """Vectors of the chunks that accepted claims quote.

    Only evidence is embedded: it is the text the graph vouches for, and it
    keeps the cost bounded (full texts have hundreds of chunks).
    """
    limit = int(
        load_catalog("pipeline")
        .get("graph_context", {})
        .get("evidence_embedding_max_chunks", 0)
    )
    embed = getattr(semantic, "embed", None)
    status = metadata.get("semantic", {}).get("status")
    if embed is None or not limit or status != "ok":
        return {}
    texts = {chunk.chunk_id: chunk.text for chunk in document.chunks}
    chosen = list(
        dict.fromkeys(
            [
                span.chunk_id
                for assertion in assertions
                if assertion.status == "accepted"
                for span in assertion.evidence
            ]
            + [item.chunk_id for item in economic]
        )
    )
    chosen = [chunk_id for chunk_id in chosen if texts.get(chunk_id)][:limit]
    if not chosen:
        return {}
    vectors = await asyncio.to_thread(
        embed, [texts[chunk_id] for chunk_id in chosen], False
    )
    if vectors is None:
        metadata["semantic"]["evidence_error"] = getattr(
            semantic, "failure", None
        )
        return {}
    metadata["semantic"]["embedded_evidence_chunks"] = len(chosen)
    return dict(zip(chosen, vectors))


def _semantic_reader(reader, semantic):
    """Give graph search the query vector when the store can use it."""
    embed = getattr(semantic, "embed", None)
    if reader is None or embed is None:
        return reader
    try:
        accepts = "query_vector" in inspect.signature(reader).parameters
    except (TypeError, ValueError):
        accepts = False
    if not accepts:
        return reader

    async def read(**kwargs):
        vectors = await asyncio.to_thread(embed, [kwargs["query"]], False)
        if vectors:
            kwargs["query_vector"] = vectors[0]
        return await resolve(reader(**kwargs))

    return read


def _pending_claims(requests) -> Any:
    """Claims an unanswered context request leaves unclear.

    ``True`` when a request names no claims (the whole packet waits),
    otherwise the named claim IDs; other claims keep their review.
    """
    if not requests:
        return frozenset()
    if any(not request.claim_ids for request in requests):
        return True
    return frozenset(
        claim_id for request in requests for claim_id in request.claim_ids
    )


CHUNK_ID = re.compile(r"chunk:[0-9a-f]{24}")


class _ChunkAliases:
    """Short names of the document's chunk IDs in extraction requests.

    A chunk ID ("chunk:" and 24 hex digits) costs about 20 output tokens and
    every evidence span repeats it; "c12" costs two. Requests carry the
    aliases, and answers are mapped back before any validation. Documents
    with other ID forms (fixtures) keep their IDs.
    """

    def __init__(self, document: DocumentEnvelope):
        ids = [chunk.chunk_id for chunk in document.chunks]
        usable = bool(ids) and all(CHUNK_ID.fullmatch(cid) for cid in ids)
        self.forward = (
            {cid: f"c{number}" for number, cid in enumerate(ids, 1)}
            if usable
            else {}
        )
        self.back = {alias: cid for cid, alias in self.forward.items()}

    def _text(self, value: str) -> str:
        if value in self.forward:
            return self.forward[value]
        if "chunk:" not in value:
            return value
        # Feedback strings quote chunk IDs inside JSON text.
        return CHUNK_ID.sub(
            lambda match: self.forward.get(match.group(0), match.group(0)),
            value,
        )

    def encode(self, value: Any) -> Any:
        if not self.forward:
            return value
        if isinstance(value, dict):
            return {
                self._text(key) if isinstance(key, str) else key: self.encode(
                    item
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.encode(item) for item in value]
        if isinstance(value, str):
            return self._text(value)
        return value

    def decode(self, extraction: Extraction) -> Extraction:
        """The answer with real chunk IDs; unknown names stay for the
        validator to reject."""
        if not self.back:
            return extraction
        data = extraction.model_dump(mode="python")
        for item in [*data["entities"], *data["claims"]]:
            for span in item["evidence"]:
                span["chunk_id"] = self.back.get(
                    span["chunk_id"].strip(), span["chunk_id"]
                )
        for request in data["context_requests"]:
            if request["tool"] == "read_chunk":
                request["argument"] = self.back.get(
                    request["argument"].strip(), request["argument"]
                )
        return Extraction.model_validate(data)


def _packet_workers(provider: Provider, settings: PipelineSettings) -> int:
    """Packets of one document processed at once: never more than the
    provider serves concurrently (1 for a single GigaChat key)."""
    capacity = getattr(provider, "max_concurrency", 1)
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        capacity = 1
    return max(1, min(settings.packet_workers, capacity))


def _timing(calls: list, wall_seconds: float, workers: int) -> dict:
    """Where a document's time went: model requests, waiting for a free
    key, and the rest (context, validation, resolution, embeddings)."""
    chat = [call for call in calls if call.get("stage") != "embed"]

    def seconds(rows, field):
        return round(sum(row.get(field) or 0 for row in rows) / 1000, 1)

    def tokens(rows, field):
        return sum((row.get("tokens") or {}).get(field) or 0 for row in rows)

    stages = {}
    for stage in sorted({call.get("stage") for call in calls} - {None}):
        rows = [call for call in calls if call.get("stage") == stage]
        stages[stage] = {
            "calls": len(rows),
            "request_seconds": seconds(rows, "duration_ms"),
            "queue_seconds": seconds(rows, "queue_ms"),
            "completion_tokens": tokens(rows, "completion_tokens"),
        }
    return {
        "wall_seconds": round(wall_seconds, 1),
        "packet_workers": workers,
        "request_seconds": seconds(chat, "duration_ms"),
        "queue_seconds": seconds(calls, "queue_ms"),
        "prompt_tokens": tokens(chat, "prompt_tokens"),
        "completion_tokens": tokens(chat, "completion_tokens"),
        "stages": stages,
    }


class _Budget:
    def __init__(
        self,
        provider: Provider,
        settings: PipelineSettings,
        trace: list[dict],
        event=None,
        issues: list | None = None,
    ):
        self.provider, self.settings, self.trace = provider, settings, trace
        self.limit = settings.max_model_calls
        self.used = 0
        self.event = event
        self.issues = issues

    async def call(
        self, schema, prompt, payload, stage: str, reserve: int = 0
    ):
        for attempt in range(self.settings.max_retries + 1):
            if self.used + reserve >= self.limit:
                raise LLMError(
                    "call_budget", "Document model call budget exhausted"
                )
            self.used += 1
            _emit(
                self.event,
                stage=stage,
                status="running",
                model_calls=self.used,
                max_model_calls=self.limit,
                attempt=attempt + 1,
            )
            try:
                result = await resolve(
                    self.provider.generate(
                        schema, prompt, payload, stage=stage
                    )
                )
                dropped = list(getattr(result, "_dropped_items", None) or [])
                result = schema.model_validate(
                    result.model_dump()
                    if hasattr(result, "model_dump")
                    else result
                )
                if dropped and self.issues is not None:
                    # Malformed elements were dropped, the rest is kept.
                    self.issues.append(
                        {
                            "code": "invalid_items",
                            "stage": stage,
                            "call": self.used,
                            "items": dropped,
                        }
                    )
                self.trace.append(
                    {
                        "stage": stage,
                        "call": self.used,
                        "attempt": attempt + 1,
                        "status": "succeeded",
                    }
                )
                _emit(
                    self.event,
                    stage=stage,
                    status="succeeded",
                    model_calls=self.used,
                )
                return result
            except LLMError as exc:
                self.trace.append(
                    {
                        "stage": stage,
                        "call": self.used,
                        "attempt": attempt + 1,
                        "status": "failed",
                        "code": exc.code,
                    }
                )
                if rate_limited(exc):
                    # Refused before any model work: retries after HTTP 429
                    # must not spend the document's call budget.
                    self.used -= 1
                _emit(
                    self.event,
                    stage=stage,
                    status="failed",
                    model_calls=self.used,
                    code=exc.code,
                )
                if (
                    not exc.retryable
                    or attempt == self.settings.max_retries
                    or self.used + reserve >= self.limit
                ):
                    logger.warning(
                        "LLM %s call failed: %s (%s)", stage, exc.code, exc
                    )
                    raise
                # Exponential backoff unless the provider named a delay:
                # a rate-limited provider needs longer than a fixed pause.
                delay = (
                    exc.retry_after
                    if exc.retry_after is not None
                    else self.settings.retry_delay_seconds * 2**attempt
                )
                delay = min(
                    max(0, delay), self.settings.max_retry_delay_seconds
                )
                logger.warning(
                    "LLM %s attempt %d failed: %s; retrying in %.1fs",
                    stage,
                    attempt + 1,
                    exc.code,
                    delay,
                )
                await _sleep(delay)
            except Exception as exc:
                # A stopping server closes the loop's executor under a
                # request in flight: that is an interruption, not a bad
                # answer of the model.
                interrupted = _shutting_down(exc)
                code = "interrupted" if interrupted else "invalid_response"
                self.trace.append(
                    {
                        "stage": stage,
                        "call": self.used,
                        "status": "failed",
                        "code": code,
                    }
                )
                _emit(
                    self.event,
                    stage=stage,
                    status="failed",
                    model_calls=self.used,
                    code=code,
                )
                if interrupted:
                    logger.info(
                        "LLM %s call interrupted: the process is stopping",
                        stage,
                    )
                    raise LLMError(
                        "interrupted", "The process stopped during the call"
                    ) from None
                logger.warning(
                    "LLM %s returned an invalid response", stage, exc_info=True
                )
                raise LLMError(
                    "invalid_response", "Provider returned an invalid response"
                ) from None


def _shutting_down(exc: BaseException) -> bool:
    """The event loop or its executor was closed under the call."""
    seen: set = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, RuntimeError) and (
            "after shutdown" in str(exc) or "loop is closed" in str(exc)
        ):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


async def _review(
    document,
    packet_id,
    valid,
    visible,
    settings,
    related_context,
    budget,
    prompt,
    decisions,
    metadata,
    trace,
):
    """Review valid claims in groups that fit the payload budget."""
    try:
        groups, unfit = review_batches(
            document,
            valid.model_dump(mode="python"),
            visible,
            settings,
            related_context=related_context,
        )
    except Exception as exc:
        logger.warning(
            "Packet %s review failed: %s",
            packet_id,
            getattr(exc, "code", type(exc).__name__),
        )
        metadata["issues"].append(
            {
                "packet_id": packet_id,
                "code": getattr(exc, "code", "review_contract"),
            }
        )
        return
    for claim_id in unfit:
        logger.warning(
            "Packet %s claim %s does not fit the review budget",
            packet_id,
            claim_id,
        )
        metadata["issues"].append(
            {
                "packet_id": packet_id,
                "claim_id": claim_id,
                "code": "review_payload_budget",
            }
        )
    for claim_ids, payload in groups:
        try:
            trace.append(
                {
                    "stage": "review_context",
                    "packet_id": packet_id,
                    "claim_ids": claim_ids,
                    "context": payload["review_context"],
                }
            )
            review = await budget.call(Review, prompt, payload, "review")
            review = validate_review(review, claim_ids)
        except Exception as exc:
            logger.warning(
                "Packet %s review failed: %s",
                packet_id,
                getattr(exc, "code", type(exc).__name__),
            )
            metadata["issues"].append(
                {
                    "packet_id": packet_id,
                    "claim_ids": claim_ids,
                    "code": getattr(exc, "code", "review_contract"),
                }
            )
            continue
        decisions.update({item.claim_id: item for item in review.items})
        metadata["issues"].extend(
            {
                "packet_id": packet_id,
                "claim_id": item.claim_id,
                "code": "review_unclear",
                "reason": item.reason,
            }
            for item in review.items
            if item.decision == "unclear"
        )
        trace.append(
            {
                "stage": "verification",
                "packet_id": packet_id,
                "items": review.model_dump()["items"],
            }
        )


async def process_document(
    document: DocumentEnvelope,
    provider: Provider,
    registry: Iterable[Concept] | ConceptRegistry = (),
    settings: PipelineSettings | None = None,
    semantic=None,
    event=None,
    context_reader=None,
) -> ExtractionResult:
    """Process one document; its provider calls are audited separately
    even when one provider serves several concurrent documents.
    """
    log: list = []
    token = CALL_LOG.set(log)
    try:
        return await _process_document(
            document,
            provider,
            registry,
            settings,
            semantic,
            event,
            log,
            _semantic_reader(context_reader, semantic),
        )
    finally:
        CALL_LOG.reset(token)


async def _process_document(
    document: DocumentEnvelope,
    provider: Provider,
    registry: Iterable[Concept] | ConceptRegistry,
    settings: PipelineSettings | None,
    semantic,
    event,
    call_log: list,
    context_reader=None,
) -> ExtractionResult:
    """Return an auditable extraction with explicit partial coverage."""
    settings = settings or PipelineSettings.from_catalog()
    clock = perf_counter()
    prompts = {stage: _prompt(stage) for stage in ("extract", "review")}
    trace: list[dict] = []
    metadata: dict = {
        "coverage": {},
        "issues": [],
        "unresolved_claims": [],
        "invalid_entities": [],
        # Non-blocking anchoring choices; see validate_local_extraction.
        "anchoring_notes": [],
        "source_truth_assessed": False,
        "demo": bool(getattr(provider, "demo", False)),
        "source_snapshot": document.artifact.model_dump(),
        "input_coverage": document.coverage,
        # The version stays when a PDF appears; the PDF identifies the input.
        "input_fulltext_sha256": fulltext_sha256(document),
        "input_quality_status": document.quality_status,
        "parse_warnings": document.metadata.get("parse_warnings", []),
    }
    started = datetime.now(timezone.utc).isoformat()
    run = ProcessingRun(
        run_id=stable_id("run", document.document_version_id, uuid4()),
        pipeline_version="material-llm/1",
        parser="llm_packets",
        started_at=started,
        prompt_hash=stable_id("prompts", json_value(prompts)),
        config_hash=stable_id(
            "config",
            json_value(settings.model_dump()),
            json_value(load_catalog("pipeline")),
            json_value(getattr(provider, "models", {})),
            getattr(provider, "base_url", "replay"),
            json_value(load_catalog("llm_schema")),
            json_value(load_catalog("resolver")),
            json_value(load_catalog("llm")),
        ),
        model_revision=json_value(getattr(provider, "models", {})),
        metadata=metadata,
        trace=trace,
    )
    # Pydantic can copy containers on construction; use the actual run
    # containers.
    metadata, trace = run.metadata, run.trace
    budget = _Budget(provider, settings, trace, event, metadata["issues"])
    # Providers that do not use CALL_LOG (test doubles) keep a plain list.
    call_offset = len(getattr(provider, "calls", []))
    event_offset = len(getattr(provider, "model_events", []))
    _emit(event, stage="plan", status="running")
    plan = plan_packets(document, settings)
    budget.limit = settings.call_limit(len(plan.packets))
    logger.info(
        "%s: %d chunks planned into %d packets",
        document.document_version_id,
        len(document.chunks),
        len(plan.packets),
    )
    _emit(
        event,
        stage="plan",
        status="succeeded",
        total_packets=len(plan.packets),
        total_chunks=len(document.chunks),
        omitted_chunks=len(plan.omitted_chunk_ids),
    )
    trace.append(
        {
            "stage": "plan",
            "packets": [p.model_dump() for p in plan.packets],
            "omitted_reasons": plan.omitted_reasons,
            "support_omissions": plan.support_omissions,
        }
    )
    aliases = _ChunkAliases(document)
    # Reference names from the registry per packet (linking.known).
    if not isinstance(registry, (ConceptRegistry, list, tuple)):
        registry = list(registry)
    known = (
        await asyncio.to_thread(
            known_layer.KnownConcepts.for_registry, registry, semantic
        )
        if known_layer.enabled()
        else None
    )
    by_chunk = {chunk.chunk_id: chunk for chunk in document.chunks}

    async def known_for(packet) -> list:
        if not known:
            return []
        found = await asyncio.to_thread(
            known.lookup,
            [by_chunk[chunk_id].text for chunk_id in packet.focus_chunk_ids],
        )
        trace.append(
            {
                "stage": "known_concepts",
                "packet_id": packet.packet_id,
                "concepts": [
                    [item["match"], item["kind"], item["label"]]
                    for item in found
                ],
            }
        )
        return found

    async def extract(payload):
        return aliases.decode(
            await budget.call(
                Extraction,
                prompts["extract"],
                aliases.encode(payload),
                "extract",
                reserve=1,
            )
        )

    # Packets run concurrently when the provider serves several requests
    # (a key pool); results keep the plan order, and the halves of a split
    # packet take its place.
    processed, failed = [], []
    batches = []
    queue = deque(
        ((position,), item) for position, item in enumerate(plan.packets)
    )
    running: set = set()
    workers = _packet_workers(provider, settings)

    async def run_packet(order: tuple, original, packet_number: int) -> None:
        _emit(
            event,
            stage="packet",
            status="running",
            packet_id=original.packet_id,
            packet_number=packet_number,
            total_packets=packet_number + len(queue),
        )
        if budget.used + 2 > budget.limit:
            failed.append((order, original.packet_id))
            metadata["issues"].append(
                {"packet_id": original.packet_id, "code": "call_budget"}
            )
            _emit(
                event,
                stage="packet",
                status="failed",
                packet_id=original.packet_id,
                code="call_budget",
            )
            logger.warning(
                "Packet %s skipped: model call budget exhausted",
                original.packet_id,
            )
            return
        packet = original
        related_context = []
        try:
            known_concepts = await known_for(packet)
            extraction = await extract(
                build_payload(
                    document, packet, settings, known_concepts=known_concepts
                )
            )
            trace.append(
                {
                    "stage": "extraction_response",
                    "packet_id": packet.packet_id,
                    "response": extraction.model_dump(mode="python"),
                }
            )
            # Every later packet keeps its extraction and review calls: an
            # early packet's context rounds must not starve the rest. A
            # budget below that minimum cannot cover all packets anyway.
            reserved = (
                2 * (len(queue) + len(running) - 1)
                if budget.limit >= 2 * len(plan.packets)
                else 0
            )
            for context_round in range(settings.max_context_rounds):
                if (
                    not extraction.context_requests
                    or budget.used + 2 + reserved > budget.limit
                ):
                    break
                _emit(
                    event,
                    stage="context",
                    status="running",
                    packet_id=packet.packet_id,
                    context_round=context_round + 1,
                )
                expanded, expanded_related, outcomes = await expand_context(
                    document,
                    packet,
                    extraction.context_requests,
                    settings,
                    related_context,
                    context_reader,
                )
                trace.append(
                    {
                        "stage": "context",
                        "packet_id": packet.packet_id,
                        "round": context_round + 1,
                        "requests": [
                            r.model_dump() for r in extraction.context_requests
                        ],
                        "outcomes": outcomes,
                        "related_sources": [
                            {
                                key: row[key]
                                for key in (
                                    "chunk_id",
                                    "document_id",
                                    "document_version_id",
                                    "title",
                                )
                            }
                            | {
                                "text_hash": stable_id(
                                    "context-text", row["text"]
                                )
                            }
                            for row in expanded_related
                        ],
                    }
                )
                if (
                    expanded == packet
                    and expanded_related == related_context
                    and not (
                        outcomes
                        and all(
                            o.get("status") in {"already_visible", "no_match"}
                            for o in outcomes
                        )
                    )
                ):
                    break
                try:
                    next_payload = build_payload(
                        document,
                        expanded,
                        settings,
                        feedback=[
                            "Re-read the original requested chunks, "
                            "including already visible blocks; return a "
                            "complete replacement extraction.",
                            json.dumps(outcomes, ensure_ascii=False),
                        ],
                        related_context=expanded_related,
                        known_concepts=known_concepts,
                    )
                except ContextBudgetError:
                    metadata["issues"].append(
                        {
                            "packet_id": packet.packet_id,
                            "code": "context_payload_budget",
                        }
                    )
                    break
                try:
                    extraction_with_context = await extract(next_payload)
                except LLMError as exc:
                    # The first answer is valid and anchored in the packet
                    # it saw; its context request stays unresolved (B-7).
                    metadata["issues"].append(
                        {
                            "packet_id": packet.packet_id,
                            "code": "context_reextraction_failed",
                            "error": exc.code,
                        }
                    )
                    break
                extraction = extraction_with_context
                packet = expanded
                related_context = expanded_related
                trace.append(
                    {
                        "stage": "extraction_response",
                        "packet_id": packet.packet_id,
                        "response": extraction.model_dump(mode="python"),
                    }
                )
            visible = set(packet.focus_chunk_ids + packet.support_chunk_ids)
            _emit(
                event,
                stage="validate",
                status="running",
                packet_id=packet.packet_id,
            )
            anchoring: list = []
            extraction, issues = validate_local_extraction(
                document, extraction, visible, notes=anchoring
            )
            metadata["anchoring_notes"].extend(
                {"packet_id": packet.packet_id, **note} for note in anchoring
            )
            _emit(
                event,
                stage="validate",
                status="succeeded",
                packet_id=packet.packet_id,
                validation_issues=len(issues),
            )
            metadata["issues"].extend(
                {
                    "packet_id": packet.packet_id,
                    "item": key,
                    "reasons": reasons,
                }
                for key, reasons in issues.items()
            )
            metadata["unresolved_claims"].extend(
                {
                    "packet_id": packet.packet_id,
                    "claim": c.model_dump(),
                    "reason": issues[f"claim:{c.claim_id}"],
                }
                for c in extraction.claims
                if f"claim:{c.claim_id}" in issues
            )
            metadata["invalid_entities"].extend(
                {
                    "packet_id": packet.packet_id,
                    "entity": e.model_dump(),
                    "reason": issues[f"entity:{e.local_id}"],
                }
                for e in extraction.entities
                if f"entity:{e.local_id}" in issues
            )
            valid_entities = [
                e
                for e in extraction.entities
                if f"entity:{e.local_id}" not in issues
            ]
            valid_claims = [
                c
                for c in extraction.claims
                if f"claim:{c.claim_id}" not in issues
            ]
            valid = Extraction(entities=valid_entities, claims=valid_claims)
            trace.append(
                {
                    "stage": "validation",
                    "packet_id": packet.packet_id,
                    "valid_entities": len(valid_entities),
                    "valid_claims": len(valid_claims),
                    "issues": issues,
                }
            )
            decisions = {}
            context_pending = _pending_claims(extraction.context_requests)
            if extraction.context_requests:
                metadata["issues"].append(
                    {
                        "packet_id": packet.packet_id,
                        "code": "unresolved_context",
                        "requests": [
                            r.model_dump() for r in extraction.context_requests
                        ],
                    }
                )
            if valid_claims:
                await _review(
                    document,
                    packet.packet_id,
                    valid,
                    visible,
                    settings,
                    related_context,
                    budget,
                    prompts["review"],
                    decisions,
                    metadata,
                    trace,
                )
            batches.append(
                (order, (packet.packet_id, valid, decisions, context_pending))
            )
            processed.extend(original.focus_chunk_ids)
            _emit(
                event,
                stage="packet",
                status="succeeded",
                packet_id=packet.packet_id,
                packet_number=packet_number,
                total_packets=packet_number + len(queue),
            )
        except Exception as exc:
            halves = (
                split_packet(document, original, settings)
                if getattr(exc, "code", None) == "incomplete_response"
                else []
            )
            if halves:
                # The answer hit the output limit: retry in two halves
                # instead of losing the whole packet (B-6).
                queue.extendleft(
                    reversed(
                        [
                            (order + (half,), item)
                            for half, item in enumerate(halves)
                        ]
                    )
                )
                metadata["issues"].append(
                    {
                        "packet_id": original.packet_id,
                        "code": "incomplete_response",
                        "split_into": [item.packet_id for item in halves],
                    }
                )
                logger.info(
                    "Packet %s hit the output limit; split in two",
                    original.packet_id,
                )
                return
            logger.warning(
                "Packet %s failed: %s",
                original.packet_id,
                getattr(exc, "code", type(exc).__name__),
            )
            logger.debug(
                "Packet %s traceback", original.packet_id, exc_info=True
            )
            failed.append((order, original.packet_id))
            metadata["issues"].append(
                {
                    "packet_id": original.packet_id,
                    "code": getattr(exc, "code", type(exc).__name__),
                }
            )
            _emit(
                event,
                stage="packet",
                status="failed",
                packet_id=original.packet_id,
                code=getattr(exc, "code", type(exc).__name__),
            )

    packet_number = 0
    try:
        while queue or running:
            while queue and len(running) < workers:
                order, original = queue.popleft()
                packet_number += 1
                running.add(
                    asyncio.create_task(
                        run_packet(order, original, packet_number)
                    )
                )
            done, _ = await asyncio.wait(
                running, return_when=asyncio.FIRST_COMPLETED
            )
            running.difference_update(done)
            for task in done:
                task.result()
    finally:
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
    batches = [batch for _, batch in sorted(batches, key=lambda i: i[0])]
    failed = [packet_id for _, packet_id in sorted(failed)]

    # Assemble only source-anchored entities; matching names alone does not
    # dedup evidence.
    _emit(event, stage="assemble", status="running")
    mentions: dict[str, Mention] = {}
    chunks = {chunk.chunk_id: chunk for chunk in document.chunks}
    entity_mentions: dict[str, list[str]] = {}
    definitions: dict[str, str] = {}
    claims = []
    for packet_id, extraction, decisions, pending in batches:
        for entity in extraction.entities:
            key = f"{packet_id}:{entity.local_id}"
            # The ISO code, not a language-specific name, identifies a
            # country across documents and links it to metadata countries.
            canonical = (
                entity.country_code
                if entity.kind == ConceptKind.COUNTRY and entity.country_code
                else entity.label
            )
            ids = []
            for span in entity.evidence:
                mention_id = stable_id(
                    "mention",
                    document.document_version_id,
                    json_value(_source_key(chunks, span)),
                    canonical,
                    entity.kind.value,
                )
                mention = Mention(
                    mention_id=mention_id,
                    chunk_id=span.chunk_id,
                    surface_text=span.quote,
                    canonical_text=canonical,
                    start=span.start,
                    end=span.end,
                    type_candidates=[entity.kind],
                    mention_role="entity",
                    definition=entity.definition,
                    declared_aliases=entity.aliases,
                )
                mentions.setdefault(mention_id, mention)
                ids.append(mention_id)
            entity_mentions[key] = ids
            if entity.definition:
                definitions[key] = entity.definition
        for claim in extraction.claims:
            mapping = {
                e.local_id: f"{packet_id}:{e.local_id}"
                for e in extraction.entities
            }
            data = claim.model_dump()
            data["roles"] = {
                role: mapping[ref] for role, ref in claim.roles.items()
            }
            data["qualifiers"] = _refs(claim.qualifiers, mapping)
            data["values"] = _refs(claim.values, mapping)
            decision = decisions.get(claim.claim_id)
            waiting = pending is True or claim.claim_id in pending
            state = (
                decision.decision if decision and not waiting else "unclear"
            )
            claims.append(
                (
                    data,
                    state,
                    decision.reason if decision else "Review unavailable",
                )
            )
    _emit(event, stage="resolution", status="running", mentions=len(mentions))
    if isinstance(registry, ConceptRegistry):
        # Shared job registry: resolution is serialized so concurrent
        # documents see each other's new concepts, without a per-document
        # copy of the whole registry.
        concepts, resolutions = await registry.resolve(
            list(mentions.values()), semantic, semantic_candidates=True
        )
    else:
        concepts, resolutions = await asyncio.to_thread(
            resolve_mentions,
            list(mentions.values()),
            deepcopy(list(registry)),
            semantic,
            True,
        )
    mention_concept = {
        d.mention_id: d.concept_id
        for d in resolutions
        if d.status in {"accepted", "provisional"} and d.concept_id
    }
    identities = {}
    for entity_key, ids in entity_mentions.items():
        resolved = {mention_concept.get(mid) for mid in ids}
        if len(resolved) == 1 and None not in resolved:
            identities[entity_key] = resolved.pop()
    metadata["entity_bindings"] = {
        key: {
            "mention_ids": ids,
            "concept_id": identities.get(key),
            "local_definition": definitions.get(key),
        }
        for key, ids in entity_mentions.items()
    }
    assertions: dict[str, Assertion] = {}
    conflicted = set()
    for data, state, reason in claims:
        try:
            roles = {
                role: identities[ref] for role, ref in data["roles"].items()
            }
            qualifiers, values = (
                _refs(data["qualifiers"], identities),
                _refs(data["values"], identities),
            )
        except KeyError:
            metadata["unresolved_claims"].append(
                {
                    "claim": data,
                    "reason": "Entity identity ambiguous",
                    "review": state,
                }
            )
            continue
        evidence = [EvidenceSpan(**span) for span in data["evidence"]]
        identity = {
            key: data[key]
            for key in (
                "predicate",
                "polarity",
                "modality",
                "attribution_kind",
            )
        }
        identity.update(
            roles=roles,
            qualifiers=qualifiers,
            values=values,
            evidence=sorted(
                [
                    {"source": _source_key(chunks, s), "quote": s.quote}
                    for s in evidence
                ],
                key=json_value,
            ),
        )
        aid = stable_id(
            "assertion", document.document_version_id, json_value(identity)
        )
        assertion = Assertion(
            assertion_id=aid,
            predicate=data["predicate"],
            roles=roles,
            qualifiers=qualifiers,
            values=values,
            evidence=evidence,
            polarity=data["polarity"],
            modality=data["modality"],
            attribution_kind=data["attribution_kind"],
            verification_status={
                "supported": "supported",
                "unsupported": "unsupported",
                "unclear": "unverified",
            }[state],
            status={
                "supported": "accepted",
                "unsupported": "rejected",
                "unclear": "needs_review",
            }[state],
        )
        previous = assertions.get(aid)
        if previous:
            assertion.evidence = previous.evidence
        if (
            aid in conflicted
            or previous
            and previous.verification_status != assertion.verification_status
        ):
            conflicted.add(aid)
            assertion.status, assertion.verification_status = (
                "needs_review",
                "unverified",
            )
            metadata["issues"].append(
                {"assertion_id": aid, "code": "conflicting_reviews"}
            )
        assertions[aid] = assertion
        trace.append(
            {
                "stage": "assemble",
                "assertion_id": aid,
                "review": state,
                "reason": reason,
            }
        )
    covered = set(processed)
    # Data and administrative sections were read by the plan and skipped on
    # purpose (linking.sections); they do not make the run partial.
    skipped_sections = sorted(
        chunk_id
        for chunk_id, reason in plan.omitted_reasons.items()
        if reason.startswith(SKIPPED_REASON)
    )
    decided = covered | set(skipped_sections)
    metadata["coverage"] = {
        "total_chunks": len(document.chunks),
        "processed_focus_chunk_ids": sorted(covered),
        "skipped_section_chunk_ids": skipped_sections,
        "unprocessed_chunk_ids": [
            c.chunk_id for c in document.chunks if c.chunk_id not in decided
        ],
        "omitted_chunk_ids": plan.omitted_chunk_ids,
        "failed_packet_ids": failed,
    }
    metadata["budgets"] = {
        **settings.model_dump(),
        "effective_model_calls": budget.limit,
    }
    metadata["model_calls"] = budget.used
    # A document without calls of its own must not take the calls that a
    # concurrent document made meanwhile (B-9); the plain-list fallback is
    # only for providers that do not report into CALL_LOG.
    metadata["provider_calls"] = deepcopy(
        call_log
        if call_log or not budget.used
        else list(getattr(provider, "calls", []))[call_offset:]
    )
    # Model retirements are provider-wide; report those seen meanwhile.
    metadata["model_events"] = deepcopy(
        list(getattr(provider, "model_events", []))[event_offset:]
    )
    trace.append(
        {
            "stage": "resolution",
            "accepted": sum(d.status == "accepted" for d in resolutions),
            "provisional": sum(d.status == "provisional" for d in resolutions),
            "ambiguous": sum(d.status == "ambiguous" for d in resolutions),
        }
    )
    # Rejected or unclear items are already excluded or stored as
    # needs_review, and projections use accepted assertions only; they do not
    # make the document's coverage incomplete. Packet-level gaps do.
    blocking = [item for item in metadata["issues"] if not _item_issue(item)]
    metadata["item_issue_count"] = len(metadata["issues"]) - len(blocking)
    # Nothing to read is not a failed extraction: no model was called and
    # a later text of this version must still be processed (A-10).
    run.status = (
        "skipped_no_text"
        if not document.chunks
        else "succeeded"
        if len(decided) == len(document.chunks) and not blocking
        else ("partial" if covered else "failed")
    )
    embeddings, embedding_model = await _concept_embeddings(
        concepts, resolutions, semantic, metadata
    )
    economic = extract_economic_evidence(
        document.chunks, list(mentions.values()), concepts, resolutions
    ) + economic_evidence_from_assertions(
        document, list(assertions.values()), concepts
    )
    chunk_embeddings = await _evidence_embeddings(
        document, list(assertions.values()), economic, semantic, metadata
    )
    result = ExtractionResult(
        document_version_id=document.document_version_id,
        run=run,
        mentions=list(mentions.values()),
        concepts=concepts,
        resolutions=resolutions,
        assertions=list(assertions.values()),
        economic_evidence=economic,
        concept_embeddings=embeddings,
        chunk_embeddings=chunk_embeddings,
        embedding_model=embedding_model,
    )
    validate_extraction(document, result)
    timing = _timing(
        list(metadata["provider_calls"]), perf_counter() - clock, workers
    )
    metadata["timing"] = timing
    logger.info(
        "%s: run %s, chunks %d/%d, model calls %d, issues %d; "
        "%.1fs total: model requests %.1fs, waiting for a key %.1fs, "
        "%d packets x %d at once, out %d tok",
        document.document_version_id,
        run.status,
        len(covered),
        len(document.chunks),
        budget.used,
        len(metadata["issues"]),
        timing["wall_seconds"],
        timing["request_seconds"],
        timing["queue_seconds"],
        len(plan.packets),
        workers,
        timing["completion_tokens"],
    )
    _emit(
        event,
        stage="done",
        status=run.status,
        mentions=len(result.mentions),
        assertions=len(result.assertions),
        model_calls=budget.used,
        processed_chunks=len(covered),
        total_chunks=len(document.chunks),
    )
    return result
