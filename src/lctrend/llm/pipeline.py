"""Bounded document processing: packets, extraction, review, resolution.

The coordinator reads bounded original context through an injected reader.
The caller publishes its validated result to Neo4j.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
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
from ..extraction.resolver import ConceptRegistry, resolve_mentions
from .client import CALL_LOG, LLMError, Provider
from .context import (
    ContextBudgetError,
    PipelineSettings,
    build_payload,
    expand_context,
    plan_packets,
    review_batches,
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
            embed, [concept.preferred_label for concept in chosen]
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
    return {
        concept.concept_id: vector for concept, vector in zip(chosen, vectors)
    }, model


class _Budget:
    def __init__(
        self,
        provider: Provider,
        settings: PipelineSettings,
        trace: list[dict],
        event=None,
    ):
        self.provider, self.settings, self.trace = provider, settings, trace
        self.limit = settings.max_model_calls
        self.used = 0
        self.event = event

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
                result = schema.model_validate(
                    result.model_dump()
                    if hasattr(result, "model_dump")
                    else result
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
            except Exception:
                self.trace.append(
                    {
                        "stage": stage,
                        "call": self.used,
                        "status": "failed",
                        "code": "invalid_response",
                    }
                )
                _emit(
                    self.event,
                    stage=stage,
                    status="failed",
                    model_calls=self.used,
                    code="invalid_response",
                )
                logger.warning(
                    "LLM %s returned an invalid response", stage, exc_info=True
                )
                raise LLMError(
                    "invalid_response", "Provider returned an invalid response"
                ) from None


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
            context_reader,
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
    budget = _Budget(provider, settings, trace, event)
    # Providers that do not use CALL_LOG (test doubles) keep a plain list.
    call_offset = len(getattr(provider, "calls", []))
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
    processed, failed = [], []
    batches = []
    for packet_number, original in enumerate(plan.packets, 1):
        _emit(
            event,
            stage="packet",
            status="running",
            packet_id=original.packet_id,
            packet_number=packet_number,
            total_packets=len(plan.packets),
        )
        if budget.used + 2 > budget.limit:
            failed.append(original.packet_id)
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
            continue
        packet = original
        related_context = []
        try:
            extraction = await budget.call(
                Extraction,
                prompts["extract"],
                build_payload(document, packet, settings),
                "extract",
                reserve=1,
            )
            trace.append(
                {
                    "stage": "extraction_response",
                    "packet_id": packet.packet_id,
                    "response": extraction.model_dump(mode="python"),
                }
            )
            for context_round in range(settings.max_context_rounds):
                if (
                    not extraction.context_requests
                    or budget.used + 2 > budget.limit
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
                    )
                except ContextBudgetError:
                    metadata["issues"].append(
                        {
                            "packet_id": packet.packet_id,
                            "code": "context_payload_budget",
                        }
                    )
                    break
                packet = expanded
                related_context = expanded_related
                extraction = await budget.call(
                    Extraction,
                    prompts["extract"],
                    next_payload,
                    "extract",
                    reserve=1,
                )
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
            context_pending = bool(extraction.context_requests)
            if context_pending:
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
                (packet.packet_id, valid, decisions, context_pending)
            )
            processed.extend(original.focus_chunk_ids)
            _emit(
                event,
                stage="packet",
                status="succeeded",
                packet_id=packet.packet_id,
                packet_number=packet_number,
                total_packets=len(plan.packets),
            )
        except Exception as exc:
            logger.warning(
                "Packet %s failed: %s",
                original.packet_id,
                getattr(exc, "code", type(exc).__name__),
            )
            logger.debug(
                "Packet %s traceback", original.packet_id, exc_info=True
            )
            failed.append(original.packet_id)
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
            state = (
                decision.decision if decision and not pending else "unclear"
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
    metadata["coverage"] = {
        "total_chunks": len(document.chunks),
        "processed_focus_chunk_ids": sorted(covered),
        "unprocessed_chunk_ids": [
            c.chunk_id for c in document.chunks if c.chunk_id not in covered
        ],
        "omitted_chunk_ids": plan.omitted_chunk_ids,
        "failed_packet_ids": failed,
    }
    metadata["budgets"] = {
        **settings.model_dump(),
        "effective_model_calls": budget.limit,
    }
    metadata["model_calls"] = budget.used
    metadata["provider_calls"] = deepcopy(
        call_log or list(getattr(provider, "calls", []))[call_offset:]
    )
    metadata["model_events"] = deepcopy(getattr(provider, "model_events", []))
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
    run.status = (
        "succeeded"
        if len(covered) == len(document.chunks)
        and document.chunks
        and not blocking
        else ("partial" if covered else "failed")
    )
    embeddings, embedding_model = await _concept_embeddings(
        concepts, resolutions, semantic, metadata
    )
    result = ExtractionResult(
        document_version_id=document.document_version_id,
        run=run,
        mentions=list(mentions.values()),
        concepts=concepts,
        resolutions=resolutions,
        assertions=list(assertions.values()),
        economic_evidence=extract_economic_evidence(
            document.chunks, list(mentions.values()), concepts, resolutions
        )
        + economic_evidence_from_assertions(
            document, list(assertions.values()), concepts
        ),
        concept_embeddings=embeddings,
        embedding_model=embedding_model,
    )
    validate_extraction(document, result)
    logger.info(
        "%s: run %s, chunks %d/%d, model calls %d, issues %d",
        document.document_version_id,
        run.status,
        len(covered),
        len(document.chunks),
        budget.used,
        len(metadata["issues"]),
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
