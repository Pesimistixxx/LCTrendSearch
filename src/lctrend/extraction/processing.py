"""Shared extraction entry point for CLI and the ingestion web interface."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from ..core.config import load_catalog
from ..core.models import ExtractionResult, ProcessingRun, stable_id
from ..llm.pipeline import process_document
from .assertions import extract_assertions
from .economics import extract_economic_evidence
from .ner import extract_mentions
from .resolver import (
    ConceptRegistry,
    SemanticDeduplicator,
    resolve_mentions,
)


class NerRuntime:
    """Reuse one model and serialize inference across web jobs."""

    def __init__(self, model=None, model_name=None):
        self.model = model
        self.model_name = model_name or load_catalog("runtime")["ner_model"]
        self._lock = RLock()

    def prepare(self):
        with self._lock:
            if self.model is None:
                from gliner import GLiNER

                self.model = GLiNER.from_pretrained(self.model_name)
        return self

    def predict_entities(self, *args, **kwargs):
        with self._lock:
            self.prepare()
            return self.model.predict_entities(*args, **kwargs)


async def _extract(
    document, model_name: str, registry, semantic: SemanticDeduplicator, model
) -> ExtractionResult:
    mentions = await asyncio.to_thread(extract_mentions, document, model)
    if isinstance(registry, ConceptRegistry):
        concepts, resolutions = await registry.resolve(mentions, semantic)
    else:
        concepts, resolutions = await asyncio.to_thread(
            resolve_mentions, mentions, registry, semantic
        )
    economic_evidence = extract_economic_evidence(
        document.chunks, mentions, concepts, resolutions
    )
    started_at = datetime.now(timezone.utc).isoformat()
    return ExtractionResult(
        document_version_id=document.document_version_id,
        run=ProcessingRun(
            run_id=stable_id(
                "run", document.document_version_id, model_name, started_at
            ),
            parser="gliner",
            model_revision=model_name,
            config_hash=stable_id(
                "config",
                model_name,
                load_catalog("extraction"),
                load_catalog("resolver"),
            ),
            started_at=started_at,
        ),
        mentions=mentions,
        concepts=concepts,
        resolutions=resolutions,
        economic_evidence=economic_evidence,
        assertions=extract_assertions(
            document.chunks, mentions, concepts, resolutions
        ),
    )


def _semantic_deduplicator() -> SemanticDeduplicator:
    settings = load_catalog("resolver")["semantic"]
    return SemanticDeduplicator(
        cosine_threshold=float(
            os.getenv("DEDUP_COSINE_THRESHOLD", settings["cosine_threshold"])
        ),
        decision_threshold=float(
            os.getenv(
                "DEDUP_DECISION_THRESHOLD", settings["decision_threshold"]
            )
        ),
        embedding_provider=os.getenv("DEDUP_EMBEDDING_PROVIDER") or None,
        embedding_model=os.getenv("DEDUP_EMBEDDING_MODEL") or None,
        decision_model=os.getenv(
            "DEDUP_DECISION_MODEL", settings["decision_model"]
        ),
    )


_SHARED_SEMANTIC: dict = {}
SEMANTIC_ENV = (
    "DEDUP_EMBEDDING_PROVIDER",
    "DEDUP_EMBEDDING_MODEL",
    "DEDUP_COSINE_THRESHOLD",
    "DEDUP_DECISION_THRESHOLD",
    "DEDUP_DECISION_MODEL",
)


def _llm_semantic() -> SemanticDeduplicator | None:
    """The semantic layer for llm/hybrid runs, shared by the process so its
    label-vector cache spans documents. DEDUP_IN_LLM=0/1 overrides
    resolver.json semantic.use_in_llm.
    """
    flag = os.getenv("DEDUP_IN_LLM")
    enabled = (
        flag.strip().lower() in {"1", "true", "yes"}
        if flag
        else bool(load_catalog("resolver")["semantic"].get("use_in_llm"))
    )
    if not enabled:
        return None
    key = tuple(os.getenv(name) for name in SEMANTIC_ENV)
    if key not in _SHARED_SEMANTIC:
        _SHARED_SEMANTIC.clear()
        _SHARED_SEMANTIC[key] = _semantic_deduplicator()
    return _SHARED_SEMANTIC[key]


async def process_material(
    document,
    mode="hybrid",
    provider=None,
    ner_model=None,
    model_name=None,
    registry=(),
    event=None,
    ner_runtime=None,
    semantic=None,
) -> ExtractionResult:
    """Use identical extraction semantics and contracts in CLI and web."""
    if mode not in {"hybrid", "llm", "gliner", "none"}:
        raise ValueError("mode must be hybrid, llm, gliner or none")
    if mode == "none":
        return ExtractionResult(
            document_version_id=document.document_version_id,
            run=ProcessingRun(
                run_id=stable_id("run", document.document_version_id, uuid4()),
                parser="metadata",
                started_at=datetime.now(timezone.utc).isoformat(),
                config_hash=stable_id("config", "metadata"),
                metadata={"extraction": "disabled"},
            ),
        )
    model = ner_runtime if ner_runtime is not None else ner_model
    name = model_name or getattr(model, "model_name", None)
    if mode in {"llm", "hybrid"}:
        if provider is None:
            from ..llm.client import JsonLLM

            provider = JsonLLM.from_environment()
        return await process_document(
            document,
            provider,
            registry,
            semantic=semantic if semantic is not None else _llm_semantic(),
            ner=model if mode == "hybrid" else None,
            ner_name=name if mode == "hybrid" and model is not None else None,
            event=event,
        )
    model = model if model is not None else NerRuntime(model_name=name)
    return await _extract(
        document,
        name or getattr(model, "model_name", None),
        registry,
        semantic if semantic is not None else _semantic_deduplicator(),
        model,
    )
