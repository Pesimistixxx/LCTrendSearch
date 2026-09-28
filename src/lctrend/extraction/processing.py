"""Shared extraction entry point for CLI and the ingestion web interface."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

from ..core.config import load_catalog
from ..core.models import ExtractionResult, ProcessingRun, stable_id
from ..llm.pipeline import process_document
from .resolver import SemanticDeduplicator

logger = logging.getLogger(__name__)


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
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_SCOPE",
    "GIGACHAT_BASE_URL",
    "GIGACHAT_CA_BUNDLE_FILE",
    "LLM_BASE_URL",
    "LLM_API_KEY",
)


def _llm_semantic() -> SemanticDeduplicator | None:
    """The semantic layer for LLM runs, shared by the process so its
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
    key = stable_id(
        "semantic-config",
        load_catalog("resolver"),
        *(os.getenv(name) for name in SEMANTIC_ENV),
    )
    if key not in _SHARED_SEMANTIC:
        _SHARED_SEMANTIC.clear()
        _SHARED_SEMANTIC[key] = _semantic_deduplicator()
    return _SHARED_SEMANTIC[key]


async def seed_semantic(store) -> None:
    """Fill the process-wide label-vector cache from the graph once."""
    semantic = _llm_semantic()
    reader = getattr(store, "read_label_vectors", None)
    if (
        semantic is None
        or reader is None
        or semantic.seeded_model == semantic.embedding_model_name
    ):
        return
    kinds = list(load_catalog("resolver")["semantic"]["embedded_kinds"])
    try:
        rows = await reader(kinds, semantic.embedding_model_name)
    except Exception as exc:
        # A cold cache only costs embedding requests, never a job.
        logger.warning("Label vectors not preloaded (%s)", type(exc).__name__)
        return
    added = semantic.seed(rows)
    logger.info("Semantic cache preloaded with %d label vectors", added)


async def process_material(
    document,
    mode="llm",
    provider=None,
    registry=(),
    event=None,
    semantic=None,
    context_reader=None,
) -> ExtractionResult:
    """Use identical extraction semantics and contracts in CLI and web."""
    if mode not in {"llm", "none"}:
        raise ValueError("mode must be llm or none")
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
    # Registry names are context hints, not document-local identity evidence.
    # Even economic records need full extraction: a known skill must not
    # hide new approaches or disambiguation elsewhere in the same record.
    if provider is None:
        from ..llm.client import JsonLLM

        provider = JsonLLM.from_environment()
    return await process_document(
        document,
        provider,
        registry,
        semantic=semantic if semantic is not None else _llm_semantic(),
        event=event,
        context_reader=context_reader,
    )
