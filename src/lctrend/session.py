"""Graph session: opening Neo4j and the reads every entry point shares.

The CLI, the modeling levels and the servers all open the same store from
``NEO4J_*`` and read the same dated corpus. They import it from here, so
the library never depends on the CLI.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from .core.aio import resolve
from .core.config import load_catalog, load_environment
from .graph.temporal import TemporalCorpus

if TYPE_CHECKING:
    from .graph.store import GraphStore

logger = logging.getLogger(__name__)


def open_store() -> "GraphStore":
    """A store for the configured ``NEO4J_URI``, ``NEO4J_USER``,
    ``NEO4J_PASSWORD``."""
    # Resolved per call: the Neo4j driver loads only when a store opens.
    from .graph.store import GraphStore

    load_environment()
    return GraphStore(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "change-me-now"),
    )


@asynccontextmanager
async def opened(store):
    """Open a graph store; offline test doubles may be synchronous."""
    if hasattr(store, "__aenter__"):
        async with store as entered:
            yield entered
    else:
        with store as entered:
            yield entered


async def with_graph(action, store_factory=open_store):
    """Run ``await action(store)`` on a freshly opened store."""
    async with opened(store_factory()) as store:
        return await action(store)


async def temporal_corpus(store, as_known=False) -> TemporalCorpus:
    return TemporalCorpus(
        await resolve(store.read_temporal_data()), as_known=as_known
    )


async def embed_concepts(store, force=False):
    """Backfill label vectors, e.g. for documents processed while the
    embedding endpoint was unavailable. A vector is dated by its concept's
    first appearance in snapshots, so a late backfill changes no history.
    """
    from .extraction.processing import _semantic_deduplicator
    from .extraction.resolver import context_text

    semantic = _semantic_deduplicator()
    model = semantic.embedding_model_name
    kinds = list(load_catalog("resolver")["semantic"]["embedded_kinds"])
    pending = await resolve(store.read_concepts_to_embed(kinds, model, force))
    written = 0
    for start in range(0, len(pending), semantic.batch_size):
        batch = pending[start : start + semantic.batch_size]
        texts = [
            context_text(row["label"], row.get("definition")) for row in batch
        ]
        vectors = await asyncio.to_thread(semantic.embed, texts)
        if vectors is None:
            raise RuntimeError(
                "Embedding endpoint unavailable "
                f"({semantic.failure}); {written} vectors written"
            )
        await resolve(
            store.write_concept_embeddings(
                [
                    {**row, "vector": vector, "text": text}
                    for row, vector, text in zip(batch, vectors, texts)
                ],
                model,
            )
        )
        written += len(batch)
        logger.info("Embedded %d/%d concepts", written, len(pending))
    return {"model": model, "pending": len(pending), "embedded": written}
