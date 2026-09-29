"""The weak-signal search on its own: ``GET /api/search`` and health.

The light search image runs only this module: it reads the shared Neo4j
(``NEO4J_*``) and never writes, so it needs no ingestion stack. A GigaChat
key (``GIGACHAT_*``) adds the semantic half of the hybrid search; without
it the search answers by BM25 and says so. The full server (``app.py``)
mounts the same route.

    python -m frontend.server.search_api --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date
from typing import Iterable, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.trustedhost import TrustedHostMiddleware

logger = logging.getLogger(__name__)

LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]")


def allowed_hosts(configured: Optional[Iterable[str]] = None) -> list:
    """Local names, the public domain (``LCTREND_DOMAIN``) and any extra
    names in ``LCTREND_ALLOWED_HOSTS``."""
    if configured is None:
        configured = [
            item.strip()
            for item in (
                os.getenv("LCTREND_DOMAIN", "")
                + ","
                + os.getenv("LCTREND_ALLOWED_HOSTS", "")
            ).split(",")
            if item.strip()
        ]
    return list(dict.fromkeys([*LOCAL_HOSTS, *configured]))


def _store():
    from lctrend.session import open_store

    return open_store()


async def read_graph() -> dict:
    """The whole dated graph for the search ranking."""
    async with _store() as store:
        return await store.read_temporal_data()


async def read_labels() -> dict:
    """Model scores and LLM labels on Technology nodes (05_graph)."""
    async with _store() as store:
        return await store.read_technology_labels()


def add_search_route(app: FastAPI) -> None:
    """``GET /api/search`` served by ``app.state.search`` (created on the
    first request unless a test supplied one)."""

    @app.get("/api/search")
    async def search_signals(
        q: str = Query(max_length=200),
        snapshot: Optional[str] = Query(default=None, alias="date"),
    ):
        query = q.strip()
        if not query:
            raise HTTPException(422, "Введите запрос")
        try:
            cutoff = date.fromisoformat(snapshot) if snapshot else None
        except ValueError:
            raise HTTPException(422, "date: ожидается YYYY-MM-DD") from None
        if getattr(app.state, "search", None) is None:
            from lctrend.ranking.search import SearchService

            app.state.search = SearchService(
                read_graph, read_labels=read_labels
            )
        try:
            return await app.state.search.search(query, cutoff)
        except Exception as exc:
            # Connection errors name hosts; keep them in the server log.
            logger.exception("Search %r failed", query)
            from lctrend.llm.client import LLMError
            from lctrend.ranking.search import EmbeddingIndexError

            if isinstance(exc, LLMError):
                detail = "GigaChat недоступен: проверьте ключ и подключение"
            elif isinstance(exc, EmbeddingIndexError):
                detail = "Эмбеддинги технологий несовместимы с поиском"
            else:
                detail = "Граф недоступен: проверьте подключение к Neo4j"
            raise HTTPException(503, detail) from None


def create_app(search_service=None, hosts=None) -> FastAPI:
    app = FastAPI(title="LCTrend: поиск слабых сигналов")
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=allowed_hosts(hosts)
    )
    app.state.search = search_service
    add_search_route(app)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    return app


def main(argv=None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Поиск слабых сигналов")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.getenv("LCTREND_LOG_LEVEL", "INFO"))
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
