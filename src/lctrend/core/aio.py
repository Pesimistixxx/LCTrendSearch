"""Small asyncio helpers shared by the pipeline, CLI and web server.

Network work (LLM, Neo4j, source APIs) is asynchronous. CPU-bound work
(Docling, GLiNER, entity resolution) and injected synchronous callables run
in worker threads so they never block the event loop.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


async def resolve(value: Any) -> Any:
    """Await a value if it is awaitable (sync or async implementations)."""
    while inspect.isawaitable(value):
        value = await value
    return value


def _is_async(function: Callable) -> bool:
    target = getattr(function, "__func__", function)
    return inspect.iscoroutinefunction(target) or inspect.iscoroutinefunction(
        getattr(target, "__call__", None)
    )


async def call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a sync or async callable without blocking the running loop.

    Coroutine functions are awaited directly; plain callables run in a worker
    thread, and an awaitable they return is awaited afterwards.
    """
    if _is_async(function):
        return await resolve(function(*args, **kwargs))
    return await resolve(await asyncio.to_thread(function, *args, **kwargs))


def run_sync(awaitable: Awaitable[T]) -> T:
    """Run a coroutine from synchronous code.

    Inside a worker thread (no running loop) a private loop is used. Calling
    it from a thread that already runs a loop would deadlock, so that is an
    explicit error.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_wrap(awaitable))
    raise RuntimeError("run_sync() cannot be called from a running loop")


async def _wrap(awaitable: Awaitable[T]) -> T:
    return await awaitable


class LoopThread:
    """A dedicated event loop in a daemon thread.

    Synchronous callers (FastAPI sync endpoints, the crawl thread, tests)
    submit coroutines and receive ``concurrent.futures.Future`` objects.
    """

    def __init__(self, name: str = "lctrend-loop") -> None:
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name=name, daemon=True
        )
        self._thread.start()
        self._ready.wait()

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(self._ready.set)
        self.loop.run_forever()

    def submit(self, coroutine: Awaitable[T]):
        return asyncio.run_coroutine_threadsafe(_wrap(coroutine), self.loop)

    def run(self, coroutine: Awaitable[T], timeout: Optional[float] = None):
        if threading.current_thread() is self._thread:
            raise RuntimeError("LoopThread.run() called from its own loop")
        return self.submit(coroutine).result(timeout)

    def stop(self, wait: bool = False) -> None:
        if self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        if wait:
            self._thread.join()
