"""Process-wide LLM timing for server monitoring.

Every finished call feeds rolling statistics per key and stage: how long
requests take, how long they waited for a free key, output speed and
errors. A summary line is logged periodically and /api/llm/stats serves
the same numbers. Only numbers and names are kept, never prompts or keys.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import Counter, deque
from time import monotonic, time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
# Calls kept for percentiles and the recent throughput window.
WINDOW_CALLS = 2000
RECENT_SECONDS = 300.0


def _report_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("LCTREND_LLM_STATS_SECONDS", "60")))
    except ValueError:
        return 60.0


def _percentile(values: list, share: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    return int(ordered[min(len(ordered) - 1, int(share * len(ordered)))])


class LLMStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = time()
        self.calls: deque = deque(maxlen=WINDOW_CALLS)
        self.totals: Dict[str, Counter] = {}
        self.capacity: Dict[str, int] = {}
        self.in_flight: Counter = Counter()
        self.waiting = 0
        self._reported = monotonic()
        self._reported_calls = 0

    def reset(self) -> None:
        self.__init__()

    def register(self, key: str, capacity: int) -> None:
        with self._lock:
            self.capacity[key] = capacity

    def wait(self, delta: int) -> None:
        """Requests waiting for a free key slot."""
        with self._lock:
            self.waiting = max(0, self.waiting + delta)

    def begin(self, key: str) -> None:
        with self._lock:
            self.in_flight[key] += 1

    def end(self, key: str) -> None:
        with self._lock:
            self.in_flight[key] = max(0, self.in_flight[key] - 1)

    def record(self, call: Dict[str, Any]) -> None:
        if call.get("status") == "started":
            return
        tokens = call.get("tokens") or {}
        row = {
            "at": time(),
            "key": call.get("key") or "default",
            "stage": call.get("stage") or "unknown",
            "model": call.get("model"),
            "ok": call.get("status") == "ok",
            "code": call.get("error_code"),
            "cache_hit": bool(call.get("cache_hit")),
            "queue_ms": int(call.get("queue_ms") or 0),
            "duration_ms": int(call.get("duration_ms") or 0),
            "prompt_tokens": int(tokens.get("prompt_tokens") or 0),
            "cached_tokens": int(tokens.get("precached_prompt_tokens") or 0),
            "completion_tokens": int(tokens.get("completion_tokens") or 0),
        }
        with self._lock:
            self.calls.append(row)
            total = self.totals.setdefault(row["key"], Counter())
            total["calls"] += 1
            total["errors"] += not row["ok"]
            total["duration_ms"] += row["duration_ms"]
            total["queue_ms"] += row["queue_ms"]
            total["prompt_tokens"] += row["prompt_tokens"]
            total["completion_tokens"] += row["completion_tokens"]
        self._maybe_report()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            calls = list(self.calls)
            totals = {key: dict(value) for key, value in self.totals.items()}
            capacity = dict(self.capacity)
            in_flight = dict(self.in_flight)
            waiting = self.waiting
        now = time()
        recent = [row for row in calls if now - row["at"] <= RECENT_SECONDS]
        keys = sorted(set(capacity) | set(totals) | set(in_flight))
        return {
            "uptime_seconds": round(now - self.started),
            "waiting_requests": waiting,
            "capacity": sum(capacity.values()),
            "in_flight": sum(in_flight.values()),
            "recent_seconds": RECENT_SECONDS,
            "recent": self._summary(recent, RECENT_SECONDS),
            "stages": {
                stage: self._summary(
                    [row for row in calls if row["stage"] == stage]
                )
                for stage in sorted({row["stage"] for row in calls})
            },
            "keys": {
                key: {
                    "capacity": capacity.get(key),
                    "in_flight": in_flight.get(key, 0),
                    "totals": totals.get(key, {}),
                    "recent": self._summary(
                        [row for row in recent if row["key"] == key],
                        RECENT_SECONDS,
                    ),
                }
                for key in keys
            },
            "errors": dict(
                Counter(row["code"] for row in calls if not row["ok"])
            ),
        }

    @staticmethod
    def _summary(
        rows: list, seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        paid = [row for row in rows if not row["cache_hit"]]
        durations = [row["duration_ms"] for row in paid]
        queues = [row["queue_ms"] for row in rows]
        output = sum(row["completion_tokens"] for row in paid)
        busy = sum(durations) / 1000
        result = {
            "calls": len(rows),
            "errors": sum(not row["ok"] for row in rows),
            "request_ms_p50": _percentile(durations, 0.5),
            "request_ms_p95": _percentile(durations, 0.95),
            "request_ms_max": max(durations) if durations else None,
            "queue_ms_p50": _percentile(queues, 0.5),
            "queue_ms_p95": _percentile(queues, 0.95),
            "prompt_tokens": sum(row["prompt_tokens"] for row in paid),
            "cached_prompt_tokens": sum(row["cached_tokens"] for row in paid),
            "completion_tokens": output,
            # Generation speed of one request, not of the whole pool.
            "output_tokens_per_request_second": round(output / busy, 1)
            if busy
            else None,
        }
        if seconds:
            # Pool throughput over the window: grows with keys and workers.
            result["output_tokens_per_second"] = round(output / seconds, 1)
        return result

    def _maybe_report(self) -> None:
        period = _report_seconds()
        with self._lock:
            if not period or monotonic() - self._reported < period:
                return
            elapsed = monotonic() - self._reported
            fresh = len(self.calls) - self._reported_calls
            self._reported = monotonic()
            self._reported_calls = len(self.calls)
            window = list(self.calls)[-fresh:] if fresh > 0 else []
            in_flight = dict(self.in_flight)
            capacity = dict(self.capacity)
            waiting = self.waiting
        if not window:
            return
        summary = self._summary(window, elapsed)
        per_key = ", ".join(
            f"{key} {in_flight.get(key, 0)}/{capacity.get(key, '?')}"
            for key in sorted(set(capacity) | set(in_flight))
        )
        logger.info(
            "LLM stats %.0fs: calls=%d errors=%d request p50=%.1fs "
            "p95=%.1fs queue p50=%.1fs p95=%.1fs out=%d tok (%.0f tok/s "
            "pool, %s tok/s per request) busy keys: %s; waiting=%d",
            elapsed,
            summary["calls"],
            summary["errors"],
            (summary["request_ms_p50"] or 0) / 1000,
            (summary["request_ms_p95"] or 0) / 1000,
            (summary["queue_ms_p50"] or 0) / 1000,
            (summary["queue_ms_p95"] or 0) / 1000,
            summary["completion_tokens"],
            summary.get("output_tokens_per_second") or 0,
            summary["output_tokens_per_request_second"],
            per_key or "-",
            waiting,
        )


STATS = LLMStats()
