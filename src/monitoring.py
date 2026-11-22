"""Request metrics and structured access logs for the FastAPI service.

Counters live in memory in the single uvicorn worker. They reset whenever the
process restarts: a redeploy, a crash, or Azure Container Apps scaling the
app to zero replicas after an idle period. With at most one replica, a
snapshot covers all traffic the running replica has served since
``started_at``; it is not long-term or multi-instance telemetry.

Counting rule: every HTTP request is counted and logged except reads of
``GET /metrics`` itself, so checking the numbers does not change them. A
response with status 400 or higher, or an unhandled exception (recorded as
500), counts as a failure. Paths that match no route are grouped as
``unmatched`` so arbitrary URLs cannot create new metric keys or log values.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

ACCESS_LOGGER_NAME = "legal_aid_rag.access"
METRICS_PATH = "/metrics"
UNMATCHED_ROUTE = "unmatched"
REQUEST_ID_HEADER = b"x-request-id"


class _Counter:
    __slots__ = ("requests", "failures", "elapsed_ms")

    def __init__(self) -> None:
        self.requests = 0
        self.failures = 0
        self.elapsed_ms = 0.0

    def add(self, failed: bool, elapsed_ms: float) -> None:
        self.requests += 1
        self.failures += int(failed)
        self.elapsed_ms += elapsed_ms

    def as_dict(self) -> dict[str, Any]:
        average = self.elapsed_ms / self.requests if self.requests else 0.0
        return {
            "request_count": self.requests,
            "success_count": self.requests - self.failures,
            "failure_count": self.failures,
            "average_latency_ms": round(average, 1),
        }


class RequestMetrics:
    """Thread-safe request counters, in total and per route."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self._started_at = datetime.now(timezone.utc)
        self._lock = threading.Lock()
        self._total = _Counter()
        self._routes: dict[str, _Counter] = {}

    def record(self, route: str, status_code: int, elapsed_ms: float) -> None:
        failed = status_code >= 400
        with self._lock:
            self._total.add(failed, elapsed_ms)
            self._routes.setdefault(route, _Counter()).add(failed, elapsed_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self._total.as_dict()
            routes = {
                route: counter.as_dict()
                for route, counter in sorted(self._routes.items())
            }
        return {
            "started_at": self._started_at.isoformat(timespec="seconds"),
            "uptime_seconds": round(max(0.0, self._clock() - self._started), 3),
            **total,
            "routes": routes,
        }


def default_access_log() -> Callable[[str], None]:
    """Return a writer that prints one access-log line per call to stdout."""
    logger = logging.getLogger(ACCESS_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger.info


class RequestMonitoringMiddleware:
    """ASGI middleware that records and logs every counted request.

    It reads only the method, the path (to name the route), and the response
    status. It never reads request bodies, query strings, or headers, so
    questions, answers, and credentials cannot reach the log. Each response
    gets an ``X-Request-ID`` header matching its log line. Exceptions are
    recorded as status 500 and re-raised unchanged.
    """

    def __init__(
        self,
        app: Any,
        *,
        metrics: RequestMetrics,
        route_paths: Callable[[], Iterable[str]],
        log: Callable[[str], None],
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.app = app
        self._metrics = metrics
        self._route_paths = route_paths
        self._log = log
        self._timer = timer

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope["path"]
        route = path if path in set(self._route_paths()) else UNMATCHED_ROUTE
        request_id = uuid.uuid4().hex
        status_code = 500
        started = self._timer()

        async def send_with_request_id(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER, request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            elapsed_ms = (self._timer() - started) * 1000
            if not (method == "GET" and route == METRICS_PATH):
                self._metrics.record(f"{method} {route}", status_code, elapsed_ms)
                self._log(
                    json.dumps(
                        {
                            "event": "http_request",
                            "timestamp": datetime.now(timezone.utc).isoformat(
                                timespec="milliseconds"
                            ),
                            "request_id": request_id,
                            "method": method,
                            "route": route,
                            "status": status_code,
                            "duration_ms": round(elapsed_ms, 1),
                        }
                    )
                )
