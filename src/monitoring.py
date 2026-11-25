"""Request metrics and structured access logs for the FastAPI service.

Counters live in memory in the single uvicorn worker. They reset whenever the
process restarts: a redeploy, a crash, or Azure Container Apps scaling the
app to zero replicas after an idle period. With at most one replica, a
snapshot covers all traffic the running replica has served since
``started_at``; it is not long-term or multi-instance telemetry.

Counting rule: every HTTP request to an API route is counted and logged
except reads of ``GET /metrics`` and ``GET /stats`` themselves, so checking
the numbers does not change them. Requests served by a mounted
sub-application (the Gradio interface's page, assets, and event stream under
``/ui``) are not counted, because one page load makes dozens of asset
requests and its event stream stays open for minutes; each question asked in
the interface is recorded instead as ``POST /ui/predict`` through
``RequestMonitor.observe``. A response with status 400 or higher, or an
unhandled exception (recorded as 500), counts as a failure. Paths that match
no route are grouped as ``unmatched`` so arbitrary URLs cannot create new
metric keys or log values.

Questions (``POST /predict`` and ``POST /ui/predict``) are also aggregated by
``QueryStats`` for ``GET /stats``: outcome counts, error rate, rate per
minute, and latency percentiles over recent successful answers.

Privacy: only the method, the matched route name, the status code, and the
elapsed time are ever recorded. Request bodies, query strings, headers,
questions, retrieved text, and answers are never read here.

Fail-safe: monitoring runs alongside user requests and must never break one.
Every monitoring step taken during a request catches its own exceptions and
reports them to the ``src.monitoring`` logger without request data.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

from starlette.routing import Mount

ACCESS_LOGGER_NAME = "legal_aid_rag.access"
METRICS_PATH = "/metrics"
STATS_PATH = "/stats"
UNMATCHED_ROUTE = "unmatched"
REQUEST_ID_HEADER = b"x-request-id"
# Reading the monitoring endpoints never changes what they report.
UNCOUNTED_REQUESTS = frozenset({("GET", METRICS_PATH), ("GET", STATS_PATH)})
# The question-answering endpoints aggregated by QueryStats.
QUERY_ROUTES = frozenset({"POST /predict", "POST /ui/predict"})
# Latency percentiles cover at most this many recent successful answers.
LATENCY_WINDOW = 1000
PERCENTILES = (50, 90, 95, 99)

_LOGGER = logging.getLogger(__name__)


def _report_monitoring_failure(step: str) -> None:
    """Log a monitoring bug without request data; never raise."""
    try:
        _LOGGER.warning(
            "Request monitoring step %r failed; the request itself was not affected",
            step,
            exc_info=True,
        )
    except Exception:  # pragma: no cover - logging itself failed
        pass


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


def percentile(sorted_values: Sequence[float], pct: float) -> float | None:
    """Linear-interpolation percentile (NumPy's default); None when empty.

    ``sorted_values`` must be in ascending order. For values 10, 20, 30, 40,
    and 50, the 50th percentile is 30 and the 90th is 46.
    """
    if not sorted_values:
        return None
    if not 0 <= pct <= 100:
        raise ValueError("pct must be between 0 and 100")
    rank = (len(sorted_values) - 1) * pct / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    low_value = sorted_values[lower]
    return low_value + (sorted_values[upper] - low_value) * (rank - lower)


class QueryStats:
    """Thread-safe aggregate statistics for question-answering requests.

    ``record`` is O(1) under a lock. Latency samples of successful answers
    are kept in a bounded window, so memory stays constant and percentiles
    describe recent behaviour; they are computed only when read.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        window: int = LATENCY_WINDOW,
    ) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        self._clock = clock
        self._started = clock()
        self._window = window
        self._lock = threading.Lock()
        self._total = 0
        self._failed = 0
        self._by_status: Counter[str] = Counter()
        self._by_endpoint: Counter[str] = Counter()
        self._latencies: deque[float] = deque(maxlen=window)

    def record(self, endpoint: str, status_code: int, elapsed_ms: float) -> None:
        failed = status_code >= 400
        with self._lock:
            self._total += 1
            self._failed += int(failed)
            self._by_status[str(status_code)] += 1
            self._by_endpoint[endpoint] += 1
            if not failed:
                self._latencies.append(elapsed_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self._total
            failed = self._failed
            by_status = dict(sorted(self._by_status.items()))
            by_endpoint = dict(sorted(self._by_endpoint.items()))
            latencies = sorted(self._latencies)
        uptime_minutes = max(0.0, self._clock() - self._started) / 60

        def rounded(value: float | None) -> float | None:
            return None if value is None else round(value, 1)

        latency: dict[str, Any] = {
            "window": self._window,
            "samples": len(latencies),
            "mean": rounded(sum(latencies) / len(latencies)) if latencies else None,
            "min": rounded(latencies[0]) if latencies else None,
            "max": rounded(latencies[-1]) if latencies else None,
        }
        for pct in PERCENTILES:
            latency[f"p{pct}"] = rounded(percentile(latencies, pct))
        return {
            "total": total,
            "succeeded": total - failed,
            "failed": failed,
            "error_rate": round(failed / total, 4) if total else None,
            "per_minute": round(total / uptime_minutes, 3) if uptime_minutes else None,
            "by_status": by_status,
            "by_endpoint": by_endpoint,
            "latency_ms": latency,
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


class RequestMonitor:
    """Records one counted request in the metrics and stats and logs it.

    ``observe`` never raises: each of its three steps (metrics, query
    stats, log line) fails on its own without affecting the others or the
    request being observed.
    """

    def __init__(
        self,
        metrics: RequestMetrics,
        log: Callable[[str], None],
        query_stats: QueryStats | None = None,
    ) -> None:
        self.metrics = metrics
        self.query_stats = query_stats
        self._log = log

    def observe(
        self,
        method: str,
        route: str,
        status_code: int,
        elapsed_ms: float,
        request_id: str | None = None,
    ) -> None:
        endpoint = f"{method} {route}"
        try:
            self.metrics.record(endpoint, status_code, elapsed_ms)
        except Exception:
            _report_monitoring_failure("metrics")
        if self.query_stats is not None and endpoint in QUERY_ROUTES:
            try:
                self.query_stats.record(endpoint, status_code, elapsed_ms)
            except Exception:
                _report_monitoring_failure("query_stats")
        try:
            self._log(
                json.dumps(
                    {
                        "event": "http_request",
                        "timestamp": datetime.now(timezone.utc).isoformat(
                            timespec="milliseconds"
                        ),
                        "request_id": request_id or uuid.uuid4().hex,
                        "method": method,
                        "route": route,
                        "status": status_code,
                        "duration_ms": round(elapsed_ms, 1),
                    }
                )
            )
        except Exception:
            _report_monitoring_failure("access_log")


def resolve_route(path: str, routes: Iterable[Any]) -> str | None:
    """Name the route serving ``path``; None for a mounted sub-application."""
    mounted = False
    for route in routes:
        route_path = getattr(route, "path", None)
        if route_path is None:
            continue
        if isinstance(route, Mount):
            prefix = route_path.rstrip("/")
            if prefix and (path == prefix or path.startswith(prefix + "/")):
                mounted = True
        elif path == route_path:
            return route_path
    return None if mounted else UNMATCHED_ROUTE


class RequestMonitoringMiddleware:
    """ASGI middleware that records and logs every counted request.

    It reads only the method, the path (to name the route), and the response
    status. It never reads request bodies, query strings, or headers, so
    questions, answers, and credentials cannot reach the log. Each counted
    response gets an ``X-Request-ID`` header matching its log line.
    Exceptions from the application are recorded as status 500 and re-raised
    unchanged; a failure inside the monitoring code is logged and the request
    proceeds as if it were not monitored.
    """

    def __init__(
        self,
        app: Any,
        *,
        monitor: RequestMonitor,
        routes: Callable[[], Iterable[Any]],
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.app = app
        self._monitor = monitor
        self._routes = routes
        self._timer = timer

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            method = scope["method"]
            route = resolve_route(scope["path"], self._routes())
            request_id = uuid.uuid4().hex
            started = self._timer()
        except Exception:
            _report_monitoring_failure("start")
            route = None
        if route is None or (method, route) in UNCOUNTED_REQUESTS:
            await self.app(scope, receive, send)
            return

        status_code = 500

        async def send_with_request_id(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                try:
                    status_code = message["status"]
                    headers = list(message.get("headers", []))
                    headers.append((REQUEST_ID_HEADER, request_id.encode("ascii")))
                    message = {**message, "headers": headers}
                except Exception:
                    _report_monitoring_failure("response_header")
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            try:
                elapsed_ms = (self._timer() - started) * 1000
            except Exception:
                _report_monitoring_failure("timer")
                elapsed_ms = 0.0
            self._monitor.observe(method, route, status_code, elapsed_ms, request_id)
