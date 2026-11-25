"""GET /stats: aggregate question statistics that never expose user content."""

import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import app as api
from src import monitoring
from src.monitoring import QUERY_ROUTES, QueryStats, percentile

PII_QUESTION = (
    "My name is Jane Doe, email jane.doe@example.com, phone +92 300 1234567, "
    "CNIC 35202-1234567-1. Who inherits my house?"
)
RETRIEVED_TEXT = "Retrieved clause about the widow's one-eighth share"
ANSWER = "A general inheritance-law answer for this family."
SOURCE = "private-source-document.pdf"

TOP_LEVEL_KEYS = {
    "started_at",
    "uptime_seconds",
    "request_count",
    "success_count",
    "failure_count",
    "average_latency_ms",
    "routes",
    "queries",
}
QUERY_KEYS = {
    "total",
    "succeeded",
    "failed",
    "error_rate",
    "per_minute",
    "by_status",
    "by_endpoint",
    "latency_ms",
}
LATENCY_KEYS = {"window", "samples", "mean", "min", "max", "p50", "p90", "p95", "p99"}


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _retrievers():
    return {
        strategy: {"chunking_strategy": strategy, "pinecone_namespace": None}
        for strategy in ("recursive", "fixed")
    }


def _result(error=None):
    return {
        "retrieval": {"chunks": [{"text": RETRIEVED_TEXT, "source": SOURCE}]},
        "generation": {
            "answer": None if error else ANSWER,
            "provider": "groq",
            "model_used": "qwen/test",
            "fallback_used": False,
            "error": error,
        },
        "timings": {"retrieval_ms": 10, "generation_ms": 20, "total_ms": 30},
    }


def _app(monkeypatch, run=None, logs=None):
    monkeypatch.setattr(api, "build_profile", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(api, "run_rag_query", run or (lambda *a, **k: _result()))
    sink = logs if logs is not None else []
    return api.create_app(retriever_loader=_retrievers, access_log=sink.append)


# --- /stats endpoint ---------------------------------------------------------


def test_stats_on_a_fresh_app_returns_the_schema_with_zeros_and_nulls(monkeypatch):
    with TestClient(_app(monkeypatch)) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == TOP_LEVEL_KEYS
    assert body["request_count"] == 0
    assert body["average_latency_ms"] == 0.0
    assert body["routes"] == {}
    queries = body["queries"]
    assert set(queries) == QUERY_KEYS
    assert (queries["total"], queries["succeeded"], queries["failed"]) == (0, 0, 0)
    assert queries["error_rate"] is None
    assert queries["per_minute"] in (0.0, None)
    assert queries["by_status"] == {}
    assert queries["by_endpoint"] == {}
    latency = queries["latency_ms"]
    assert set(latency) == LATENCY_KEYS
    assert latency["samples"] == 0
    assert latency["window"] == monitoring.LATENCY_WINDOW
    assert all(latency[key] is None for key in LATENCY_KEYS - {"window", "samples"})


def test_stats_updates_after_successful_and_failed_queries(monkeypatch):
    outcomes = iter([_result(), _result(), _result(error="provider unavailable")])
    with TestClient(_app(monkeypatch, run=lambda *a, **k: next(outcomes))) as client:
        assert client.post("/predict", json={"query": "First question"}).status_code == 200
        assert client.post("/predict", json={"query": "Second question"}).status_code == 200
        assert client.post("/predict", json={"query": "Third question"}).status_code == 502
        assert client.post("/predict", json={"query": "hi"}).status_code == 422
        assert client.get("/health").status_code == 200
        first = client.get("/stats").json()
        second = client.get("/stats").json()

    queries = first["queries"]
    assert queries["total"] == 4
    assert queries["succeeded"] == 2
    assert queries["failed"] == 2
    assert queries["error_rate"] == 0.5
    assert queries["by_status"] == {"200": 2, "422": 1, "502": 1}
    assert queries["by_endpoint"] == {"POST /predict": 4}
    assert queries["latency_ms"]["samples"] == 2
    # /health is a request but not a question; /stats reads are neither.
    assert first["request_count"] == 5
    assert second["request_count"] == 5
    assert second["queries"]["total"] == 4
    assert "GET /stats" not in first["routes"]


def test_reading_stats_and_metrics_does_not_change_them(monkeypatch):
    logs = []
    with TestClient(_app(monkeypatch, logs=logs)) as client:
        for _ in range(3):
            client.get("/stats")
            client.get("/metrics")
        body = client.get("/stats").json()

    assert body["request_count"] == 0
    assert body["queries"]["total"] == 0
    assert logs == []


def test_stats_never_contain_question_text_retrieved_text_or_pii(monkeypatch):
    logs = []
    with TestClient(_app(monkeypatch, logs=logs)) as client:
        assert client.post("/predict", json={"query": PII_QUESTION}).status_code == 200
        client.post("/predict", json={"query": PII_QUESTION, "email": "x@example.com"})
        client.get("/no-such-page/jane.doe@example.com?cnic=35202-1234567-1")
        stats_text = client.get("/stats").text
        metrics_text = client.get("/metrics").text

    for output in (stats_text, metrics_text, "\n".join(logs)):
        for private in (
            "Jane Doe",
            "jane.doe",
            "example.com",
            "1234567",
            "35202",
            "Who inherits",
            RETRIEVED_TEXT,
            ANSWER,
            SOURCE,
            "no-such-page",
        ):
            assert private not in output

    body = json.loads(stats_text)
    assert set(body["queries"]["by_endpoint"]) <= QUERY_ROUTES
    assert all(key.isdigit() for key in body["queries"]["by_status"])


def test_stats_is_documented_in_openapi(monkeypatch):
    with TestClient(_app(monkeypatch)) as client:
        schema = client.get("/openapi.json").json()

    assert schema["paths"]["/stats"]["get"]["tags"] == ["operations"]
    assert "StatsResponse" in schema["components"]["schemas"]


# --- fail-safe ---------------------------------------------------------------


def _explode(*args, **kwargs):
    raise RuntimeError("monitoring bug")


@pytest.mark.parametrize(
    "target",
    [
        (QueryStats, "record"),
        (monitoring.RequestMetrics, "record"),
    ],
)
def test_a_broken_collector_never_breaks_a_query(monkeypatch, target):
    monkeypatch.setattr(*target, _explode)
    with TestClient(_app(monkeypatch)) as client:
        response = client.post("/predict", json={"query": "Valid question"})
        stats = client.get("/stats")

    assert response.status_code == 200
    assert response.json()["answer"] == ANSWER
    assert stats.status_code == 200


def test_a_broken_access_log_never_breaks_a_query(monkeypatch):
    monkeypatch.setattr(api, "build_profile", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(api, "run_rag_query", lambda *a, **k: _result())
    application = api.create_app(retriever_loader=_retrievers, access_log=_explode)

    with TestClient(application) as client:
        response = client.post("/predict", json={"query": "Valid question"})
        stats = client.get("/stats").json()

    assert response.status_code == 200
    assert stats["queries"]["total"] == 1


def test_a_broken_route_lookup_serves_the_query_unmonitored(monkeypatch):
    monkeypatch.setattr(monitoring, "resolve_route", _explode)
    with TestClient(_app(monkeypatch)) as client:
        response = client.post("/predict", json={"query": "Valid question"})
        stats = client.get("/stats")

    assert response.status_code == 200
    assert "x-request-id" not in response.headers
    assert stats.status_code == 200


# --- QueryStats and percentile calculations -----------------------------------


def test_latency_percentiles_and_summary_are_correct():
    stats = QueryStats(clock=FakeClock())
    for elapsed in (50.0, 10.0, 40.0, 20.0, 30.0):
        stats.record("POST /predict", 200, elapsed)

    latency = stats.snapshot()["latency_ms"]

    assert latency["samples"] == 5
    assert latency["mean"] == 30.0
    assert latency["min"] == 10.0
    assert latency["max"] == 50.0
    assert latency["p50"] == 30.0
    assert latency["p90"] == 46.0
    assert latency["p95"] == 48.0
    assert latency["p99"] == 49.6


def test_failed_queries_count_but_do_not_enter_latency():
    stats = QueryStats(clock=FakeClock())
    stats.record("POST /predict", 200, 100.0)
    stats.record("POST /predict", 422, 1.0)
    stats.record("POST /ui/predict", 502, 3000.0)

    snapshot = stats.snapshot()

    assert snapshot["total"] == 3
    assert snapshot["failed"] == 2
    assert snapshot["error_rate"] == round(2 / 3, 4)
    assert snapshot["latency_ms"]["samples"] == 1
    assert snapshot["latency_ms"]["p99"] == 100.0
    assert snapshot["by_endpoint"] == {"POST /predict": 2, "POST /ui/predict": 1}


def test_rate_per_minute_uses_uptime_and_handles_zero_uptime():
    clock = FakeClock()
    stats = QueryStats(clock=clock)
    for _ in range(6):
        stats.record("POST /predict", 200, 5.0)

    assert stats.snapshot()["per_minute"] is None  # no time has passed
    clock.now += 120
    assert stats.snapshot()["per_minute"] == 3.0


def test_latency_window_keeps_only_recent_successes():
    stats = QueryStats(clock=FakeClock(), window=3)
    for elapsed in (1.0, 2.0, 3.0, 4.0, 5.0):
        stats.record("POST /predict", 200, elapsed)

    snapshot = stats.snapshot()

    assert snapshot["total"] == 5
    assert snapshot["latency_ms"]["samples"] == 3
    assert snapshot["latency_ms"]["min"] == 3.0
    assert snapshot["latency_ms"]["window"] == 3


def test_concurrent_records_are_not_lost():
    stats = QueryStats()

    def record_many():
        for index in range(1000):
            stats.record("POST /predict", 200 if index % 4 else 502, 1.0)

    threads = [threading.Thread(target=record_many) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = stats.snapshot()
    assert snapshot["total"] == 8000
    assert snapshot["failed"] == 2000
    assert snapshot["by_status"] == {"200": 6000, "502": 2000}


@pytest.mark.parametrize(
    ("values", "pct", "expected"),
    [
        ([], 50, None),
        ([7.0], 99, 7.0),
        ([10.0, 20.0], 50, 15.0),
        ([10.0, 20.0, 30.0, 40.0, 50.0], 0, 10.0),
        ([10.0, 20.0, 30.0, 40.0, 50.0], 100, 50.0),
        ([10.0, 20.0, 30.0, 40.0, 50.0], 25, 20.0),
    ],
)
def test_percentile(values, pct, expected):
    assert percentile(values, pct) == expected


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        percentile([1.0], 101)
