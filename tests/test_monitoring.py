"""Request metrics, the /metrics endpoint, and structured access logs."""

import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import app as api
from src.monitoring import RequestMetrics

QUESTION = "Who inherits a house when there is no will?"
ANSWER = "A general inheritance-law answer."
SOURCE = "law-a.pdf"
PRIVATE_VALUE = "do-not-log-this-value"
LOG_FIELDS = {
    "event",
    "timestamp",
    "request_id",
    "method",
    "route",
    "status",
    "duration_ms",
}


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _retrievers():
    return {
        strategy: {
            "bm25": object(),
            "bm25_ids": ["chunk-1"],
            "bm25_texts": ["Evidence"],
            "pinecone_index": object(),
            "embedder": object(),
            "reranker": object(),
            "chunking_strategy": strategy,
            "pinecone_namespace": "fixed" if strategy == "fixed" else None,
        }
        for strategy in ("recursive", "fixed")
    }


def _successful_result(*args, **kwargs):
    return {
        "retrieval": {"chunks": [{"text": "Evidence", "source": SOURCE}]},
        "generation": {
            "answer": ANSWER,
            "provider": "groq",
            "model_used": "qwen/qwen3.8-27b",
            "fallback_used": False,
            "error": None,
        },
        "timings": {"retrieval_ms": 10, "generation_ms": 20, "total_ms": 30},
    }


def _monitored_app(monkeypatch, logs):
    monkeypatch.setattr(api, "build_profile", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(api, "run_rag_query", _successful_result)
    return api.create_app(retriever_loader=_retrievers, access_log=logs.append)


def test_new_metrics_are_empty():
    snapshot = RequestMetrics(clock=FakeClock()).snapshot()

    assert snapshot["request_count"] == 0
    assert snapshot["success_count"] == 0
    assert snapshot["failure_count"] == 0
    assert snapshot["average_latency_ms"] == 0.0
    assert snapshot["uptime_seconds"] == 0.0
    assert snapshot["routes"] == {}


def test_record_counts_outcomes_and_average_latency():
    metrics = RequestMetrics(clock=FakeClock())

    metrics.record("POST /predict", 200, 10.0)
    metrics.record("POST /predict", 422, 30.0)
    metrics.record("GET /health", 200, 2.0)
    snapshot = metrics.snapshot()

    assert snapshot["request_count"] == 3
    assert snapshot["success_count"] == 2
    assert snapshot["failure_count"] == 1
    assert snapshot["average_latency_ms"] == 14.0
    assert snapshot["routes"]["POST /predict"] == {
        "request_count": 2,
        "success_count": 1,
        "failure_count": 1,
        "average_latency_ms": 20.0,
    }
    assert snapshot["routes"]["GET /health"]["average_latency_ms"] == 2.0


@pytest.mark.parametrize(
    ("status_code", "failed"),
    [(200, False), (204, False), (307, False), (399, False)]
    + [(400, True), (404, True), (422, True), (500, True), (502, True), (503, True)],
)
def test_status_400_and_above_counts_as_failure(status_code, failed):
    metrics = RequestMetrics(clock=FakeClock())

    metrics.record("GET /health", status_code, 1.0)

    assert metrics.snapshot()["failure_count"] == int(failed)


def test_uptime_is_nonnegative_and_increases():
    clock = FakeClock()
    metrics = RequestMetrics(clock=clock)

    first = metrics.snapshot()["uptime_seconds"]
    clock.now += 2.5
    second = metrics.snapshot()["uptime_seconds"]

    assert first == 0.0
    assert second == 2.5


def test_concurrent_records_are_not_lost():
    metrics = RequestMetrics()

    def record_many():
        for _ in range(2000):
            metrics.record("GET /health", 200, 1.0)

    threads = [threading.Thread(target=record_many) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = metrics.snapshot()
    assert snapshot["request_count"] == 16000
    assert snapshot["routes"]["GET /health"]["request_count"] == 16000


def test_metrics_endpoint_counts_requests_but_not_itself(monkeypatch):
    logs = []
    application = _monitored_app(monkeypatch, logs)

    with TestClient(application) as client:
        before = client.get("/metrics").json()
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json={"query": QUESTION}).status_code == 200
        assert client.post("/predict", json={"query": "hi"}).status_code == 422
        assert client.get("/no-such-page").status_code == 404
        after = client.get("/metrics")

    assert before["request_count"] == 0
    assert after.status_code == 200
    body = after.json()
    assert set(body) == {
        "started_at",
        "uptime_seconds",
        "request_count",
        "success_count",
        "failure_count",
        "average_latency_ms",
        "routes",
    }
    assert body["request_count"] == 4
    assert body["success_count"] == 2
    assert body["failure_count"] == 2
    assert body["uptime_seconds"] >= 0
    assert set(body["routes"]) == {"GET /health", "POST /predict", "GET unmatched"}
    assert body["routes"]["POST /predict"]["request_count"] == 2
    assert body["routes"]["POST /predict"]["failure_count"] == 1
    assert body["routes"]["GET unmatched"]["failure_count"] == 1

    weighted = sum(
        route["average_latency_ms"] * route["request_count"]
        for route in body["routes"].values()
    ) / body["request_count"]
    assert body["average_latency_ms"] == pytest.approx(weighted, abs=0.1)


def test_access_log_has_one_safe_json_line_per_counted_request(monkeypatch):
    logs = []
    application = _monitored_app(monkeypatch, logs)

    with TestClient(application) as client:
        answered = client.post("/predict", json={"query": QUESTION})
        rejected = client.post(
            "/predict", json={"query": QUESTION, "secret": PRIVATE_VALUE}
        )
        client.get(f"/no-such-page/{PRIVATE_VALUE}?key={PRIVATE_VALUE}")
        client.get("/metrics")

    assert answered.status_code == 200
    assert rejected.status_code == 422
    assert len(logs) == 3
    entries = [json.loads(line) for line in logs]
    assert all(set(entry) == LOG_FIELDS for entry in entries)
    assert [(e["method"], e["route"], e["status"]) for e in entries] == [
        ("POST", "/predict", 200),
        ("POST", "/predict", 422),
        ("GET", "unmatched", 404),
    ]
    assert all(entry["event"] == "http_request" for entry in entries)
    assert all(entry["duration_ms"] >= 0 for entry in entries)
    assert entries[0]["request_id"] == answered.headers["x-request-id"]
    assert entries[1]["request_id"] == rejected.headers["x-request-id"]
    assert entries[0]["request_id"] != entries[1]["request_id"]

    written = "\n".join(logs)
    for private in (QUESTION, ANSWER, SOURCE, PRIVATE_VALUE, "no-such-page"):
        assert private not in written


def test_unhandled_errors_are_counted_and_still_raised(monkeypatch):
    logs = []
    application = _monitored_app(monkeypatch, logs)

    @application.get("/boom")
    def boom():
        raise RuntimeError("boom")

    with TestClient(application) as client:
        with pytest.raises(RuntimeError, match="boom"):
            client.get("/boom")
        body = client.get("/metrics").json()

    assert body["routes"]["GET /boom"]["request_count"] == 1
    assert body["routes"]["GET /boom"]["failure_count"] == 1
    assert json.loads(logs[-1])["status"] == 500


def test_metrics_endpoint_is_documented(monkeypatch):
    application = _monitored_app(monkeypatch, [])

    with TestClient(application) as client:
        schema = client.get("/openapi.json").json()

    operation = schema["paths"]["/metrics"]["get"]
    assert operation["tags"] == ["operations"]
    assert "MetricsResponse" in schema["components"]["schemas"]
