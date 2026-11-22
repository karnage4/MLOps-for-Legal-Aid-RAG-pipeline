"""The deployed entry point: API routes plus the Miras Gradio interface."""

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import gradio_app
from src import web

QUESTION = "What share does a daughter receive in Islamic inheritance?"
ANSWER = "The share depends on the surviving heirs."


def _retrievers():
    return {
        strategy: {"chunking_strategy": strategy, "marker": object()}
        for strategy in ("recursive", "fixed")
    }


def _result(*args, **kwargs):
    return {
        "retrieval": {
            "chunks": [{"source": "law.pdf", "text": "Evidence", "rerank_score": 0.9}]
        },
        "generation": {
            "answer": ANSWER,
            "provider": "groq",
            "model_used": "qwen/test",
            "fallback_used": False,
            "error": None,
        },
        "timings": {"retrieval_ms": 11, "generation_ms": 22, "total_ms": 33},
    }


def _web_app(logs, retrievers=None):
    loaded = retrievers or _retrievers()
    return web.create_web_app(retriever_loader=lambda: loaded, access_log=logs.append)


def _ask_interface(client, data):
    """Call the interface's predict event the way the browser does."""
    started = client.post("/ui/gradio_api/call/predict", json={"data": data})
    assert started.status_code == 200, started.text
    event_id = started.json()["event_id"]
    return client.get(f"/ui/gradio_api/call/predict/{event_id}").text


def test_root_redirects_to_the_interface():
    with TestClient(_web_app([])) as client:
        response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_interface_is_served_and_its_assets_are_not_counted():
    logs = []

    with TestClient(_web_app(logs)) as client:
        page = client.get("/ui/")
        config = client.get("/ui/config")
        metrics = client.get("/metrics").json()

    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert config.status_code == 200
    assert config.json()["title"] == gradio_app.APP_TITLE
    assert metrics["request_count"] == 0
    assert logs == []


def test_api_routes_still_work_beside_the_interface():
    with TestClient(_web_app([])) as client:
        health = client.get("/health")
        docs = client.get("/docs")
        schema = client.get("/openapi.json").json()

    assert health.status_code == 200
    assert health.json()["retriever_loaded"] is True
    assert docs.status_code == 200
    assert {"/health", "/predict", "/metrics"} <= set(schema["paths"])
    assert "/" not in schema["paths"]


def test_interface_question_reuses_loaded_retrievers_and_is_counted(monkeypatch):
    observed = {}
    retrievers = _retrievers()

    def fake_run(query, components, **kwargs):
        observed.update(query=query, components=components)
        return _result()

    monkeypatch.setattr(gradio_app, "run_rag_query", fake_run)
    monkeypatch.setattr(
        gradio_app, "build_profile", lambda **kwargs: SimpleNamespace(**kwargs)
    )

    def fail_local_load(strategy):
        raise AssertionError("the interface must not load its own retrievers")

    monkeypatch.setattr(gradio_app, "load_retriever_components", fail_local_load)
    logs = []

    with TestClient(_web_app(logs, retrievers)) as client:
        stream = _ask_interface(client, [QUESTION, "fixed", "hybrid", 5, 0.2, True])
        metrics = client.get("/metrics").json()

    assert "event: complete" in stream
    assert ANSWER in stream
    assert observed["query"] == QUESTION
    assert observed["components"] is retrievers["fixed"]
    ui_metrics = metrics["routes"]["POST /ui/predict"]
    assert ui_metrics["request_count"] == 1
    assert ui_metrics["success_count"] == 1
    assert ui_metrics["failure_count"] == 0
    assert metrics["request_count"] == 1
    entry = json.loads(logs[-1])
    assert (entry["method"], entry["route"], entry["status"]) == (
        "POST",
        "/ui/predict",
        200,
    )
    written = "\n".join(logs)
    assert QUESTION not in written
    assert ANSWER not in written


def test_interface_rejection_is_counted_as_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(gradio_app, "run_rag_query", lambda *a, **k: calls.append(1))
    logs = []

    with TestClient(_web_app(logs)) as client:
        stream = _ask_interface(client, ["hi", "recursive", "hybrid", 5, 0.2, True])
        metrics = client.get("/metrics").json()

    assert "event: error" in stream
    assert calls == []
    assert metrics["routes"]["POST /ui/predict"]["failure_count"] == 1
    assert json.loads(logs[-1])["status"] == 422
