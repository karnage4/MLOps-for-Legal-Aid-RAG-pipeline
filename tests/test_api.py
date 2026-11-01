"""Unit tests for FastAPI model loading, validation, and prediction."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from src import app as api


def _components():
    return {
        "bm25": object(),
        "bm25_ids": ["chunk-1"],
        "bm25_texts": ["Evidence"],
        "pinecone_index": object(),
        "embedder": object(),
        "reranker": object(),
        "chunking_strategy": "recursive",
        "pinecone_namespace": None,
    }


def _successful_result():
    return {
        "retrieval": {
            "chunks": [
                {"text": "Evidence one", "source": "law-a.pdf"},
                {"text": "Evidence two", "source": "law-a.pdf"},
                {"text": "Evidence three", "source": "law-b.pdf"},
            ]
        },
        "generation": {
            "answer": "A general inheritance-law answer.",
            "provider": "groq",
            "model_used": "qwen/qwen3.8-27b",
            "fallback_used": True,
            "error": None,
        },
        "timings": {"retrieval_ms": 10, "generation_ms": 20, "total_ms": 30},
    }


def test_load_retriever_components_caches_and_validates(monkeypatch):
    api.load_retriever_components.cache_clear()
    calls = []

    def fake_build(strategy):
        calls.append(strategy)
        return _components()

    monkeypatch.setattr(api, "build_retriever", fake_build)
    try:
        first = api.load_retriever_components()
        second = api.load_retriever_components()
    finally:
        api.load_retriever_components.cache_clear()

    assert first is second
    assert calls == ["recursive"]


def test_load_retriever_components_rejects_incomplete_loader(monkeypatch):
    api.load_retriever_components.cache_clear()
    monkeypatch.setattr(
        api,
        "build_retriever",
        lambda strategy: {"chunking_strategy": strategy},
    )
    try:
        try:
            api.load_retriever_components()
        except RuntimeError as exc:
            assert "incomplete components" in str(exc)
            assert "pinecone_index" in str(exc)
        else:
            raise AssertionError("Incomplete retriever components were accepted")
    finally:
        api.load_retriever_components.cache_clear()


def test_predict_returns_validated_compact_response(monkeypatch):
    observed = {}

    def fake_profile(**kwargs):
        return SimpleNamespace(chunking_strategy="recursive", **kwargs)

    def fake_run(query, components, *, temperature, profile):
        observed.update(
            query=query,
            components=components,
            temperature=temperature,
            profile=profile,
        )
        return _successful_result()

    monkeypatch.setattr(api, "build_profile", fake_profile)
    monkeypatch.setattr(api, "run_rag_query", fake_run)
    application = api.create_app(retriever_loader=_components)

    with TestClient(application) as client:
        response = client.post(
            "/predict",
            json={
                "query": "  Who inherits without a will?  ",
                "retrieval_mode": "semantic",
                "top_k": 3,
                "temperature": 0.1,
            },
        )
        docs_response = client.get("/docs")
        health_response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "answer": "A general inheritance-law answer.",
        "provider": "groq",
        "model": "qwen/qwen3.8-27b",
        "fallback_used": True,
        "retrieved_chunks": 3,
        "sources": ["law-a.pdf", "law-b.pdf"],
        "chunking_strategy": "recursive",
        "retrieval_mode": "semantic",
        "top_k": 3,
        "timings": {"retrieval_ms": 10, "generation_ms": 20, "total_ms": 30},
    }
    assert observed["query"] == "Who inherits without a will?"
    assert observed["components"]["chunking_strategy"] == "recursive"
    assert observed["temperature"] == 0.1
    assert observed["profile"].retrieval_mode == "semantic"
    assert observed["profile"].top_k == 3
    assert docs_response.status_code == 200
    assert health_response.json()["retriever_loaded"] is True


def test_predict_rejects_invalid_requests_before_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "run_rag_query", lambda *args, **kwargs: calls.append(1))
    application = api.create_app(retriever_loader=_components)

    with TestClient(application) as client:
        blank = client.post("/predict", json={"query": "   "})
        invalid_top_k = client.post(
            "/predict", json={"query": "Valid question", "top_k": 11}
        )
        extra_field = client.post(
            "/predict", json={"query": "Valid question", "secret": "not allowed"}
        )

    assert blank.status_code == 422
    assert invalid_top_k.status_code == 422
    assert extra_field.status_code == 422
    assert calls == []


def test_predict_maps_generation_failure_to_bad_gateway(monkeypatch):
    result = _successful_result()
    result["generation"] = {
        "answer": None,
        "provider": None,
        "model_used": None,
        "fallback_used": True,
        "error": "all providers unavailable",
    }
    monkeypatch.setattr(
        api,
        "build_profile",
        lambda **kwargs: SimpleNamespace(chunking_strategy="recursive", **kwargs),
    )
    monkeypatch.setattr(api, "run_rag_query", lambda *args, **kwargs: result)
    application = api.create_app(retriever_loader=_components)

    with TestClient(application) as client:
        response = client.post("/predict", json={"query": "Valid question"})

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Answer generation failed: all providers unavailable"
    )
