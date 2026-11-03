"""Tests for the local Gradio adapter without cloud calls."""

from __future__ import annotations

import pytest

import gradio_app


def _result(*, error=None, chunks=True):
    retrieved = (
        [
            {
                "source": "law.pdf",
                "text": "A daughter may receive a prescribed share.",
                "rerank_score": 0.91,
            },
            {
                "source": "guide.pdf",
                "text": "The share depends on the surviving heirs.",
                "rerank_score": 0.83,
            },
        ]
        if chunks
        else []
    )
    return {
        "retrieval": {"chunks": retrieved},
        "generation": {
            "answer": None if error else "The share depends on the surviving heirs.",
            "provider": "groq",
            "model_used": "qwen/test",
            "fallback_used": True,
            "error": error,
        },
        "timings": {"retrieval_ms": 11, "generation_ms": 22, "total_ms": 33},
    }


def test_run_prediction_reuses_traced_pipeline_and_formats_output(monkeypatch):
    calls = {}
    components = {"retriever": "cached"}
    profile = object()

    def fake_load(strategy):
        calls["loaded_strategy"] = strategy
        return components

    monkeypatch.setattr(gradio_app, "load_retriever_components", fake_load)

    def fake_build_profile(**kwargs):
        calls["profile_kwargs"] = kwargs
        return profile

    monkeypatch.setattr(gradio_app, "build_profile", fake_build_profile)

    def fake_run(query, loaded, **kwargs):
        calls.update(query=query, loaded=loaded, run_kwargs=kwargs)
        return _result()

    monkeypatch.setattr(gradio_app, "run_rag_query", fake_run)

    answer, status, context, details = gradio_app.run_prediction(
        "  What share does a daughter receive?  ",
        "fixed",
        "hybrid",
        5,
        0.2,
        True,
    )

    assert calls["query"] == "What share does a daughter receive?"
    assert calls["loaded"] is components
    assert calls["loaded_strategy"] == "fixed"
    assert calls["profile_kwargs"] == {
        "chunking_strategy": "fixed",
        "retrieval_mode": "hybrid",
        "top_k": 5,
    }
    assert calls["run_kwargs"] == {"temperature": 0.2, "profile": profile}
    assert answer.startswith("The share")
    assert "groq" in status and "2 chunks / 2 sources" in status
    assert "law.pdf" in context and "guide.pdf" in context
    assert details["timings_ms"]["total"] == 33
    assert details["generation"]["fallback_used"] is True
    assert details["profile"]["chunking_strategy"] == "fixed"


@pytest.mark.parametrize(
    ("query", "strategy", "mode", "message"),
    [
        (" ", "recursive", "hybrid", "at least 3"),
        ("valid question", "invalid", "hybrid", "recursive or fixed"),
        ("valid question", "recursive", "invalid", "hybrid or semantic"),
    ],
)
def test_run_prediction_validates_public_inputs(query, strategy, mode, message):
    with pytest.raises(ValueError, match=message):
        gradio_app.run_prediction(query, strategy, mode, 5, 0.2, True)


def test_run_prediction_fails_closed_on_missing_context(monkeypatch):
    monkeypatch.setattr(gradio_app, "load_retriever_components", lambda _: {})
    monkeypatch.setattr(gradio_app, "build_profile", lambda **_: object())
    monkeypatch.setattr(gradio_app, "run_rag_query", lambda *_, **__: _result(chunks=False))

    with pytest.raises(RuntimeError, match="No relevant legal context"):
        gradio_app.run_prediction(
            "valid question", "recursive", "semantic", 3, 0.1, True
        )


def test_run_prediction_fails_closed_on_generation_error(monkeypatch):
    monkeypatch.setattr(gradio_app, "load_retriever_components", lambda _: {})
    monkeypatch.setattr(gradio_app, "build_profile", lambda **_: object())
    monkeypatch.setattr(
        gradio_app,
        "run_rag_query",
        lambda *_, **__: _result(error="provider unavailable\nsecret details omitted"),
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        gradio_app.run_prediction(
            "valid question", "fixed", "hybrid", 5, 0.2, False
        )
