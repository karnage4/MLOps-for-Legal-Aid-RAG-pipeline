"""Tests for the fixed-profile live retrieval smoke-test contract."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src import retrieval_smoke_test
from src.retrieval import RetrievalProfile


def _result(profile: RetrievalProfile) -> dict:
    chunks = [
        {
            "id": f"fixed-{index}",
            "text": f"fixed legal text {index}",
            "source": "law.pdf",
            "strategy": "fixed",
        }
        for index in range(profile.top_k)
    ]
    return {
        "chunks": chunks,
        "debug": {
            "mode": profile.retrieval_mode,
            "chunking_strategy": "fixed",
            "pinecone_namespace": "fixed",
            "top_k": profile.top_k,
            "bm25_count": 0 if profile.semantic_only else 20,
            "semantic_count": 20,
            "fused_count": 20,
            "final_count": profile.top_k,
        },
    }


def test_validate_smoke_result_returns_compact_summary():
    profile = RetrievalProfile("fixed", "hybrid", 3)

    summary = retrieval_smoke_test.validate_smoke_result(
        profile,
        _result(profile),
    )

    assert summary["label"] == "fixed-hybrid-top3"
    assert summary["chunk_count"] == 3
    assert summary["pinecone_namespace"] == "fixed"
    assert summary["chunk_ids"] == ["fixed-0", "fixed-1", "fixed-2"]
    assert "text" not in summary


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pinecone_namespace", "__default__", "pinecone_namespace"),
        ("chunking_strategy", "recursive", "chunking_strategy"),
        ("mode", "semantic", "mode"),
    ],
)
def test_validate_smoke_result_rejects_profile_mismatch(field, value, message):
    profile = RetrievalProfile("fixed", "hybrid", 3)
    result = _result(profile)
    result["debug"][field] = value

    with pytest.raises(RuntimeError, match=message):
        retrieval_smoke_test.validate_smoke_result(profile, result)


def test_run_smoke_executes_all_four_fixed_profiles_and_writes_report(tmp_path):
    output_path = tmp_path / "fixed-smoke.json"

    def fake_retrieve(query, **kwargs):
        profile = RetrievalProfile(
            "fixed",
            "semantic" if kwargs["semantic_only"] else "hybrid",
            kwargs["top_k"],
        )
        return _result(profile)

    span = Mock()
    root_span = SimpleNamespace(
        name="fixed_retrieval_smoke",
        span_id="root-span",
        parent_id=None,
    )
    retriever_span = SimpleNamespace(
        name="hybrid_legal_retrieval",
        span_id="retriever-span",
        parent_id="root-span",
    )
    trace = SimpleNamespace(
        info=SimpleNamespace(trace_id="trace-id"),
        data=SimpleNamespace(spans=[root_span, retriever_span]),
    )
    with (
        patch.object(retrieval_smoke_test, "configure_mlflow"),
        patch.object(
            retrieval_smoke_test,
            "build_retriever",
            return_value={"component": "fixed"},
        ) as build_retriever,
        patch.object(
            retrieval_smoke_test,
            "retrieve",
            side_effect=fake_retrieve,
        ) as retrieve,
        patch.object(
            retrieval_smoke_test.mlflow,
            "start_span",
            side_effect=lambda **_: nullcontext(span),
        ),
        patch.object(
            retrieval_smoke_test.mlflow,
            "get_last_active_trace_id",
            return_value="trace-id",
        ),
        patch.object(retrieval_smoke_test.mlflow, "flush_trace_async_logging"),
        patch.object(
            retrieval_smoke_test.mlflow,
            "get_trace",
            return_value=trace,
        ),
    ):
        report = retrieval_smoke_test.run_fixed_retrieval_smoke(
            output_path=output_path
        )

    build_retriever.assert_called_once_with("fixed")
    assert retrieve.call_count == 4
    assert report["status"] == "passed"
    assert report["profile_count"] == 4
    assert {item["trace_id"] for item in report["results"]} == {"trace-id"}
    assert {item["label"] for item in report["results"]} == {
        "fixed-semantic-top3",
        "fixed-semantic-top5",
        "fixed-hybrid-top3",
        "fixed-hybrid-top5",
    }
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "passed"


def test_run_smoke_rejects_non_fixed_profile_before_loading_models(tmp_path):
    profiles = [RetrievalProfile("recursive", "semantic", 3)]

    with (
        patch.object(retrieval_smoke_test, "build_retriever") as build_retriever,
        pytest.raises(ValueError, match="only fixed"),
    ):
        retrieval_smoke_test.run_fixed_retrieval_smoke(
            profiles=profiles,
            output_path=tmp_path / "unused.json",
        )

    build_retriever.assert_not_called()


def test_validate_nested_trace_rejects_unrelated_retriever_span():
    trace = SimpleNamespace(
        info=SimpleNamespace(trace_id="trace-id"),
        data=SimpleNamespace(
            spans=[
                SimpleNamespace(
                    name="fixed_retrieval_smoke",
                    span_id="root-span",
                    parent_id=None,
                ),
                SimpleNamespace(
                    name="hybrid_legal_retrieval",
                    span_id="retriever-span",
                    parent_id="another-root",
                ),
            ]
        ),
    )

    with (
        patch.object(retrieval_smoke_test.mlflow, "flush_trace_async_logging"),
        patch.object(retrieval_smoke_test.mlflow, "get_trace", return_value=trace),
        pytest.raises(RuntimeError, match="not nested"),
    ):
        retrieval_smoke_test.validate_nested_trace("trace-id")
