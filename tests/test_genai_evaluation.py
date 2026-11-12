"""Tests for the controlled MLflow GenAI evaluation runner."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src import genai_evaluation
from src.evaluation_expectations import EXPECTED_FACTS_BY_QUERY


class FakeDataset:
    name = "legal-aid-rag-held-out-v1"
    dataset_id = "d-test"
    digest = "digest-test"

    def to_df(self):
        return pd.DataFrame(
            [
                {
                    "inputs": {"query": query},
                    "expectations": {"expected_facts": facts},
                }
                for query, facts in EXPECTED_FACTS_BY_QUERY.items()
            ]
        )


def test_validate_managed_dataset_requires_exact_approved_facts():
    report = genai_evaluation.validate_managed_dataset(FakeDataset())
    assert report["record_count"] == 3
    assert report["expectation_version"] == "1"

    invalid = FakeDataset()
    invalid.to_df = lambda: pd.DataFrame(
        [{"inputs": {"query": "Different"}, "expectations": {"expected_facts": []}}]
    )
    with pytest.raises(ValueError, match="approved expectations"):
        genai_evaluation.validate_managed_dataset(invalid)


def test_load_registered_judges_pins_latest_versions(monkeypatch):
    calls = []

    def fake_get_scorer(*, name, experiment_id):
        calls.append((name, experiment_id))
        return SimpleNamespace(name=name, scorer_version=1)

    monkeypatch.setattr(genai_evaluation, "get_scorer", fake_get_scorer)
    judges = genai_evaluation.load_registered_judges("7", ("one", "two"))

    assert [judge.scorer_version for judge in judges] == [1, 1]
    assert calls == [("one", "7"), ("two", "7")]


def test_build_predict_fn_preserves_profile_and_records_provider(monkeypatch):
    observed = {}

    def fake_run(query, components, *, temperature, profile):
        observed.update(
            query=query,
            components=components,
            temperature=temperature,
            profile=profile,
        )
        return {
            "retrieval": {"chunks": [{"text": "context"}] * 3},
            "generation": {
                "answer": "answer",
                "error": None,
                "provider": "groq",
                "model_used": "model",
                "fallback_used": True,
            },
            "timings": {"retrieval_ms": 1, "generation_ms": 2, "total_ms": 3},
        }

    monkeypatch.setattr(genai_evaluation, "run_rag_query", fake_run)
    records = []
    predict = genai_evaluation.build_predict_fn({"component": "value"}, call_records=records)

    assert predict("question") == "answer"
    assert observed["profile"] == genai_evaluation.PROFILE
    assert observed["profile"].chunking_strategy == "fixed"
    assert observed["profile"].retrieval_mode == "hybrid"
    assert observed["profile"].top_k == 3
    assert records[0]["provider"] == "groq"
    assert records[0]["fallback_used"] is True


def test_build_predict_fn_fails_closed_on_generation_error(monkeypatch):
    monkeypatch.setattr(
        genai_evaluation,
        "run_rag_query",
        lambda *args, **kwargs: {
            "retrieval": {"chunks": []},
            "generation": {"answer": None, "error": "provider failed"},
            "timings": {},
        },
    )
    predict = genai_evaluation.build_predict_fn({}, call_records=[])
    with pytest.raises(RuntimeError, match="provider failed"):
        predict("question")


def test_scorer_failures_requires_every_judge_value():
    complete = {
        f"{name}/value": "yes" for name in genai_evaluation.JUDGE_NAMES
    }
    assert genai_evaluation._scorer_failures([complete]) == []

    complete["correctness/value"] = None
    assert "correctness" in genai_evaluation._scorer_failures([complete])[0]


def test_scorer_failures_accepts_core_judge_subset():
    core_names = ("relevance_to_query", "fact_coverage")
    row = {f"{name}/value": 1.0 for name in core_names}

    assert genai_evaluation._scorer_failures([row], core_names) == []


@pytest.mark.parametrize(
    "row",
    [
        {"inputs": {"query": "from inputs"}},
        {"request": '{"query": "from request"}'},
        {"inputs/query": "from flattened inputs"},
    ],
)
def test_compact_evaluation_rows_preserves_query_across_mlflow_shapes(row):
    row["trace_id"] = "trace-1"
    compact = genai_evaluation._compact_evaluation_rows(pd.DataFrame([row]))

    assert compact[0]["query"].startswith("from ")


def test_generation_summary_aggregates_latency_and_fallback_rate():
    summary = genai_evaluation._generation_summary(
        [
            {
                "fallback_used": True,
                "retrieval_ms": 10,
                "generation_ms": 2,
                "total_ms": 12,
            },
            {
                "fallback_used": False,
                "retrieval_ms": 20,
                "generation_ms": 4,
                "total_ms": 24,
            },
        ]
    )

    assert summary["fallback_rate"] == 0.5
    assert summary["avg_retrieval_ms"] == 15.0
    assert summary["avg_generation_ms"] == 3.0
    assert summary["avg_total_ms"] == 18.0
    assert summary["avg_bm25_ms"] == 0.0
    assert summary["avg_rerank_ms"] == 0.0

