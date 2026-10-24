"""Unit tests for comparable RAG experiment summaries and selection."""

from unittest.mock import patch

import pytest

from src import evaluation


def _summary(faithfulness: float, relevancy: float, latency: float) -> dict:
    return {
        "successful_queries": 2,
        "failed_queries": 0,
        "faithfulness_mean": faithfulness,
        "relevancy_mean": relevancy,
        "avg_total_ms": latency,
    }


def test_summarize_results_exposes_failures_and_latency():
    results = [
        {
            "error": None,
            "faithfulness": {"score": 0.75},
            "relevancy": {"score": 0.5},
            "timings": {
                "retrieval_ms": 100,
                "generation_ms": 300,
                "total_ms": 500,
            },
        },
        {"error": "provider unavailable"},
    ]

    summary = evaluation.summarize_results(results, "test-config")

    assert summary["n_queries"] == 2
    assert summary["successful_queries"] == 1
    assert summary["failed_queries"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["faithfulness_mean"] == 0.75
    assert summary["relevancy_mean"] == 0.5
    assert summary["avg_total_ms"] == 500.0


def test_batch_claim_verification_requires_one_verdict_per_claim():
    response = "1. SUPPORTED\n2. NOT_SUPPORTED"
    with patch.object(evaluation, "call_judge", return_value=response):
        verdicts = evaluation.verify_claims(
            ["First claim", "Second claim"],
            "Supporting legal context",
        )

    assert verdicts == [True, False]


def test_batch_claim_verification_rejects_incomplete_judge_output():
    with patch.object(evaluation, "call_judge", return_value="1. SUPPORTED"):
        verdicts = evaluation.verify_claims(
            ["First claim", "Second claim"],
            "Supporting legal context",
        )

    assert verdicts is None


def test_run_evaluation_uses_traced_rag_boundary():
    profile = evaluation.RetrievalProfile("recursive", "hybrid", 5)
    rag_result = {
        "retrieval": {
            "chunks": [{"source": "law.txt", "text": "Legal context"}],
            "debug": {"mode": "hybrid"},
        },
        "generation": {
            "answer": "Grounded answer",
            "model_used": "test-model",
            "error": None,
        },
        "timings": {
            "retrieval_ms": 100,
            "generation_ms": 200,
            "total_ms": 300,
        },
    }
    faithfulness = {
        "score": 1.0,
        "total_claims": 1,
        "supported_claims": 1,
        "claims": ["Grounded claim"],
        "verifications": [{"claim": "Grounded claim", "supported": True}],
        "error": None,
    }
    relevancy = {
        "score": 0.9,
        "generated_questions": ["Question?"],
        "similarities": [0.9],
        "error": None,
    }

    with (
        patch.object(evaluation, "run_rag_query", return_value=rag_result) as traced_rag,
        patch.object(
            evaluation, "evaluate_faithfulness", return_value=faithfulness
        ),
        patch.object(evaluation, "evaluate_relevancy", return_value=relevancy),
        patch.object(
            evaluation, "log_evaluation_summary", return_value="run-id"
        ) as log_summary,
        patch.object(evaluation.time, "sleep"),
    ):
        experiment = evaluation.run_evaluation(
            ["Legal question?"],
            {"component": "value"},
            object(),
            config_label="trace-test",
            profile=profile,
            temperature=0.2,
        )

    traced_rag.assert_called_once_with(
        "Legal question?",
        {"component": "value"},
        temperature=0.2,
        profile=profile,
    )
    result = experiment["results"][0]
    assert result["answer"] == "Grounded answer"
    assert result["timings"]["retrieval_ms"] == 100
    assert result["timings"]["generation_ms"] == 200
    assert result["config"]["chunking_strategy"] == "recursive"
    assert result["config"]["retrieval_mode"] == "hybrid"
    logged_params = log_summary.call_args.kwargs["params"]
    logged_tags = log_summary.call_args.kwargs["tags"]
    assert logged_params["chunking_strategy"] == "recursive"
    assert logged_params["retrieval_mode"] == "hybrid"
    assert logged_params["top_k"] == 5
    assert logged_params["pinecone_namespace"] == "__default__"
    assert logged_tags["evaluation.chunking_strategy"] == "recursive"
    assert logged_tags["evaluation.retrieval_mode"] == "hybrid"


def test_best_configuration_uses_documented_tie_breakers():
    experiments = {
        "A": {"summary": _summary(0.7, 0.9, 1000)},
        "B": {"summary": _summary(0.8, 0.7, 2000)},
        "C": {"summary": _summary(0.8, 0.7, 1500)},
    }

    assert evaluation.select_best_configuration(experiments) == "C"


def test_best_configuration_excludes_partial_runs():
    partial = _summary(1.0, 1.0, 100)
    partial["successful_queries"] = 1
    partial["failed_queries"] = 1
    experiments = {
        "partial": {"summary": partial},
        "complete": {"summary": _summary(0.8, 0.8, 1000)},
    }

    assert evaluation.select_best_configuration(experiments) == "complete"


def test_experiment_matrix_is_the_complete_cartesian_product():
    profiles = [config["profile"] for config in evaluation.EXPERIMENT_CONFIGS]

    assert len(profiles) == 8
    assert len({config["label"] for config in evaluation.EXPERIMENT_CONFIGS}) == 8
    assert {
        (profile.chunking_strategy, profile.retrieval_mode, profile.top_k)
        for profile in profiles
    } == {
        (strategy, mode, top_k)
        for strategy in ("recursive", "fixed")
        for mode in ("semantic", "hybrid")
        for top_k in (3, 5)
    }


def test_experiment_suite_runs_eight_profiles_and_marks_best():
    queries = ["query one", "query two"]
    simulated = {
        config["label"]: _summary(0.7, 0.7, 1500)
        for config in evaluation.EXPERIMENT_CONFIGS
    }
    simulated["fixed-hybrid-top5"] = _summary(0.9, 0.8, 1000)
    retrievers = {
        "recursive": {"chunking_strategy": "recursive"},
        "fixed": {"chunking_strategy": "fixed"},
    }

    def fake_run(test_queries, components, model, config_label, **kwargs):
        assert test_queries == queries
        assert components["chunking_strategy"] == kwargs["profile"].chunking_strategy
        return {
            "config": {"label": config_label},
            "results": [],
            "summary": simulated[config_label],
            "run_id": f"run-{config_label}",
        }

    with (
        patch.object(evaluation, "run_evaluation", side_effect=fake_run) as run,
        patch.object(evaluation, "mark_best_evaluation_run") as mark_best,
    ):
        suite = evaluation.run_experiment_suite(queries, retrievers, object())

    assert run.call_count == 8
    called_profiles = {
        (
            call.kwargs["profile"].chunking_strategy,
            call.kwargs["profile"].retrieval_mode,
            call.kwargs["profile"].top_k,
        )
        for call in run.call_args_list
    }
    assert called_profiles == {
        (strategy, mode, top_k)
        for strategy in ("recursive", "fixed")
        for mode in ("semantic", "hybrid")
        for top_k in (3, 5)
    }
    assert suite["configuration_count"] == 8
    assert suite["best_configuration"] == "fixed-hybrid-top5"
    assert suite["best_run_id"] == "run-fixed-hybrid-top5"
    mark_best.assert_called_once()


def test_experiment_suite_requires_both_strategy_retrievers():
    with pytest.raises(ValueError, match="fixed"):
        evaluation.run_experiment_suite(
            ["query"],
            {"recursive": {}},
            object(),
        )
