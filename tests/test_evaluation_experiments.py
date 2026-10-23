"""Unit tests for comparable RAG experiment summaries and selection."""

from unittest.mock import patch

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
        patch.object(evaluation, "log_evaluation_summary", return_value="run-id"),
        patch.object(evaluation.time, "sleep"),
    ):
        experiment = evaluation.run_evaluation(
            ["Legal question?"],
            {"component": "value"},
            object(),
            config_label="trace-test",
            semantic_only=False,
            top_k=5,
            temperature=0.2,
        )

    traced_rag.assert_called_once_with(
        "Legal question?",
        {"component": "value"},
        top_k=5,
        semantic_only=False,
        temperature=0.2,
    )
    result = experiment["results"][0]
    assert result["answer"] == "Grounded answer"
    assert result["timings"]["retrieval_ms"] == 100
    assert result["timings"]["generation_ms"] == 200


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


def test_experiment_suite_runs_distinct_configs_and_marks_best():
    queries = ["query one", "query two"]
    simulated = {
        "A-semantic-top5": _summary(0.7, 0.8, 1000),
        "B-hybrid-top3": _summary(0.8, 0.75, 1200),
        "C-hybrid-top5": _summary(0.8, 0.8, 1500),
    }

    def fake_run(test_queries, components, model, config_label, **kwargs):
        assert test_queries == queries
        return {
            "config": {"label": config_label, **kwargs},
            "results": [],
            "summary": simulated[config_label],
            "run_id": f"run-{config_label}",
        }

    with (
        patch.object(evaluation, "run_evaluation", side_effect=fake_run) as run,
        patch.object(evaluation, "mark_best_evaluation_run") as mark_best,
    ):
        suite = evaluation.run_experiment_suite(queries, {}, object())

    assert run.call_count == 3
    called_configs = {
        call.kwargs["config_label"] for call in run.call_args_list
    }
    assert called_configs == {
        "A-semantic-top5",
        "B-hybrid-top3",
        "C-hybrid-top5",
    }
    assert suite["best_configuration"] == "C-hybrid-top5"
    assert suite["best_run_id"] == "run-C-hybrid-top5"
    mark_best.assert_called_once()
