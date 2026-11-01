"""Tests for the resumable eight-profile GenAI comparison."""

from types import SimpleNamespace

from src import genai_comparison
from src.genai_evaluation import EvaluationProfile


def _report(label="fixed-hybrid-top3", *, correctness=0.0, latency=100.0):
    return {
        "status": "completed",
        "run_id": f"run-{label}",
        "config": {"label": label},
        "dataset": {
            "dataset_id": "dataset-1",
            "dataset_digest": "digest-1",
            "expectation_version": "1",
            "record_count": 3,
        },
        "judges": [{"name": "judge", "version": 2}],
        "metrics": {
            "correctness/mean": correctness,
            "retrieval_sufficiency/mean": 0.5,
            "retrieval_groundedness/mean": 1.0,
            "retrieval_relevance/mean": 0.8,
            "retrieval_relevance/precision/mean": 0.75,
            "relevance_to_query/mean": 1.0,
        },
        "generation": {
            "summary": {"avg_total_ms": latency, "fallback_rate": 1.0}
        },
        "rows": [
            {"legal_answer_quality/value": value}
            for value in ("pass", "fail", "pass")
        ],
        "scorer_failures": [],
    }


def test_profile_matrix_contains_all_eight_configurations():
    assert [profile.label for profile in genai_comparison.PROFILE_MATRIX] == [
        "recursive-semantic-top3",
        "recursive-semantic-top5",
        "recursive-hybrid-top3",
        "recursive-hybrid-top5",
        "fixed-semantic-top3",
        "fixed-semantic-top5",
        "fixed-hybrid-top3",
        "fixed-hybrid-top5",
    ]
    assert EvaluationProfile("recursive", "semantic", 3).pinecone_namespace is None
    assert EvaluationProfile("fixed", "hybrid", 5).pinecone_namespace == "fixed"


def test_reusable_report_requires_matching_lineage():
    profile = EvaluationProfile("fixed", "hybrid", 3)
    dataset = _report()["dataset"]
    judges = [SimpleNamespace(name="judge", scorer_version=2)]

    assert genai_comparison.is_reusable_report(
        _report(), profile=profile, dataset_report=dataset, judges=judges
    )

    stale = _report()
    stale["dataset"]["dataset_digest"] = "old"
    assert not genai_comparison.is_reusable_report(
        stale, profile=profile, dataset_report=dataset, judges=judges
    )


def test_quality_ranking_precedes_latency_and_is_deterministic():
    slow_better = genai_comparison.score_card(
        _report("recursive-hybrid-top5", correctness=1.0, latency=500.0)
    )
    fast_worse = genai_comparison.score_card(
        _report("fixed-semantic-top3", correctness=0.0, latency=10.0)
    )
    assert genai_comparison.rank_score_cards([fast_worse, slow_better])[0]["label"] == (
        "recursive-hybrid-top5"
    )

    equal_a = dict(slow_better, label="b-profile")
    equal_b = dict(slow_better, label="a-profile")
    assert genai_comparison.rank_score_cards([equal_a, equal_b])[0]["label"] == (
        "a-profile"
    )


def test_score_card_calculates_categorical_legal_quality_rate():
    card = genai_comparison.score_card(_report())
    assert card["legal_answer_quality"] == 2 / 3


def test_retrying_scorer_retries_transient_failure(monkeypatch):
    monkeypatch.setattr(genai_comparison.time, "sleep", lambda _: None)

    class FlakyJudge:
        name = "flaky"
        description = "test"
        aggregations = None
        pass_if = None
        timeout = None

        def __init__(self):
            self.calls = 0

        def run(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("Failed to generate JSON")
            return True

    judge = FlakyJudge()
    wrapped = genai_comparison.build_retrying_scorers([judge], max_attempts=3)[0]

    assert wrapped.run(inputs={"query": "test"}, outputs="answer") is True
    assert judge.calls == 3
    assert wrapped.timeout == genai_comparison.SCORER_TIMEOUT_SECONDS
