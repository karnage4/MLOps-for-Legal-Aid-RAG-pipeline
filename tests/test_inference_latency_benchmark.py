from src.evaluation_queries import MANAGED_EVALUATION_QUERIES
from src.inference_ablation import INFERENCE_CONFIGS
from src.inference_latency_benchmark import _rotated_configs, summarize_records


def test_latency_benchmark_uses_managed_three_query_evaluation_set():
    assert MANAGED_EVALUATION_QUERIES == (
        "Who inherits property when a person dies without a will?",
        "What is the share of a daughter in Islamic inheritance?",
        "Can a non-Muslim inherit from a Muslim's estate?",
    )


def test_summarize_records_separates_stage_latency_and_retries():
    records = [
        {
            "success": True,
            "retrieval_ms": 100,
            "generation_ms": 20,
            "total_ms": 120,
            "request_attempts": 1,
        },
        {
            "success": True,
            "retrieval_ms": 200,
            "generation_ms": 40,
            "total_ms": 240,
            "request_attempts": 2,
        },
        {
            "success": False,
            "retrieval_ms": 300,
            "generation_ms": 60,
            "total_ms": 360,
            "request_attempts": 3,
        },
    ]

    result = summarize_records(records)

    assert result["sample_count"] == 3
    assert result["successful_count"] == 2
    assert result["failure_count"] == 1
    assert result["retried_call_count"] == 2
    assert result["retrieval_ms"]["median"] == 150
    assert result["generation_ms"]["median"] == 30
    assert result["total_ms"]["median"] == 180


def test_rotated_configs_changes_quota_position_without_changing_membership():
    first = _rotated_configs(INFERENCE_CONFIGS, 0)
    second = _rotated_configs(INFERENCE_CONFIGS, 1)

    assert second[0] == first[1]
    assert set(second) == set(first)
