"""Tests for repeatable retrieval latency summaries."""

import pytest

from src import latency_benchmark


def test_percentile_interpolates_and_handles_single_value():
    assert latency_benchmark._percentile([10.0], 0.95) == 10.0
    assert latency_benchmark._percentile([0.0, 100.0], 0.95) == 95.0


def test_summarize_latency_reports_median_and_p95_per_stage():
    records = [
        {"timings_ms": {"bm25_ms": 1.0, "retrieval_total_ms": 10.0}},
        {"timings_ms": {"bm25_ms": 2.0, "retrieval_total_ms": 20.0}},
        {"timings_ms": {"bm25_ms": 3.0, "retrieval_total_ms": 30.0}},
    ]

    summary = latency_benchmark.summarize_latency(records)

    assert summary["sample_count"] == 3
    assert summary["stages"]["bm25_ms"]["median"] == 2.0
    assert summary["stages"]["retrieval_total_ms"]["mean"] == 20.0
    assert summary["stages"]["retrieval_total_ms"]["p95"] == pytest.approx(29.0)


def test_latency_benchmark_rejects_invalid_sample_configuration():
    profile = latency_benchmark.RetrievalProfile("fixed", "hybrid", 5)
    with pytest.raises(ValueError, match="repeats"):
        latency_benchmark.run_latency_benchmark(profile, repeats=0)
    with pytest.raises(ValueError, match="query"):
        latency_benchmark.run_latency_benchmark(profile, queries=[])
