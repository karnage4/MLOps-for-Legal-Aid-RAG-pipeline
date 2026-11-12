"""Repeatable retrieval latency benchmark with warmup, median, and p95 metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlflow

from src.evaluation_queries import TEST_QUERIES
from src.mlflow_config import configure_mlflow
from src.retrieval import (
    BM25_TOP_K,
    RERANK_CANDIDATE_K,
    SEMANTIC_TOP_K,
    RetrievalProfile,
    build_retriever,
    retrieve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_NAMES = (
    "bm25_ms",
    "embedding_ms",
    "pinecone_ms",
    "semantic_total_ms",
    "fusion_ms",
    "rerank_ms",
    "retrieval_total_ms",
)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return an interpolated percentile without requiring NumPy."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_latency(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate warm request timings into median and p95 stage statistics."""
    summary: dict[str, Any] = {"sample_count": len(records), "stages": {}}
    for stage_name in STAGE_NAMES:
        values = [
            float(record.get("timings_ms", {}).get(stage_name, 0.0))
            for record in records
        ]
        summary["stages"][stage_name] = {
            "mean": round(statistics.fmean(values), 3) if values else 0.0,
            "median": round(statistics.median(values), 3) if values else 0.0,
            "p95": round(_percentile(values, 0.95), 3),
            "min": round(min(values), 3) if values else 0.0,
            "max": round(max(values), 3) if values else 0.0,
        }
    return summary


def run_latency_benchmark(
    profile: RetrievalProfile,
    *,
    repeats: int = 3,
    queries: Sequence[str] = TEST_QUERIES,
) -> dict[str, Any]:
    """Benchmark warm retrieval calls without consuming generation/judge quota."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if not queries:
        raise ValueError("at least one query is required")

    configure_mlflow()
    load_started = time.perf_counter()
    components = build_retriever(profile.chunking_strategy)
    component_load_ms = round((time.perf_counter() - load_started) * 1000, 3)

    warmup = retrieve(
        queries[0],
        **components,
        top_k=profile.top_k,
        semantic_only=profile.semantic_only,
    )
    records = []

    with mlflow.start_run(run_name=f"latency-benchmark-{profile.label}") as run:
        mlflow.log_params(
            {
                "benchmark_type": "warm_retrieval_latency",
                "chunking_strategy": profile.chunking_strategy,
                "retrieval_mode": profile.retrieval_mode,
                "top_k": profile.top_k,
                "repeats": repeats,
                "query_count": len(queries),
                "bm25_candidate_k": BM25_TOP_K,
                "semantic_candidate_k": SEMANTIC_TOP_K,
                "rerank_candidate_k": RERANK_CANDIDATE_K,
                "generation_enabled": False,
                "judge_enabled": False,
            }
        )
        for repeat_index in range(1, repeats + 1):
            for query in queries:
                result = retrieve(
                    query,
                    **components,
                    top_k=profile.top_k,
                    semantic_only=profile.semantic_only,
                )
                records.append(
                    {
                        "repeat": repeat_index,
                        "query": query,
                        "chunk_ids": [chunk.get("id") for chunk in result["chunks"]],
                        "timings_ms": result["debug"]["timings_ms"],
                    }
                )

        summary = summarize_latency(records)
        metrics = {"component_load_ms": component_load_ms}
        for stage_name, values in summary["stages"].items():
            metrics[f"{stage_name}_median"] = values["median"]
            metrics[f"{stage_name}_p95"] = values["p95"]
        mlflow.log_metrics(metrics)

        report = {
            "status": "completed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run.info.run_id,
            "profile": {
                "label": profile.label,
                "chunking_strategy": profile.chunking_strategy,
                "retrieval_mode": profile.retrieval_mode,
                "top_k": profile.top_k,
            },
            "candidate_limits": {
                "bm25": BM25_TOP_K,
                "semantic": SEMANTIC_TOP_K,
                "rerank": RERANK_CANDIDATE_K,
            },
            "component_load_ms": component_load_ms,
            "warmup_timings_ms": warmup["debug"]["timings_ms"],
            "summary": summary,
            "records": records,
        }
        mlflow.log_dict(report, "latency_benchmark.json")

    output_path = PROJECT_ROOT / "output" / f"latency_benchmark_{profile.label}.json"
    artifact_path = PROJECT_ROOT / "artifacts" / f"latency_benchmark_{profile.label}.json"
    payload = json.dumps(report, indent=2)
    for path in (output_path, artifact_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("recursive", "fixed"), default="fixed")
    parser.add_argument("--mode", choices=("semantic", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, choices=(3, 5), default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    profile = RetrievalProfile(args.strategy, args.mode, args.top_k)
    print(json.dumps(run_latency_benchmark(profile, repeats=args.repeats), indent=2))


if __name__ == "__main__":
    main()
