"""Warm prompt/model/temperature latency benchmark without evaluation judges."""

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

from src.evaluation_queries import MANAGED_EVALUATION_QUERIES
from src.generation import generate_answer
from src.inference_ablation import BASELINE_PROFILE, INFERENCE_CONFIGS, InferenceConfig
from src.mlflow_config import configure_mlflow
from src.retrieval import build_retriever, retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "output" / "inference_latency_benchmark_core_v2.json"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "inference_latency_benchmark_core_v2.json"


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize successful calls while preserving retry and failure counts."""
    successful = [record for record in records if record.get("success")]
    summary: dict[str, Any] = {
        "sample_count": len(records),
        "successful_count": len(successful),
        "failure_count": len(records) - len(successful),
        "retried_call_count": sum(int(record.get("request_attempts", 0)) > 1 for record in records),
    }
    for name in ("retrieval_ms", "generation_ms", "total_ms"):
        values = [float(record[name]) for record in successful]
        summary[name] = {
            "mean": round(statistics.fmean(values), 3) if values else 0.0,
            "median": round(statistics.median(values), 3) if values else 0.0,
            "p95": round(_percentile(values, 0.95), 3),
            "min": round(min(values), 3) if values else 0.0,
            "max": round(max(values), 3) if values else 0.0,
        }
    return summary


def _request_attempts(result: Mapping[str, Any]) -> int:
    return sum(
        int(attempt.get("request_attempts", 0))
        for attempt in result.get("attempts", [])
        if attempt.get("provider") == "groq"
    )


def _rotated_configs(
    configs: Sequence[InferenceConfig], offset: int
) -> list[InferenceConfig]:
    """Rotate call order so no configuration always occupies a quota position."""
    start = offset % len(configs)
    return [*configs[start:], *configs[:start]]


def run_inference_latency_benchmark(
    *,
    repeats: int = 2,
    queries: Sequence[str] = MANAGED_EVALUATION_QUERIES,
    configs: Sequence[InferenceConfig] = INFERENCE_CONFIGS,
    cooldown_seconds: float = 15.0,
) -> dict[str, Any]:
    """Benchmark generation configs against identical warm retrieved contexts."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if not queries:
        raise ValueError("at least one query is required")
    if not configs:
        raise ValueError("at least one inference config is required")
    if cooldown_seconds < 0:
        raise ValueError("cooldown_seconds must not be negative")

    configure_mlflow()
    components = build_retriever(BASELINE_PROFILE.chunking_strategy)
    # Warm all local models and the Pinecone connection outside measured calls.
    retrieve(
        queries[0],
        **components,
        top_k=BASELINE_PROFILE.top_k,
        semantic_only=BASELINE_PROFILE.semantic_only,
    )
    records: dict[str, list[dict[str, Any]]] = {config.label: [] for config in configs}

    with mlflow.start_run(run_name="inference-latency-benchmark-core-v2") as run:
        mlflow.log_params(
            {
                "benchmark_type": "warm_controlled_inference_latency",
                "retrieval_profile": BASELINE_PROFILE.label,
                "repeats": repeats,
                "query_count": len(queries),
                "config_count": len(configs),
                "judge_enabled": False,
                "shared_retrieval_context": True,
                "unmeasured_generation_cooldown_seconds": cooldown_seconds,
            }
        )
        for repeat_index in range(1, repeats + 1):
            for query_index, query in enumerate(queries):
                retrieval = retrieve(
                    query,
                    **components,
                    top_k=BASELINE_PROFILE.top_k,
                    semantic_only=BASELINE_PROFILE.semantic_only,
                )
                retrieval_ms = float(
                    retrieval["debug"]["timings_ms"]["retrieval_total_ms"]
                )
                chunks = retrieval["chunks"]
                ordered_configs = _rotated_configs(
                    configs, (repeat_index - 1) * len(queries) + query_index
                )
                for config_index, config in enumerate(ordered_configs):
                    generation = generate_answer(
                        query,
                        chunks,
                        temperature=config.temperature,
                        **config.generation_config,
                    )
                    generation_ms = float(generation.get("latency_ms", 0.0))
                    records[config.label].append(
                        {
                            "repeat": repeat_index,
                            "query": query,
                            "retrieval_ms": retrieval_ms,
                            "generation_ms": generation_ms,
                            "total_ms": retrieval_ms + generation_ms,
                            "request_attempts": _request_attempts(generation),
                            "success": generation.get("error") is None,
                            "error": generation.get("error"),
                            "provider": generation.get("provider"),
                            "model": generation.get("model_used"),
                        }
                    )
                    if cooldown_seconds and not (
                        repeat_index == repeats
                        and query_index == len(queries) - 1
                        and config_index == len(ordered_configs) - 1
                    ):
                        time.sleep(cooldown_seconds)

        summaries = {
            config.label: summarize_records(records[config.label]) for config in configs
        }
        for label, summary in summaries.items():
            mlflow.log_metrics(
                {
                    f"{label}.generation_median_ms": summary["generation_ms"]["median"],
                    f"{label}.generation_p95_ms": summary["generation_ms"]["p95"],
                    f"{label}.total_median_ms": summary["total_ms"]["median"],
                    f"{label}.retried_calls": summary["retried_call_count"],
                }
            )
        report = {
            "status": "completed" if all(not item["failure_count"] for item in summaries.values()) else "partial",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run.info.run_id,
            "retrieval_profile": BASELINE_PROFILE.label,
            "repeats": repeats,
            "query_count": len(queries),
            "judge_enabled": False,
            "shared_retrieval_context": True,
            "unmeasured_generation_cooldown_seconds": cooldown_seconds,
            "configs": {
                config.label: {
                    "prompt_variant": config.prompt_variant,
                    "model": config.model,
                    "temperature": config.temperature,
                    "summary": summaries[config.label],
                    "records": records[config.label],
                }
                for config in configs
            },
        }
        mlflow.log_dict(report, "inference_latency_benchmark_core_v2.json")

    payload = json.dumps(report, indent=2)
    for path in (OUTPUT_PATH, ARTIFACT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cooldown-seconds", type=float, default=15.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run_inference_latency_benchmark(
                repeats=args.repeats,
                cooldown_seconds=args.cooldown_seconds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
