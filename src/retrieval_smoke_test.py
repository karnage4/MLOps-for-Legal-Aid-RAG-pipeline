"""Live retrieval-only smoke tests for the isolated fixed chunk profile."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import mlflow
from mlflow.entities import SpanType

try:
    from src.mlflow_config import configure_mlflow
    from src.retrieval import RetrievalProfile, build_retriever, retrieve
except ImportError:  # supports `python src/retrieval_smoke_test.py`
    from mlflow_config import configure_mlflow
    from retrieval import RetrievalProfile, build_retriever, retrieve


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SMOKE_QUERY = "Who inherits property when there is no will?"
SMOKE_OUTPUT = Path("output/fixed_retrieval_smoke.json")
FIXED_SMOKE_PROFILES = (
    RetrievalProfile("fixed", "semantic", 3),
    RetrievalProfile("fixed", "semantic", 5),
    RetrievalProfile("fixed", "hybrid", 3),
    RetrievalProfile("fixed", "hybrid", 5),
)


def profile_label(profile: RetrievalProfile) -> str:
    return (
        f"{profile.chunking_strategy}-{profile.retrieval_mode}-"
        f"top{profile.top_k}"
    )


def validate_smoke_result(
    profile: RetrievalProfile,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Require exact profile routing and return a compact result summary."""
    debug = result.get("debug", {})
    expected_namespace = profile.pinecone_namespace or "__default__"
    expected_debug = {
        "mode": profile.retrieval_mode,
        "chunking_strategy": profile.chunking_strategy,
        "pinecone_namespace": expected_namespace,
        "top_k": profile.top_k,
    }
    for field, expected in expected_debug.items():
        if debug.get(field) != expected:
            raise RuntimeError(
                f"{profile_label(profile)} reported {field}={debug.get(field)!r}; "
                f"expected {expected!r}"
            )

    chunks = result.get("chunks", [])
    if len(chunks) != profile.top_k:
        raise RuntimeError(
            f"{profile_label(profile)} returned {len(chunks)} chunks; "
            f"expected {profile.top_k}"
        )

    ids = [chunk.get("id") for chunk in chunks]
    if any(not chunk_id for chunk_id in ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f"{profile_label(profile)} returned invalid chunk IDs")
    if any(not str(chunk.get("text", "")).strip() for chunk in chunks):
        raise RuntimeError(f"{profile_label(profile)} returned empty chunk text")

    strategies = {chunk.get("strategy") for chunk in chunks}
    if strategies != {profile.chunking_strategy}:
        raise RuntimeError(
            f"{profile_label(profile)} returned strategies {strategies!r}; "
            f"expected only {profile.chunking_strategy!r}"
        )

    return {
        "label": profile_label(profile),
        "chunking_strategy": profile.chunking_strategy,
        "retrieval_mode": profile.retrieval_mode,
        "top_k": profile.top_k,
        "pinecone_namespace": expected_namespace,
        "chunk_count": len(chunks),
        "bm25_count": debug.get("bm25_count", 0),
        "semantic_count": debug.get("semantic_count", 0),
        "fused_count": debug.get("fused_count", 0),
        "chunk_ids": ids,
        "sources": [chunk.get("source", "unknown") for chunk in chunks],
    }


def validate_nested_trace(trace_id: str) -> Dict[str, Any]:
    """Read back and require the smoke root -> retriever child structure."""
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id) if trace_id else None
    if trace is None:
        raise RuntimeError("MLflow did not return the retrieval smoke trace")

    roots = [
        span for span in trace.data.spans
        if span.name == "fixed_retrieval_smoke"
    ]
    retrievers = [
        span for span in trace.data.spans
        if span.name == "hybrid_legal_retrieval"
    ]
    if len(roots) != 1 or len(retrievers) != 1:
        raise RuntimeError(
            "Retrieval smoke trace must contain exactly one root and one "
            "retriever child"
        )
    if retrievers[0].parent_id != roots[0].span_id:
        raise RuntimeError("Retrieval span is not nested under the smoke root")

    return {
        "trace_id": trace.info.trace_id,
        "root_span_id": roots[0].span_id,
        "retriever_span_id": retrievers[0].span_id,
        "span_names": [span.name for span in trace.data.spans],
    }


def run_fixed_retrieval_smoke(
    query: str = SMOKE_QUERY,
    profiles: Iterable[RetrievalProfile] = FIXED_SMOKE_PROFILES,
    output_path: Path = SMOKE_OUTPUT,
) -> Dict[str, Any]:
    """Run all fixed retrieval combinations without generation or judging."""
    selected_profiles: List[RetrievalProfile] = list(profiles)
    if not selected_profiles:
        raise ValueError("At least one retrieval profile is required")
    if any(profile.chunking_strategy != "fixed" for profile in selected_profiles):
        raise ValueError("This smoke test accepts only fixed retrieval profiles")

    configure_mlflow()
    components = build_retriever("fixed")
    results = []

    for profile in selected_profiles:
        label = profile_label(profile)
        print(f"\nRunning {label}...")
        started = time.perf_counter()
        with mlflow.start_span(
            name="fixed_retrieval_smoke",
            span_type=SpanType.CHAIN,
        ) as span:
            span.set_inputs(
                {
                    "query": query,
                    "profile": label,
                    "chunking_strategy": profile.chunking_strategy,
                    "retrieval_mode": profile.retrieval_mode,
                    "top_k": profile.top_k,
                    "pinecone_namespace": profile.pinecone_namespace,
                    "generation_enabled": False,
                    "judge_enabled": False,
                }
            )
            retrieval_result = retrieve(
                query,
                **components,
                top_k=profile.top_k,
                semantic_only=profile.semantic_only,
            )
            summary = validate_smoke_result(profile, retrieval_result)
            summary["latency_ms"] = round(
                (time.perf_counter() - started) * 1000,
                2,
            )
            span.set_outputs(summary)

        trace_id = mlflow.get_last_active_trace_id()
        summary.update(validate_nested_trace(trace_id))
        results.append(summary)
        print(
            f"Passed: {summary['chunk_count']} fixed chunks from namespace "
            f"'{summary['pinecone_namespace']}' in {summary['latency_ms']} ms; "
            f"trace {summary['trace_id']}"
        )

    report = {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "generation_enabled": False,
        "judge_enabled": False,
        "profile_count": len(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"\nSaved smoke report to {output_path}")
    return report


def main() -> None:
    print("=" * 60)
    print(" Fixed-profile retrieval-only smoke test")
    print("=" * 60)
    report = run_fixed_retrieval_smoke()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
