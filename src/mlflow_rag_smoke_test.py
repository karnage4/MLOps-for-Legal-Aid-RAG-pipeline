"""Run one real RAG query and verify its production MLflow trace."""

from __future__ import annotations

import json
import os
import time

import mlflow

from src.generation import GROQ_FALLBACK_MODEL, PRIMARY_MODEL
from src.mlflow_config import configure_mlflow
from src.pipeline import run_rag_query
from src.retrieval import build_retriever


TEST_QUERY = "What share does a daughter receive in Islamic inheritance?"


def run_real_rag_smoke_test(query: str = TEST_QUERY) -> dict:
    print("Loading hybrid retrieval components...", flush=True)
    components = build_retriever()

    print(f"Running traced query: {query}", flush=True)
    experiment = configure_mlflow()
    started_ms = int(time.time() * 1000)
    with mlflow.start_run(run_name="real-rag-smoke-test") as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "test_type": "real_rag_query",
                "query": query,
                "retrieval_mode": "hybrid",
                "top_k": 5,
                "huggingface_primary_model": PRIMARY_MODEL,
                "groq_fallback_model": GROQ_FALLBACK_MODEL,
            }
        )
        result = run_rag_query(
            query,
            components,
            top_k=5,
            semantic_only=False,
            temperature=0.2,
        )

        chunks = result["retrieval"]["chunks"]
        generation = result["generation"]
        generation_error = generation["error"] if generation else "Generation did not run"
        answer = generation["answer"].strip() if generation else ""

        trace_id = mlflow.get_last_active_trace_id()
        print(f"Trace ID: {trace_id}", flush=True)
        print("Starting trace readback", flush=True)
        mlflow.flush_trace_async_logging()
        trace = mlflow.get_trace(trace_id) if trace_id else None

        assert trace is not None, "MLflow did not return the real-query trace"
        assert trace.info.request_time >= started_ms, "Retrieved trace predates this test"
        span_names = [span.name for span in trace.data.spans]
        expected = {
            "legal_aid_rag_query",
            "hybrid_legal_retrieval",
            "legal_generation",
        }
        assert expected.issubset(span_names), (
            f"Missing expected spans: {expected - set(span_names)}"
        )

        serialized_spans = repr(
            [(span.inputs, span.outputs, span.attributes) for span in trace.data.spans]
        )
        for variable_name in ("PINECONE_API_KEY", "HF_API_TOKEN", "GROQ_API_KEY"):
            secret = os.getenv(variable_name)
            assert not secret or secret not in serialized_spans, (
                f"Secret {variable_name} was recorded in the trace"
            )

        test_passed = len(chunks) == 5 and generation_error is None and bool(answer)
        mlflow.log_metrics(
            {
                "test_passed": float(test_passed),
                "retrieved_chunks": float(len(chunks)),
                "retrieval_ms": float(result["timings"]["retrieval_ms"]),
                "generation_ms": float(result["timings"]["generation_ms"]),
                "total_ms": float(result["timings"]["total_ms"]),
                "provider_latency_ms": float(
                    generation.get("provider_latency_ms", 0) if generation else 0
                ),
                "fallback_used": float(
                    bool(generation and generation.get("fallback_used"))
                ),
            }
        )
        mlflow.set_tags(
            {
                "test.status": "passed" if test_passed else "failed",
                "trace_id": trace.info.trace_id,
                "generation.provider": generation.get("provider") or "none",
                "generation.model": generation.get("model_used") or "none",
            }
        )

        summary = {
            "status": "ok" if test_passed else "generation_failed",
            "experiment_id": experiment.experiment_id,
            "run_id": run_id,
            "trace_id": trace.info.trace_id,
            "span_names": span_names,
            "retrieval_mode": result["retrieval"]["debug"]["mode"],
            "retrieved_chunks": len(chunks),
            "sources": sorted({chunk.get("source", "unknown") for chunk in chunks}),
            "provider": generation.get("provider") if generation else None,
            "model_used": generation.get("model_used") if generation else None,
            "fallback_used": generation.get("fallback_used") if generation else None,
            "provider_attempts": generation.get("attempts", []) if generation else [],
            "timings_ms": result["timings"],
            "provider_latency_ms": (
                generation.get("provider_latency_ms") if generation else None
            ),
            "answer_preview": answer[:300],
            "generation_error": generation_error,
            "secrets_recorded": False,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    assert len(chunks) == 5, f"Expected 5 retrieved chunks, got {len(chunks)}"
    assert generation_error is None, generation_error
    assert answer, "Generation returned an empty answer"
    return summary


if __name__ == "__main__":
    run_real_rag_smoke_test()
