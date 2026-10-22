"""Credential-free runtime check for local MLflow runs and GenAI tracing."""

from __future__ import annotations

import json
import time

import mlflow
from mlflow.entities import SpanType

from src.mlflow_config import configure_mlflow


def run_smoke_test() -> dict:
    print("Configuring the local MLflow store...", flush=True)
    experiment = configure_mlflow()
    print(f"Experiment ID: {experiment.experiment_id}", flush=True)

    print("Logging a conventional MLflow run...", flush=True)
    with mlflow.start_run(run_name="mlflow-smoke-test") as run:
        mlflow.log_param("test_type", "credential_free")
        mlflow.log_metric("integration_ok", 1.0)
        run_id = run.info.run_id

    print("Creating a three-span GenAI trace...", flush=True)
    started_ms = int(time.time() * 1000)
    with mlflow.start_span(name="smoke_rag_chain", span_type=SpanType.CHAIN) as chain:
        chain.set_inputs({"question": "Can MLflow store a RAG trace?"})
        with mlflow.start_span(name="smoke_retriever", span_type=SpanType.RETRIEVER) as retriever:
            retriever.set_inputs({"query": "MLflow RAG trace"})
            documents = [{"source": "smoke-test", "text": "MLflow tracing is enabled."}]
            retriever.set_outputs({"documents": documents})
        with mlflow.start_span(name="smoke_llm", span_type=SpanType.LLM) as llm:
            llm.set_inputs({"context": documents, "question": "Is tracing enabled?"})
            answer = "Yes, MLflow tracing is enabled."
            llm.set_outputs({"answer": answer})
        chain.set_outputs({"answer": answer, "documents": documents})

    trace_id = mlflow.get_last_active_trace_id()
    print(f"Trace ID: {trace_id}", flush=True)
    print("Starting trace readback", flush=True)
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id) if trace_id else None

    assert trace is not None, "MLflow did not return the smoke-test trace"
    assert trace.info.request_time >= started_ms, "Retrieved trace predates this smoke test"
    span_names = [span.name for span in trace.data.spans]
    expected = {"smoke_rag_chain", "smoke_retriever", "smoke_llm"}
    assert expected.issubset(span_names), f"Missing expected spans: {expected - set(span_names)}"

    result = {
        "status": "ok",
        "tracking_uri": mlflow.get_tracking_uri(),
        "experiment_id": experiment.experiment_id,
        "run_id": run_id,
        "trace_id": trace.info.trace_id,
        "span_names": span_names,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_smoke_test()
