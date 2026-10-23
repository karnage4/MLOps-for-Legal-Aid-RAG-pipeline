"""Traced application boundary around the unchanged retrieval and generation logic."""

from __future__ import annotations

import time
from typing import Any, Dict

import mlflow
from mlflow.entities import SpanType

from src.generation import generate_answer
from src.mlflow_config import configure_mlflow
from src.retrieval import retrieve


def run_rag_query(
    query: str,
    retriever_components: Dict[str, Any],
    *,
    top_k: int,
    semantic_only: bool,
    temperature: float,
) -> Dict[str, Any]:
    """Run one query and store a nested chain -> retriever -> LLM trace."""
    configure_mlflow()
    with mlflow.start_span(name="legal_aid_rag_query", span_type=SpanType.CHAIN) as span:
        span.set_inputs(
            {
                "query": query,
                "top_k": top_k,
                "semantic_only": semantic_only,
                "temperature": temperature,
            }
        )

        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        retrieval = retrieve(
            query,
            **retriever_components,
            top_k=top_k,
            semantic_only=semantic_only,
        )
        retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000)

        chunks = retrieval["chunks"]
        if chunks:
            generation_started = time.perf_counter()
            generation = generate_answer(query, chunks, temperature=temperature)
            generation_ms = round((time.perf_counter() - generation_started) * 1000)
        else:
            generation = None
            generation_ms = 0

        result = {
            "retrieval": retrieval,
            "generation": generation,
            "timings": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": round((time.perf_counter() - started) * 1000),
            },
        }
        # Detailed retrieval and generation payloads live on their child spans.
        # Keep the root summary compact so SQLite trace persistence is reliable.
        span.set_outputs(
            {
                "answer": generation.get("answer") if generation else None,
                "generation_provider": (
                    generation.get("provider") if generation else None
                ),
                "generation_model": (
                    generation.get("model_used") if generation else None
                ),
                "generation_error": (
                    generation.get("error") if generation else None
                ),
                "retrieved_chunks": len(chunks),
                "retrieval_mode": retrieval.get("debug", {}).get("mode"),
                "timings": result["timings"],
            }
        )
        return result
