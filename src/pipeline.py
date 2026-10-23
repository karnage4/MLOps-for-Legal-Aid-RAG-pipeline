"""Traced application boundary around the unchanged retrieval and generation logic."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import mlflow
from mlflow.entities import SpanType

from src.generation import generate_answer
from src.mlflow_config import configure_mlflow
from src.retrieval import RetrievalProfile, retrieve


def run_rag_query(
    query: str,
    retriever_components: Dict[str, Any],
    *,
    top_k: Optional[int] = None,
    semantic_only: Optional[bool] = None,
    temperature: float,
    profile: Optional[RetrievalProfile] = None,
) -> Dict[str, Any]:
    """Run one query and store a nested chain -> retriever -> LLM trace."""
    if profile is None:
        effective_top_k = 5 if top_k is None else top_k
        effective_semantic_only = (
            False if semantic_only is None else semantic_only
        )
        chunking_strategy = retriever_components.get(
            "chunking_strategy", "recursive"
        )
    else:
        if top_k is not None and top_k != profile.top_k:
            raise ValueError("top_k conflicts with the supplied retrieval profile")
        if semantic_only is not None and semantic_only != profile.semantic_only:
            raise ValueError(
                "semantic_only conflicts with the supplied retrieval profile"
            )
        component_strategy = retriever_components.get("chunking_strategy")
        if component_strategy and component_strategy != profile.chunking_strategy:
            raise ValueError(
                "Retriever components were built for "
                f"{component_strategy!r}, not profile strategy "
                f"{profile.chunking_strategy!r}"
            )
        component_namespace = retriever_components.get("pinecone_namespace")
        if component_namespace != profile.pinecone_namespace:
            raise ValueError(
                "Retriever component namespace does not match the supplied "
                "retrieval profile"
            )
        effective_top_k = profile.top_k
        effective_semantic_only = profile.semantic_only
        chunking_strategy = profile.chunking_strategy

    configure_mlflow()
    with mlflow.start_span(name="legal_aid_rag_query", span_type=SpanType.CHAIN) as span:
        span.set_inputs(
            {
                "query": query,
                "top_k": effective_top_k,
                "semantic_only": effective_semantic_only,
                "retrieval_mode": (
                    "semantic" if effective_semantic_only else "hybrid"
                ),
                "chunking_strategy": chunking_strategy,
                "pinecone_namespace": retriever_components.get(
                    "pinecone_namespace"
                ) or "__default__",
                "temperature": temperature,
            }
        )

        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        retrieval = retrieve(
            query,
            **retriever_components,
            top_k=effective_top_k,
            semantic_only=effective_semantic_only,
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
                "chunking_strategy": retrieval.get("debug", {}).get(
                    "chunking_strategy", chunking_strategy
                ),
                "timings": result["timings"],
            }
        )
        return result
