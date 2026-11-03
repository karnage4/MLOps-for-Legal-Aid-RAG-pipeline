"""FastAPI service for the existing traced Legal-Aid RAG pipeline."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_CHUNKING_STRATEGY = "recursive"
DEFAULT_RETRIEVAL_MODE = "hybrid"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.2


def build_retriever(chunking_strategy: str) -> dict[str, Any]:
    """Lazy adapter that keeps heavy NLP imports out of API module import."""
    from src.retrieval import build_retriever as implementation

    return implementation(chunking_strategy)


def build_profile(
    *,
    retrieval_mode: str,
    top_k: int,
    chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY,
) -> Any:
    """Build the canonical profile only when an actual prediction is made."""
    from src.retrieval import RetrievalProfile

    return RetrievalProfile(
        chunking_strategy=chunking_strategy,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
    )


def run_rag_query(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy adapter around the unchanged, MLflow-traced RAG boundary."""
    from src.pipeline import run_rag_query as implementation

    return implementation(*args, **kwargs)


@lru_cache(maxsize=2)
def load_retriever_components(
    chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY,
) -> dict[str, Any]:
    """Load and cache one retriever per chunking strategy."""
    components = build_retriever(chunking_strategy)
    required = {
        "bm25",
        "bm25_ids",
        "bm25_texts",
        "pinecone_index",
        "embedder",
        "reranker",
        "chunking_strategy",
        "pinecone_namespace",
    }
    missing = sorted(required.difference(components))
    if missing:
        raise RuntimeError(
            "Retriever initialization returned incomplete components: "
            + ", ".join(missing)
        )
    if components["chunking_strategy"] != chunking_strategy:
        raise RuntimeError("Retriever strategy does not match the requested strategy")
    return components


class PredictionRequest(BaseModel):
    """Validated public prediction input."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(
        min_length=3,
        max_length=1000,
        description="Inheritance-law question to answer.",
        examples=["What share does a daughter receive in Islamic inheritance?"],
    )
    retrieval_mode: Literal["semantic", "hybrid"] = DEFAULT_RETRIEVAL_MODE
    top_k: int = Field(default=DEFAULT_TOP_K, ge=3, le=10)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value


class TimingResponse(BaseModel):
    retrieval_ms: int = Field(ge=0)
    generation_ms: int = Field(ge=0)
    total_ms: int = Field(ge=0)


class PredictionResponse(BaseModel):
    """Stable response that excludes prompts, credentials, and full documents."""

    answer: str
    provider: str
    model: str
    fallback_used: bool
    retrieved_chunks: int = Field(ge=1)
    sources: list[str]
    chunking_strategy: Literal["recursive"]
    retrieval_mode: Literal["semantic", "hybrid"]
    top_k: int
    timings: TimingResponse


class HealthResponse(BaseModel):
    status: Literal["ok"]
    retriever_loaded: bool
    chunking_strategy: Literal["recursive"]


def _safe_provider_error(value: Any) -> str:
    return " ".join(str(value or "Unknown generation error").split())[:300]


def get_retriever_components(request: Request) -> dict[str, Any]:
    components = getattr(request.app.state, "retriever_components", None)
    if components is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG retriever is not initialized",
        )
    return components


def create_app(
    *,
    retriever_loader: Callable[[], dict[str, Any]] = load_retriever_components,
) -> FastAPI:
    """Create an independently testable API with one startup model load."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.retriever_components = retriever_loader()
        yield
        application.state.retriever_components = None

    application = FastAPI(
        title="Legal-Aid RAG API",
        description=(
            "Retrieval-augmented inheritance-law answers using the existing "
            "MLflow-traced pipeline. Responses are general legal information."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    def health(
        components: dict[str, Any] = Depends(get_retriever_components),
    ) -> HealthResponse:
        return HealthResponse(
            status="ok",
            retriever_loaded=bool(components),
            chunking_strategy=DEFAULT_CHUNKING_STRATEGY,
        )

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["predictions"],
        summary="Answer an inheritance-law question",
    )
    def predict(
        payload: PredictionRequest,
        components: dict[str, Any] = Depends(get_retriever_components),
    ) -> PredictionResponse:
        profile = build_profile(
            retrieval_mode=payload.retrieval_mode,
            top_k=payload.top_k,
        )
        try:
            result = run_rag_query(
                payload.query,
                components,
                temperature=payload.temperature,
                profile=profile,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RAG prediction failed before a response was produced",
            ) from exc

        chunks = result.get("retrieval", {}).get("chunks", [])
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant legal context was retrieved",
            )

        generation = result.get("generation") or {}
        if generation.get("error") or not generation.get("answer"):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Answer generation failed: "
                    + _safe_provider_error(generation.get("error"))
                ),
            )

        sources = sorted(
            {
                str(chunk["source"])
                for chunk in chunks
                if chunk.get("source") is not None
            }
        )
        timings = result.get("timings", {})
        return PredictionResponse(
            answer=generation["answer"],
            provider=str(generation.get("provider") or "unknown"),
            model=str(generation.get("model_used") or "unknown"),
            fallback_used=bool(generation.get("fallback_used")),
            retrieved_chunks=len(chunks),
            sources=sources,
            chunking_strategy=DEFAULT_CHUNKING_STRATEGY,
            retrieval_mode=payload.retrieval_mode,
            top_k=payload.top_k,
            timings=TimingResponse(**timings),
        )

    return application


app = create_app()
