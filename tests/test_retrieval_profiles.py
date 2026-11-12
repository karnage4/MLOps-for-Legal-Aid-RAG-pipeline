"""Tests for strategy-isolated, reproducible retrieval profiles."""

from unittest.mock import MagicMock, Mock, call, patch

import pytest

from src import pipeline, retrieval


def _semantic_results(count: int = 5) -> list[dict]:
    return [
        {
            "id": f"doc-{index}",
            "text": f"legal text {index}",
            "source": "law.pdf",
            "chunk_index": index,
            "strategy": "fixed",
            "semantic_score": 1.0 - index / 10,
        }
        for index in range(count)
    ]


def test_default_profile_preserves_recursive_hybrid_top5():
    profile = retrieval.RetrievalProfile()

    assert profile.chunking_strategy == "recursive"
    assert profile.retrieval_mode == "hybrid"
    assert profile.top_k == 5
    assert profile.bm25_corpus_file == "./output/bm25_corpus.json"
    assert profile.pinecone_namespace is None
    assert profile.semantic_only is False
    assert profile.label == "recursive-hybrid-top5"


def test_fixed_semantic_profile_resolves_isolated_resources():
    profile = retrieval.RetrievalProfile(
        chunking_strategy="fixed",
        retrieval_mode="semantic",
        top_k=3,
    )

    assert profile.bm25_corpus_file == "./output/bm25_corpus_fixed.json"
    assert profile.pinecone_namespace == "fixed"
    assert profile.semantic_only is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunking_strategy": "unknown"}, "Unsupported chunking strategy"),
        ({"retrieval_mode": "keyword"}, "Unsupported retrieval mode"),
        ({"top_k": 0}, "top_k must be at least 1"),
    ],
)
def test_profile_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        retrieval.RetrievalProfile(**kwargs)


def test_semantic_search_queries_fixed_namespace_and_checks_strategy():
    embedder = Mock()
    embedder.encode.return_value.tolist.return_value = [0.1, 0.2]
    index = Mock()
    index.query.return_value = {
        "matches": [
            {
                "id": "fixed-1",
                "score": 0.91,
                "metadata": {
                    "text": "fixed legal text",
                    "source": "law.pdf",
                    "chunk_index": 1,
                    "strategy": "fixed",
                },
            }
        ]
    }

    results = retrieval.semantic_search(
        "inheritance question",
        embedder,
        index,
        namespace="fixed",
        expected_strategy="fixed",
    )

    index.query.assert_called_once_with(
        vector=[0.1, 0.2],
        top_k=retrieval.SEMANTIC_TOP_K,
        include_metadata=True,
        namespace="fixed",
    )
    assert results[0]["strategy"] == "fixed"


def test_semantic_mode_skips_sparse_search_and_cross_encoder():
    semantic_results = _semantic_results()

    with (
        patch.object(retrieval, "bm25_search") as bm25_search,
        patch.object(
            retrieval, "semantic_search", return_value=semantic_results
        ) as semantic_search,
        patch.object(retrieval, "rerank") as rerank,
    ):
        result = retrieval._retrieve_impl(
            "inheritance question",
            Mock(),
            [],
            [],
            Mock(),
            Mock(),
            Mock(),
            top_k=3,
            semantic_only=True,
            pinecone_namespace="fixed",
            chunking_strategy="fixed",
        )

    bm25_search.assert_not_called()
    rerank.assert_not_called()
    semantic_search.assert_called_once()
    assert result["chunks"] == semantic_results[:3]
    assert result["debug"]["bm25_count"] == 0
    assert result["debug"]["semantic_count"] == 5
    assert result["debug"]["fused_count"] == 5
    assert result["debug"]["rerank_candidate_count"] == 0
    assert result["debug"]["final_count"] == 3
    assert result["debug"]["mode"] == "semantic"
    assert result["debug"]["chunking_strategy"] == "fixed"
    assert result["debug"]["pinecone_namespace"] == "fixed"
    assert result["debug"]["top_k"] == 3
    assert result["debug"]["timings_ms"]["rerank_ms"] == 0.0


def test_default_hybrid_mode_preserves_fusion_and_reranking():
    bm25_results = [{"id": "doc-1", "text": "sparse result"}]
    semantic_results = _semantic_results(2)
    fused_results = [{"id": "doc-1", "text": "fused result"}]
    reranked_results = [
        {"id": "doc-1", "text": "fused result", "rerank_score": 0.9}
    ]

    with (
        patch.object(
            retrieval, "bm25_search", return_value=bm25_results
        ) as bm25_search,
        patch.object(
            retrieval, "semantic_search", return_value=semantic_results
        ) as semantic_search,
        patch.object(
            retrieval, "reciprocal_rank_fusion", return_value=fused_results
        ) as fuse,
        patch.object(
            retrieval, "rerank", return_value=reranked_results
        ) as rerank,
    ):
        result = retrieval._retrieve_impl(
            "inheritance question",
            Mock(),
            ["doc-1"],
            ["sparse result"],
            Mock(),
            Mock(),
            Mock(),
        )

    bm25_search.assert_called_once()
    semantic_search.assert_called_once()
    fuse.assert_called_once_with(bm25_results, semantic_results)
    rerank.assert_called_once()
    assert result["chunks"] == reranked_results
    assert result["debug"]["mode"] == "hybrid"
    assert result["debug"]["chunking_strategy"] == "recursive"
    assert result["debug"]["pinecone_namespace"] == "__default__"
    assert result["debug"]["rerank_candidate_count"] == 1
    assert set(result["debug"]["timings_ms"]) == {
        "bm25_ms",
        "embedding_ms",
        "pinecone_ms",
        "semantic_total_ms",
        "fusion_ms",
        "rerank_ms",
        "retrieval_total_ms",
    }


def test_hybrid_reranks_only_the_bounded_fused_pool():
    fused_results = [
        {"id": f"doc-{index}", "text": f"text {index}"}
        for index in range(retrieval.RERANK_CANDIDATE_K + 5)
    ]

    with (
        patch.object(retrieval, "bm25_search", return_value=fused_results[:2]),
        patch.object(retrieval, "semantic_search", return_value=_semantic_results()),
        patch.object(
            retrieval, "reciprocal_rank_fusion", return_value=fused_results
        ),
        patch.object(retrieval, "rerank", return_value=[]) as rerank,
    ):
        result = retrieval._retrieve_impl(
            "inheritance question",
            Mock(),
            [],
            [],
            Mock(),
            Mock(),
            Mock(),
        )

    rerank.assert_called_once()
    assert len(rerank.call_args.args[1]) == retrieval.RERANK_CANDIDATE_K
    assert result["debug"]["fused_count"] == len(fused_results)
    assert result["debug"]["rerank_candidate_count"] == retrieval.RERANK_CANDIDATE_K


def test_retrieve_records_stage_timings_on_mlflow_span():
    result = {
        "chunks": [{"id": "doc-1", "text": "context"}],
        "debug": {
            "timings_ms": {"bm25_ms": 1.5, "rerank_ms": 2.5},
            "rerank_candidate_count": 8,
        },
    }
    span = Mock()
    span_context = MagicMock()
    span_context.__enter__.return_value = span

    with (
        patch.object(retrieval, "configure_mlflow"),
        patch.object(retrieval.mlflow, "start_span", return_value=span_context),
        patch.object(retrieval, "_retrieve_impl", return_value=result),
    ):
        observed = retrieval.retrieve(
            "question",
            Mock(),
            [],
            [],
            Mock(),
            Mock(),
            Mock(),
        )

    assert observed is result
    span.set_attribute.assert_any_call("retrieval.bm25_ms", 1.5)
    span.set_attribute.assert_any_call("retrieval.rerank_ms", 2.5)
    span.set_attribute.assert_any_call("retrieval.rerank_candidate_count", 8)
    span.set_outputs.assert_called_once_with(result["chunks"])


def test_build_retriever_selects_fixed_bm25_and_namespace():
    bm25_components = (Mock(), ["fixed-1"], ["fixed text"])
    model_components = (Mock(), Mock())

    with (
        patch.object(
            retrieval, "build_bm25_index", return_value=bm25_components
        ) as build_bm25,
        patch.object(retrieval, "init_pinecone", return_value=Mock()),
        patch.object(
            retrieval, "init_models", return_value=model_components
        ),
    ):
        components = retrieval.build_retriever("fixed")

    build_bm25.assert_called_once_with("./output/bm25_corpus_fixed.json")
    assert components["chunking_strategy"] == "fixed"
    assert components["pinecone_namespace"] == "fixed"


def test_build_retrievers_shares_dense_models_and_keeps_bm25_isolated():
    recursive_bm25 = (Mock(), ["recursive-1"], ["recursive text"])
    fixed_bm25 = (Mock(), ["fixed-1"], ["fixed text"])
    shared_index = Mock()
    shared_embedder = Mock()
    shared_reranker = Mock()

    with (
        patch.object(
            retrieval,
            "build_bm25_index",
            side_effect=[recursive_bm25, fixed_bm25],
        ) as build_bm25,
        patch.object(
            retrieval, "init_pinecone", return_value=shared_index
        ) as init_pinecone,
        patch.object(
            retrieval,
            "init_models",
            return_value=(shared_embedder, shared_reranker),
        ) as init_models,
    ):
        retrievers = retrieval.build_retrievers(("recursive", "fixed"))

    assert build_bm25.call_args_list == [
        call("./output/bm25_corpus.json"),
        call("./output/bm25_corpus_fixed.json"),
    ]
    init_pinecone.assert_called_once()
    init_models.assert_called_once()
    assert retrievers["recursive"]["bm25"] is recursive_bm25[0]
    assert retrievers["fixed"]["bm25"] is fixed_bm25[0]
    assert retrievers["recursive"]["embedder"] is shared_embedder
    assert retrievers["fixed"]["embedder"] is shared_embedder
    assert retrievers["recursive"]["reranker"] is shared_reranker
    assert retrievers["fixed"]["reranker"] is shared_reranker
    assert retrievers["recursive"]["pinecone_index"] is shared_index
    assert retrievers["fixed"]["pinecone_index"] is shared_index
    assert retrievers["recursive"]["pinecone_namespace"] is None
    assert retrievers["fixed"]["pinecone_namespace"] == "fixed"


def test_pipeline_accepts_profile_and_routes_its_query_options():
    profile = retrieval.RetrievalProfile("fixed", "semantic", 3)
    components = {
        "chunking_strategy": "fixed",
        "pinecone_namespace": "fixed",
    }
    retrieval_result = {
        "chunks": [{"source": "law.pdf", "text": "fixed context"}],
        "debug": {"mode": "semantic", "chunking_strategy": "fixed"},
    }
    generation_result = {
        "answer": "answer",
        "provider": "groq",
        "model_used": "test-model",
        "error": None,
    }
    span = Mock()
    span_context = MagicMock()
    span_context.__enter__.return_value = span

    with (
        patch.object(pipeline, "configure_mlflow"),
        patch.object(pipeline.mlflow, "start_span", return_value=span_context),
        patch.object(
            pipeline, "retrieve", return_value=retrieval_result
        ) as retrieve,
        patch.object(
            pipeline, "generate_answer", return_value=generation_result
        ),
    ):
        pipeline.run_rag_query(
            "question",
            components,
            temperature=0.2,
            profile=profile,
        )

    retrieve.assert_called_once_with(
        "question",
        **components,
        top_k=3,
        semantic_only=True,
    )
    root_inputs = span.set_inputs.call_args.args[0]
    assert root_inputs["chunking_strategy"] == "fixed"
    assert root_inputs["retrieval_mode"] == "semantic"
    assert root_inputs["pinecone_namespace"] == "fixed"


def test_pipeline_rejects_profile_component_strategy_mismatch():
    profile = retrieval.RetrievalProfile("fixed", "hybrid", 5)

    with pytest.raises(ValueError, match="components were built for"):
        pipeline.run_rag_query(
            "question",
            {"chunking_strategy": "recursive"},
            temperature=0.2,
            profile=profile,
        )


def test_pipeline_rejects_profile_component_namespace_mismatch():
    profile = retrieval.RetrievalProfile("fixed", "hybrid", 5)

    with pytest.raises(ValueError, match="namespace does not match"):
        pipeline.run_rag_query(
            "question",
            {
                "chunking_strategy": "fixed",
                "pinecone_namespace": None,
            },
            temperature=0.2,
            profile=profile,
        )
