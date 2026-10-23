"""Tests for the compact MLflow root-chain output."""

from unittest.mock import MagicMock, Mock, patch

from src import pipeline


def test_root_span_keeps_compact_summary_without_changing_returned_result():
    retrieval = {
        "chunks": [{"source": "law.txt", "text": "Detailed retrieved text"}],
        "debug": {"mode": "hybrid"},
    }
    generation = {
        "answer": "Grounded answer",
        "provider": "groq",
        "model_used": "llama-3.1-8b-instant",
        "error": None,
    }
    span = Mock()
    span_context = MagicMock()
    span_context.__enter__.return_value = span

    with (
        patch.object(pipeline, "configure_mlflow"),
        patch.object(pipeline.mlflow, "start_span", return_value=span_context),
        patch.object(pipeline, "retrieve", return_value=retrieval),
        patch.object(pipeline, "generate_answer", return_value=generation),
    ):
        result = pipeline.run_rag_query(
            "Question",
            {},
            top_k=5,
            semantic_only=False,
            temperature=0.2,
        )

    assert result["retrieval"] == retrieval
    assert result["generation"] == generation
    root_output = span.set_outputs.call_args.args[0]
    assert root_output["answer"] == "Grounded answer"
    assert root_output["generation_provider"] == "groq"
    assert root_output["generation_model"] == "llama-3.1-8b-instant"
    assert root_output["retrieved_chunks"] == 1
    assert root_output["retrieval_mode"] == "hybrid"
    assert "retrieval" not in root_output
    assert "generation" not in root_output
