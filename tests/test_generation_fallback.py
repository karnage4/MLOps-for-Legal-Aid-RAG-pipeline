"""Unit tests for the Hugging Face -> Groq generation provider chain."""

from unittest.mock import MagicMock, Mock, patch

from src import generation


CHUNKS = [
    {
        "source": "test-law.txt",
        "text": "A single daughter receives one-half when there is no son.",
    }
]


def test_groq_is_used_after_huggingface_failure():
    groq_response = Mock()
    groq_response.raise_for_status.return_value = None
    groq_response.json.return_value = {
        "choices": [{"message": {"content": "The daughter receives one-half."}}]
    }

    with (
        patch.object(
            generation,
            "_generate_huggingface",
            return_value=generation._error_result("Hugging Face unavailable"),
        ),
        patch.object(generation.requests, "post", return_value=groq_response) as post,
    ):
        result = generation._generate_answer_impl(
            "What is a daughter's share?",
            CHUNKS,
            hf_token="test-hf-secret",
            groq_api_key="test-groq-secret",
        )

    assert result["error"] is None
    assert result["provider"] == "groq"
    assert result["model_used"] == generation.GROQ_FALLBACK_MODEL
    assert result["fallback_used"] is True
    assert result["attempts"][0]["provider"] == "huggingface"
    assert result["attempts"][0]["status"] == "failed"
    assert result["attempts"][1]["provider"] == "groq"
    assert result["attempts"][1]["status"] == "success"
    assert "test-hf-secret" not in repr(result)
    assert "test-groq-secret" not in repr(result)
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-groq-secret"


def test_huggingface_success_does_not_call_groq():
    hf_result = {
        "answer": "Hugging Face answer",
        "model_used": generation.PRIMARY_MODEL,
        "prompt": "prompt",
        "context": "context",
        "latency_ms": 10,
        "error": None,
    }

    with (
        patch.object(generation, "_generate_huggingface", return_value=hf_result),
        patch.object(generation.requests, "post") as post,
    ):
        result = generation._generate_answer_impl(
            "What is a daughter's share?",
            CHUNKS,
            hf_token="test-hf-secret",
            groq_api_key="test-groq-secret",
        )

    assert result["provider"] == "huggingface"
    assert result["model_used"] == generation.PRIMARY_MODEL
    assert result["fallback_used"] is False
    post.assert_not_called()


def test_mlflow_span_records_provider_model_latency_and_result():
    traced_result = {
        "answer": "Groq answer",
        "provider": "groq",
        "model_used": generation.GROQ_FALLBACK_MODEL,
        "prompt": "prompt",
        "context": "context",
        "latency_ms": 1500,
        "provider_latency_ms": 400,
        "fallback_used": True,
        "attempts": [
            {"provider": "huggingface", "status": "failed"},
            {"provider": "groq", "status": "success"},
        ],
        "error": None,
    }
    span = Mock()
    span_context = MagicMock()
    span_context.__enter__.return_value = span

    with (
        patch.object(generation, "configure_mlflow"),
        patch.object(generation.mlflow, "start_span", return_value=span_context),
        patch.object(generation, "_generate_answer_impl", return_value=traced_result),
    ):
        result = generation.generate_answer(
            "What is a daughter's share?",
            CHUNKS,
        )

    assert result == traced_result
    span.set_attribute.assert_any_call("generation.provider", "groq")
    span.set_attribute.assert_any_call(
        "generation.model", generation.GROQ_FALLBACK_MODEL
    )
    span.set_attribute.assert_any_call("generation.latency_ms", 1500)
    span.set_attribute.assert_any_call("generation.provider_latency_ms", 400)
    span.set_attribute.assert_any_call("generation.fallback_used", True)
    span.set_attribute.assert_any_call("generation.success", True)
    span.set_outputs.assert_called_once_with(traced_result)
