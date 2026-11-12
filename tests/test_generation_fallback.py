"""Unit tests for direct Groq generation and retry behavior."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src import generation


CHUNKS = [
    {
        "source": "test-law.txt",
        "text": "A single daughter receives one-half when there is no son.",
    }
]


def test_groq_is_the_only_generation_provider():
    groq_response = Mock()
    groq_response.raise_for_status.return_value = None
    groq_response.json.return_value = {
        "choices": [{"message": {"content": "The daughter receives one-half."}}]
    }

    with patch.object(
        generation.requests, "post", return_value=groq_response
    ) as post:
        result = generation._generate_answer_impl(
            "What is a daughter's share?",
            CHUNKS,
            groq_api_key="test-groq-secret",
        )

    assert result["error"] is None
    assert result["provider"] == "groq"
    assert result["model_used"] == generation.GROQ_MODEL
    assert result["fallback_used"] is False
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["provider"] == "groq"
    assert result["attempts"][0]["status"] == "success"
    assert "test-groq-secret" not in repr(result)
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-groq-secret"


def test_direct_groq_mode_uses_selected_model_and_prompt():
    groq_response = Mock()
    groq_response.raise_for_status.return_value = None
    groq_response.json.return_value = {
        "choices": [{"message": {"content": "Structured grounded answer"}}]
    }

    with patch.object(
        generation.requests, "post", return_value=groq_response
    ) as post:
        result = generation._generate_answer_impl(
            "What is a daughter's share?",
            CHUNKS,
            groq_api_key="test-groq-secret",
            groq_model="model-b",
            temperature=0.7,
            prompt_variant="structured_legal_v2",
            provider_mode="groq",
        )

    assert result["provider"] == "groq"
    assert result["model_used"] == "model-b"
    assert result["requested_model"] == "model-b"
    assert result["prompt_variant"] == "structured_legal_v2"
    assert result["fallback_used"] is False
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "model-b"
    assert payload["temperature"] == 0.7
    assert "Direct answer" in payload["messages"][0]["content"]


def test_prompt_variant_validation_fails_closed():
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        generation.build_messages("question", CHUNKS, "unknown")


def test_groq_retries_a_transient_provider_failure():
    unavailable = Mock(status_code=503)
    retry_error = generation.requests.HTTPError("temporarily unavailable")
    retry_error.response = unavailable
    failed_response = Mock()
    failed_response.raise_for_status.side_effect = retry_error

    successful_response = Mock()
    successful_response.raise_for_status.return_value = None
    successful_response.json.return_value = {
        "choices": [{"message": {"content": "Recovered answer"}}]
    }

    with (
        patch.object(
            generation.requests,
            "post",
            side_effect=[failed_response, successful_response],
        ) as post,
        patch.object(generation.time, "sleep") as sleep,
    ):
        result = generation._generate_answer_impl(
            "What is a daughter's share?",
            CHUNKS,
            groq_api_key="test-groq-secret",
        )

    assert result["provider"] == "groq"
    assert result["error"] is None
    assert result["attempts"][-1]["request_attempts"] == 2
    assert post.call_count == 2
    sleep.assert_called_once_with(1)


def test_mlflow_span_records_provider_model_latency_and_result():
    traced_result = {
        "answer": "Groq answer",
        "provider": "groq",
        "model_used": generation.GROQ_MODEL,
        "prompt": "prompt",
        "context": "context",
        "latency_ms": 1500,
        "provider_latency_ms": 400,
        "fallback_used": False,
        "attempts": [
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
        "generation.model", generation.GROQ_MODEL
    )
    span.set_attribute.assert_any_call("generation.latency_ms", 1500)
    span.set_attribute.assert_any_call("generation.provider_latency_ms", 400)
    span.set_attribute.assert_any_call("generation.fallback_used", False)
    span.set_attribute.assert_any_call("generation.success", True)
    span.set_outputs.assert_called_once_with(traced_result)
