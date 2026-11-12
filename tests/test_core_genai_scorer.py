"""Unit tests for the one-call core RAG scorer."""

from src import core_genai_scorer


def test_validate_payload_accepts_four_bounded_metrics():
    payload = {
        name: {"value": 0.5, "rationale": "Supported in part."}
        for name in core_genai_scorer.CORE_JUDGE_NAMES
    }

    assert core_genai_scorer._validate_payload(payload) == payload


def test_validate_payload_rejects_missing_metric():
    payload = {
        name: {"value": 1.0, "rationale": "Supported."}
        for name in core_genai_scorer.CORE_JUDGE_NAMES[:-1]
    }

    try:
        core_genai_scorer._validate_payload(payload)
    except ValueError as error:
        assert "fact_coverage" in str(error)
    else:
        raise AssertionError("missing metric should fail validation")


def test_validate_payload_normalizes_compact_model_response():
    payload = {
        "relevance_to_query": 0.9,
        "retrieval_groundedness": 0.8,
        "retrieval_relevance": 0.7,
        "fact_coverage": {"supported_count": 3, "total_count": 4},
    }

    result = core_genai_scorer._validate_payload(payload)

    assert result["fact_coverage"]["value"] == 0.75
    assert result["relevance_to_query"]["value"] == 0.9
    assert "compact numeric score" in result["retrieval_relevance"]["rationale"]


def test_prompt_requires_fractional_fact_coverage():
    prompt = core_genai_scorer._prompt(
        query="question",
        answer="answer",
        context="context",
        expected_facts=["one", "two"],
    )

    assert '"supported_count": 0, "total_count": 2' in prompt
    assert "Do not make fact coverage all-or-nothing" in prompt


def test_retry_delay_uses_token_reset_header():
    import requests

    response = requests.Response()
    response.status_code = 429
    response.headers["x-ratelimit-reset-tokens"] = "16.69s"
    error = requests.HTTPError(response=response)

    assert core_genai_scorer._retry_delay(error, 1) == 17.69


def test_decode_json_content_strips_markdown_fence():
    content = '```json\n{"relevance_to_query": 1.0}\n```'

    assert core_genai_scorer._decode_json_content(content) == {
        "relevance_to_query": 1.0
    }
