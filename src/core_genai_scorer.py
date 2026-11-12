"""One-call Groq scorer for the four core RAG quality signals."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Mapping

import requests
from dotenv import load_dotenv
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer
from mlflow.genai.utils.trace_utils import extract_retrieval_context_from_trace

from src.genai_judges import CORE_JUDGE_NAMES, DEFAULT_GROQ_JUDGE_MODEL


CORE_SCORER_TIMEOUT_SECONDS = 120
MAX_CONTEXT_CHARACTERS = 5_000
MIN_JUDGE_INTERVAL_SECONDS = 30.0
_RATE_LOCK = threading.Lock()
_last_request_started = 0.0


def _query_text(inputs: Any) -> str:
    if isinstance(inputs, Mapping):
        return str(inputs.get("query") or inputs)
    return str(inputs or "")


def _expected_facts(expectations: Any) -> list[str]:
    if not isinstance(expectations, Mapping):
        return []
    facts = expectations.get("expected_facts") or []
    return [str(fact) for fact in facts]


def _retrieved_text(trace: Any) -> str:
    contexts = extract_retrieval_context_from_trace(trace)
    documents = []
    for chunks in contexts.values():
        for chunk in chunks:
            content = chunk.get("content") if isinstance(chunk, Mapping) else None
            if content:
                documents.append(str(content))
    return "\n\n".join(documents)[:MAX_CONTEXT_CHARACTERS]


def _prompt(*, query: str, answer: str, context: str, expected_facts: list[str]) -> str:
    return f"""Evaluate this RAG response using only the supplied query, retrieved context,
answer, and expected facts. Return JSON only, with no prose, markdown, or
explanations, using exactly this compact schema:
{{
  "relevance_to_query": 0.0,
  "retrieval_groundedness": 0.0,
  "retrieval_relevance": 0.0,
  "fact_coverage": {{"supported_count": 0, "total_count": {len(expected_facts)}}}
}}

The first three values must be numeric scores from 0.0 to 1.0. For
fact_coverage, independently check every expected fact and return the exact
supported count and total count. Do not make fact coverage all-or-nothing.

Query:
{query}

Retrieved context:
{context}

Answer:
{answer}

Expected facts:
{json.dumps(expected_facts, ensure_ascii=False)}
"""


def _validate_payload(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("judge response must be a JSON object")
    if not any(name in payload for name in CORE_JUDGE_NAMES):
        nested = next(
            (
                value
                for value in payload.values()
                if isinstance(value, Mapping)
                and any(name in value for name in CORE_JUDGE_NAMES)
            ),
            None,
        )
        if nested is not None:
            payload = nested
    validated: dict[str, dict[str, Any]] = {}
    for name in CORE_JUDGE_NAMES:
        item = payload.get(name)
        if isinstance(item, (int, float)):
            value = float(item)
            rationale = f"Judge returned a compact numeric score of {value:.3f}."
        elif isinstance(item, Mapping):
            raw_value = item.get("value")
            if raw_value is None and name == "fact_coverage":
                supported = item.get("supported_count")
                total = item.get("total_count")
                if supported is not None and total:
                    raw_value = float(supported) / float(total)
            if raw_value is None:
                raise ValueError(f"judge response is missing {name!r} value")
            value = float(raw_value)
            rationale = str(item.get("rationale") or "").strip()
            if not rationale:
                rationale = f"Judge returned a compact numeric score of {value:.3f}."
        else:
            raise ValueError(
                f"judge response is missing {name!r}; available keys: "
                f"{sorted(str(key) for key in payload.keys())}"
            )
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} value must be between 0 and 1")
        validated[name] = {"value": value, "rationale": rationale}
    return validated


def _decode_json_content(content: str) -> Any:
    """Decode a JSON object even when the model surrounds it with prose or fences."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("judge response did not contain a JSON object")
        return json.loads(content[start : end + 1])


def _wait_for_rate_window() -> None:
    """Keep large judge prompts inside Groq's per-minute token window."""
    global _last_request_started
    with _RATE_LOCK:
        elapsed = time.monotonic() - _last_request_started
        if elapsed < MIN_JUDGE_INTERVAL_SECONDS:
            time.sleep(MIN_JUDGE_INTERVAL_SECONDS - elapsed)
        _last_request_started = time.monotonic()


def _retry_delay(error: requests.RequestException, attempt: int) -> float:
    fallback = float(attempt * 5)
    response = error.response
    if response is None:
        return fallback
    for header in ("retry-after", "x-ratelimit-reset-tokens"):
        raw = response.headers.get(header)
        if raw:
            match = re.search(r"[0-9.]+", raw)
            if match:
                return min(60.0, max(fallback, float(match.group()) + 1.0))
    match = re.search(r"try again in\s+([0-9.]+)s", response.text or "", re.I)
    return min(60.0, max(fallback, float(match.group(1)) + 1.0)) if match else fallback


def call_core_judge(
    *,
    query: str,
    answer: str,
    context: str,
    expected_facts: list[str],
    max_attempts: int = 3,
) -> tuple[dict[str, dict[str, Any]], str, float, int]:
    """Call Groq once per attempt and return four validated assessments."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    model = os.getenv("GROQ_JUDGE_MODEL") or DEFAULT_GROQ_JUDGE_MODEL
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            _wait_for_rate_window()
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Return valid JSON only."},
                        {
                            "role": "user",
                            "content": _prompt(
                                query=query,
                                answer=answer,
                                context=context,
                                expected_facts=expected_facts,
                            ),
                        },
                    ],
                    "temperature": 0,
                    "max_completion_tokens": 1_200,
                },
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            payload = _validate_payload(_decode_json_content(content))
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            return payload, model, latency_ms, attempt
        except (requests.RequestException, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if isinstance(error, requests.RequestException) and error.response is not None:
                provider_message = (error.response.text or "")[:500]
                last_error = RuntimeError(f"{error}; provider_response={provider_message}")
            else:
                last_error = error
            if attempt < max_attempts:
                if isinstance(error, requests.RequestException):
                    time.sleep(_retry_delay(error, attempt))
                else:
                    time.sleep(float(attempt))
    raise RuntimeError(f"Core judge failed after {max_attempts} attempts: {last_error}")


@scorer(name="core_rag_quality", timeout=CORE_SCORER_TIMEOUT_SECONDS)
def core_rag_quality(inputs=None, outputs=None, expectations=None, trace=None):
    """Return four MLflow Feedback records from one schema-controlled call."""
    payload, model, latency_ms, attempts = call_core_judge(
        query=_query_text(inputs),
        answer=str(outputs or ""),
        context=_retrieved_text(trace),
        expected_facts=_expected_facts(expectations),
    )
    metadata = {
        "provider": "groq",
        "model": model,
        "latency_ms": str(latency_ms),
        "attempts": str(attempts),
    }
    return [
        Feedback(
            name=name,
            value=payload[name]["value"],
            rationale=payload[name]["rationale"],
            metadata=metadata,
        )
        for name in CORE_JUDGE_NAMES
    ]
