"""
Step 4 - LLM Generation via Groq
================================
Takes retrieved chunks from Step 3 and generates a grounded answer with Groq.

Also handles:
  - Prompt construction  (system + context + query)
  - Timing measurement   (for latency reporting in your report)
  - Bounded retries      (transient Groq failures only)

Usage (standalone test):
  pip install requests
  python generation.py

Import in your app:
  from generation import generate_answer
"""

import os
import time
from typing import List, Dict, Optional

import requests
import mlflow
from mlflow.entities import SpanType

from dotenv import load_dotenv

try:
    from src.mlflow_config import configure_mlflow
except ImportError:  # supports `python src/generation.py`
    from mlflow_config import configure_mlflow

load_dotenv()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

# Direct Groq generation model.
GROQ_MODEL          = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_CHAT_URL       = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MAX_ATTEMPTS   = max(1, int(os.getenv("GROQ_MAX_ATTEMPTS", "3")))

MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.2     # low = more factual, less creative (good for legal)
TOP_P           = 0.9
CONTEXT_CHUNKS  = 5       # how many retrieved chunks to include in prompt
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1.  PROMPT CONSTRUCTION
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable legal assistant specializing in inheritance law. \
Your role is to answer questions based strictly on the provided legal context. \

Rules you must follow:
- Answer ONLY from the provided context. Do not use outside knowledge.
- If the context does not contain enough information, say: \
  "The provided documents do not contain sufficient information to answer this question."
- Cite the source document name when referencing specific rules or sections.
- Be precise and use correct legal terminology.
- Do not give personal legal advice — remind users to consult a qualified lawyer for their specific situation."""


STRUCTURED_LEGAL_PROMPT = """You are an evidence-grounded inheritance-law research assistant.

Use only the supplied legal context. Produce a concise answer with these sections:
1. Direct answer
2. Applicable rule and conditions
3. Sources
4. Important limitation

Requirements:
- Cite the source document for every material legal rule.
- Clearly distinguish general rules from conditional or disputed rules.
- If the context is insufficient, explicitly say what cannot be established.
- Never invent statutes, cases, shares, jurisdictions, or facts.
- Give general legal information only and recommend a qualified lawyer for advice."""

DEFAULT_PROMPT_VARIANT = "strict_grounded_v1"
PROMPT_VARIANTS = {
    DEFAULT_PROMPT_VARIANT: SYSTEM_PROMPT,
    "structured_legal_v2": STRUCTURED_LEGAL_PROMPT,
}


def format_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a numbered context block.
    Each chunk is prefixed with its source document so the LLM
    can cite it and the judge can verify faithfulness.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source  = chunk.get("source", "Unknown source")
        text    = chunk.get("text", "").strip()
        context_parts.append(f"[Context {i} — Source: {source}]\n{text}")

    return "\n\n".join(context_parts)


def build_messages(
    query: str,
    chunks: List[Dict],
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> List[Dict]:
    """
    Build the full prompt in the OpenAI-compatible message format used by Groq.
    """
    if prompt_variant not in PROMPT_VARIANTS:
        supported = ", ".join(sorted(PROMPT_VARIANTS))
        raise ValueError(
            f"Unknown prompt variant {prompt_variant!r}; choose one of: {supported}"
        )
    context = format_context(chunks[:CONTEXT_CHUNKS])

    user_message = f"""Using the legal context provided below, answer the following question.

--- LEGAL CONTEXT ---
{context}
--- END CONTEXT ---

Question: {query}

Answer:"""

    return [
        {"role": "system", "content": PROMPT_VARIANTS[prompt_variant]},
        {"role": "user", "content": user_message}
    ]



# ─────────────────────────────────────────────
# 3.  MAIN GENERATION FUNCTION
# ─────────────────────────────────────────────

def _safe_provider_error(message: str) -> str:
    """Keep provider diagnostics useful without storing large response bodies."""
    return " ".join(str(message).split())[:500]


def _generate_answer_impl(
    query: str,
    chunks: List[Dict],
    groq_api_key: Optional[str] = None,
    groq_model: Optional[str] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    provider_mode: str = "groq",
) -> Dict:
    """Generate directly with Groq using bounded transient-error retries."""
    if provider_mode != "groq":
        raise ValueError("provider_mode must be 'groq'")
    generation_started = time.perf_counter()
    selected_groq_model = groq_model or GROQ_MODEL
    selected_groq_key = groq_api_key or GROQ_API_KEY
    prompt_messages = (
        build_messages(query, chunks, prompt_variant) if chunks else None
    )
    context = format_context(chunks[:CONTEXT_CHUNKS]) if chunks else None
    attempts = []

    if not chunks:
        return _generation_error(
            "No chunks retrieved — cannot generate an answer.",
            prompt_messages,
            context,
            attempts,
            generation_started,
            fallback_used=False,
            prompt_variant=prompt_variant,
            requested_provider=provider_mode,
            requested_model=selected_groq_model,
        )

    if not selected_groq_key:
        return _generation_error(
            "GROQ_API_KEY is not configured.",
            prompt_messages,
            context,
            attempts,
            generation_started,
            fallback_used=False,
            prompt_variant=prompt_variant,
            requested_provider=provider_mode,
            requested_model=selected_groq_model,
        )

    groq_started = time.perf_counter()
    groq_request_attempt = 0
    try:
        print(
            f"  Calling Groq model: {selected_groq_model} ...",
            end=" ",
            flush=True,
        )
        for groq_request_attempt in range(1, GROQ_MAX_ATTEMPTS + 1):
            try:
                response = requests.post(
                    GROQ_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {selected_groq_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": selected_groq_model,
                        "messages": prompt_messages,
                        "max_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": TOP_P,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                break
            except requests.RequestException as error:
                status_code = (
                    error.response.status_code if error.response is not None else None
                )
                retryable = status_code == 429 or (
                    status_code is not None and status_code >= 500
                ) or status_code is None
                if not retryable or groq_request_attempt >= GROQ_MAX_ATTEMPTS:
                    raise
                retry_delay = 2 ** (groq_request_attempt - 1)
                print(
                    f"retrying after {status_code or 'network error'} "
                    f"({groq_request_attempt}/{GROQ_MAX_ATTEMPTS})...",
                    end=" ",
                    flush=True,
                )
                time.sleep(retry_delay)

        payload = response.json()
        answer = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not answer:
            raise ValueError("Groq returned an empty response")

        groq_latency_ms = round((time.perf_counter() - groq_started) * 1000)
        attempts.append(
            {
                "provider": "groq",
                "model": selected_groq_model,
                "status": "success",
                "latency_ms": groq_latency_ms,
                "request_attempts": groq_request_attempt,
                "error": None,
            }
        )
        print(f"done ({groq_latency_ms} ms)")
        return {
            "answer": answer,
            "provider": "groq",
            "model_used": selected_groq_model,
            "prompt": str(prompt_messages),
            "context": context,
            "latency_ms": round(
                (time.perf_counter() - generation_started) * 1000
            ),
            "provider_latency_ms": groq_latency_ms,
            "fallback_used": False,
            "attempts": attempts,
            "prompt_variant": prompt_variant,
            "requested_provider": provider_mode,
            "requested_model": selected_groq_model,
            "error": None,
        }
    except Exception as error:
        groq_latency_ms = round((time.perf_counter() - groq_started) * 1000)
        safe_error = _safe_provider_error(error)
        attempts.append(
            {
                "provider": "groq",
                "model": selected_groq_model,
                "status": "failed",
                "latency_ms": groq_latency_ms,
                "request_attempts": groq_request_attempt,
                "error": safe_error,
            }
        )
        print(f"\n  Groq generation failed: {safe_error}")
        return _generation_error(
            f"Groq generation failed. Last error: {safe_error}",
            prompt_messages,
            context,
            attempts,
            generation_started,
            provider_latency_ms=groq_latency_ms,
            fallback_used=False,
            prompt_variant=prompt_variant,
            requested_provider=provider_mode,
            requested_model=selected_groq_model,
        )


def _generation_error(
    message: str,
    prompt_messages: Optional[List[Dict]],
    context: Optional[str],
    attempts: List[Dict],
    generation_started: float,
    provider_latency_ms: int = 0,
    fallback_used: bool = False,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    requested_provider: str = "groq",
    requested_model: Optional[str] = None,
) -> Dict:
    return {
        "answer": f"[Generation error] {message}",
        "provider": None,
        "model_used": None,
        "prompt": str(prompt_messages) if prompt_messages else None,
        "context": context,
        "latency_ms": round((time.perf_counter() - generation_started) * 1000),
        "provider_latency_ms": provider_latency_ms,
        "fallback_used": fallback_used,
        "attempts": attempts,
        "prompt_variant": prompt_variant,
        "requested_provider": requested_provider,
        "requested_model": requested_model,
        "error": message,
    }


def generate_answer(
    query: str,
    chunks: List[Dict],
    groq_api_key: Optional[str] = None,
    groq_model: Optional[str] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
    provider_mode: str = "groq",
) -> Dict:
    """Generate through Groq inside a sanitized MLflow LLM span.

    Provider tokens are deliberately never written to the trace.
    """
    configure_mlflow()
    with mlflow.start_span(name="legal_generation", span_type=SpanType.LLM) as span:
        span.set_inputs(
            {
                "query": query,
                "chunks": chunks,
                "provider_mode": provider_mode,
                "provider_priority": ["groq"],
                "requested_groq_model": groq_model or GROQ_MODEL,
                "prompt_variant": prompt_variant,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        )
        result = _generate_answer_impl(
            query,
            chunks,
            groq_api_key=groq_api_key,
            groq_model=groq_model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_variant=prompt_variant,
            provider_mode=provider_mode,
        )
        span.set_attribute("generation.provider", result.get("provider") or "none")
        span.set_attribute("generation.model", result.get("model_used") or "none")
        span.set_attribute("generation.latency_ms", result.get("latency_ms", 0))
        span.set_attribute(
            "generation.provider_latency_ms",
            result.get("provider_latency_ms", 0),
        )
        span.set_attribute("generation.fallback_used", result.get("fallback_used", False))
        span.set_attribute("generation.prompt_variant", prompt_variant)
        span.set_attribute("generation.requested_provider", provider_mode)
        span.set_attribute("generation.success", result.get("error") is None)
        span.set_attribute("generation.attempt_count", len(result.get("attempts", [])))
        span.set_attribute(
            "generation.groq_request_attempts",
            sum(
                attempt.get("request_attempts", 0)
                for attempt in result.get("attempts", [])
                if attempt.get("provider") == "groq"
            ),
        )
        span.set_outputs(result)
        return result


# ─────────────────────────────────────────────
# 4.  LATENCY BREAKDOWN HELPER
#     (for your report's performance section)
# ─────────────────────────────────────────────

def measure_full_pipeline(
    query: str,
    retriever_components: Dict,
) -> Dict:
    """
    Run the full pipeline and record timing at each stage.
    Import retrieve from step3_retrieval to use this.

    Returns timing dict suitable for your report's latency table.
    """
    from step3_retrieval import retrieve

    timings = {}

    # Retrieval timing
    t0 = time.time()
    retrieval_result = retrieve(query, **retriever_components)
    timings["retrieval_ms"] = round((time.time() - t0) * 1000)

    chunks = retrieval_result["chunks"]

    # Generation timing
    t0 = time.time()
    gen_result = generate_answer(query, chunks)
    timings["generation_ms"] = round((time.time() - t0) * 1000)

    timings["total_ms"] = timings["retrieval_ms"] + timings["generation_ms"]

    return {
        "query":        query,
        "answer":       gen_result["answer"],
        "model_used":   gen_result["model_used"],
        "chunks":       chunks,
        "context":      gen_result["context"],
        "timings":      timings,
        "error":        gen_result["error"],
    }


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Legal Inheritance RAG — Step 4: Generation Test")
    print("=" * 55)

    # Dummy chunks so you can test generation without running Step 3
    dummy_chunks = [
        {
            "source": "inheritance_act.pdf",
            "text": (
                "Under the Succession Act, when a person dies intestate (without a will), "
                "their estate is distributed according to the rules of intestate succession. "
                "The spouse receives one-third of the estate, and the remainder is divided "
                "equally among the children. If there are no children, the spouse inherits "
                "the entire estate."
            ),
            "rerank_score": 0.92,
        },
        {
            "source": "islamic_inheritance_rules.txt",
            "text": (
                "In Islamic inheritance law (Fara'id), a daughter receives half the share "
                "of a son. If there are no sons, two or more daughters together receive "
                "two-thirds of the estate. A single daughter alone receives one-half."
            ),
            "rerank_score": 0.85,
        },
    ]

    test_query = "What share does a daughter receive in Islamic inheritance law?"

    print(f"\nQuery: {test_query}")
    print("\nGenerating answer...")

    result = generate_answer(
        query  = test_query,
        chunks = dummy_chunks,
    )

    if result["error"]:
        print(f"\nError: {result['error']}")
    else:
        print(f"\nModel used  : {result['model_used']}")
        print(f"Latency     : {result['latency_ms']} ms")
        print(f"\nAnswer:\n{result['answer']}")

    print("\n" + "=" * 55)
    print("Done!")


if __name__ == "__main__":
    main()
