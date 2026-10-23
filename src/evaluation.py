"""
Step 5 - LLM-as-a-Judge Evaluation (Faithfulness + Relevancy)
==============================================================
Implements the two evaluation metrics required by the assignment:

  Faithfulness  : Claim extraction → verify each claim against context
                  Score = verified_claims / total_claims

  Relevancy     : Generate 3 questions from answer → cosine similarity
                  with original query
                  Score = mean similarity of 3 generated questions

Also runs comparable experiments across:
  - Retrieval modes      (semantic-only vs hybrid + re-ranking)
  - Retrieved context    (top-3 vs top-5)

Outputs:
  ./output/rag_evaluation_details.json       — ignored per-query details
  ./artifacts/rag_evaluation_summary.json    — reviewable run summary

Usage:
  pip install requests sentence-transformers numpy
  python -m src.evaluation
"""

import os
import json
import time
import re
import numpy as np
import requests
from typing import List, Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

try:
    from src.mlflow_config import log_evaluation_summary, mark_best_evaluation_run
    from src.pipeline import run_rag_query
except ImportError:  # supports `python src/evaluation.py`
    from mlflow_config import log_evaluation_summary, mark_best_evaluation_run
    from pipeline import run_rag_query

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HF_API_TOKEN        = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
if HF_API_TOKEN:
    os.environ["HF_TOKEN"] = HF_API_TOKEN

GROQ_API_KEY        = os.getenv("GROQ_API_KEY")

# Keep the judge independently configurable while defaulting to the same active
# Groq model verified by the generation fallback.
JUDGE_MODEL         = os.getenv(
    "GROQ_JUDGE_MODEL",
    "openai/gpt-oss-20b",
)
JUDGE_REASONING_EFFORT = os.getenv("GROQ_JUDGE_REASONING_EFFORT", "low")


SIMILARITY_MODEL    = "all-MiniLM-L6-v2"   # same model used in retrieval

OUTPUT_DIR          = "./output"
ARTIFACT_DIR        = "./artifacts"
EVALUATION_QUERY_LIMIT = max(1, int(os.getenv("EVALUATION_QUERY_LIMIT", "3")))
EVALUATION_COOLDOWN_SECONDS = max(
    0.0,
    float(os.getenv("EVALUATION_COOLDOWN_SECONDS", "5")),
)

# Test set — 15 queries covering key inheritance law topics
# Replace / extend with queries relevant to YOUR corpus
TEST_QUERIES = [
    "Who inherits property when a person dies without a will?",
    "What is the share of a daughter in Islamic inheritance?",
    "Can a non-Muslim inherit from a Muslim's estate?",
    "How is the estate divided when there are multiple heirs?",
    "What rights does a spouse have over inherited property?",
    "Can a person be disinherited under Pakistani inheritance law?",
    "What is the difference between a will and intestate succession?",
    "How are debts handled before distributing an inheritance?",
    "What happens to property if all direct heirs are deceased?",
    "Can a minor child be an heir?",
    "What is the role of an executor in inheritance proceedings?",
    "How is agricultural land treated differently in inheritance?",
    "What documents are required to claim inheritance?",
    "Can a gift given during lifetime affect inheritance share?",
    "What is the waiting period before inheritance can be claimed?",
]
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1.  LLM JUDGE HELPER
# ─────────────────────────────────────────────

def call_judge(prompt: str, max_tokens: int = 1200, max_retries: int = 4) -> str:
    """
    Call the LLM judge with a prompt. Returns the raw generated text.
    Uses the configured Groq judge model and retries transient failures.
    """
    if not GROQ_API_KEY:
        print("    Judge error: GROQ_API_KEY not set in .env")
        return ""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": JUDGE_REASONING_EFFORT,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if text:
                return text
            print("    Judge returned an empty response.")
            if attempt < max_retries:
                time.sleep(min(float(2 ** (attempt - 1)), 5.0))
        except requests.RequestException as error:
            status_code = (
                error.response.status_code if error.response is not None else None
            )
            retryable = status_code == 429 or (
                status_code is not None and status_code >= 500
            ) or status_code is None
            if not retryable or attempt >= max_retries:
                print(f"    Judge error on Groq: {error}")
                break

            wait_time = _provider_retry_delay(error, attempt)
            print(
                f"    Judge retry after {status_code or 'network error'} "
                f"({attempt}/{max_retries}) in {wait_time}s..."
            )
            time.sleep(wait_time)

    return ""


def _provider_retry_delay(error: requests.RequestException, attempt: int) -> float:
    """Honor Groq retry timing when available, with a bounded fallback delay."""
    fallback = float(2 ** (attempt - 1))
    if error.response is None:
        return fallback

    retry_after = error.response.headers.get("retry-after")
    if retry_after:
        try:
            return min(60.0, max(fallback, float(retry_after)))
        except ValueError:
            pass

    match = re.search(
        r"try again in\s+([0-9.]+)s",
        error.response.text or "",
        flags=re.IGNORECASE,
    )
    if match:
        return min(60.0, max(fallback, float(match.group(1))))
    return fallback


# ─────────────────────────────────────────────
# 2.  FAITHFULNESS EVALUATION
# ─────────────────────────────────────────────

CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the answer below.
A claim is a single, verifiable statement of fact.
Return ONLY a numbered list of claims, one per line.
Do not include opinions, hedges, or meta-statements.

Answer:
{answer}

Numbered list of claims:"""


CLAIM_VERIFICATION_PROMPT = """You are a strict legal fact-checker.
Determine whether each numbered claim is fully supported by the provided context.

Context:
{context}

Claims:
{claims}

Return exactly one numbered verdict per claim using this format:
1. SUPPORTED
2. NOT_SUPPORTED

Do not add explanations or omit any claim."""


def extract_claims(answer: str) -> List[str]:
    """Ask the LLM to extract atomic factual claims from the answer."""
    prompt   = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    response = call_judge(prompt, max_tokens=1200)
    
    claims = []
    for line in response.splitlines():
        line = line.strip()
        # Match numbered list items or bullet points (e.g. "1.", "- ", "* ")
        match = re.match(r"^(?:\d+[.)]|-|\*)\s+(.+)$", line)
        if match:
            claim = match.group(1).strip()
            if len(claim) > 10:   # skip trivially short extractions
                claims.append(claim)

    return claims


def verify_claims(claims: List[str], context: str) -> Optional[List[bool]]:
    """Verify all claims in one judge request while preserving the score formula."""
    numbered_claims = "\n".join(
        f"{index}. {claim}" for index, claim in enumerate(claims, 1)
    )
    prompt = CLAIM_VERIFICATION_PROMPT.format(
        context=context[:5000],
        claims=numbered_claims,
    )
    response = call_judge(prompt, max_tokens=max(1200, len(claims) * 30))

    verdicts = {}
    for line in response.splitlines():
        match = re.match(
            r"^(\d+)[.)]\s+(SUPPORTED|NOT_SUPPORTED)\s*$",
            line.strip(),
            flags=re.IGNORECASE,
        )
        if match:
            verdicts[int(match.group(1))] = match.group(2).upper() == "SUPPORTED"

    expected = set(range(1, len(claims) + 1))
    if set(verdicts) != expected:
        print(
            f"    Judge returned {len(verdicts)}/{len(claims)} claim verdicts; "
            "marking this query as an evaluation failure."
        )
        return None
    return [verdicts[index] for index in range(1, len(claims) + 1)]


def evaluate_faithfulness(
    answer: str,
    context: str,
    verbose: bool = False,
) -> Dict:
    """
    Full faithfulness pipeline:
      1. Extract claims from answer
      2. Verify each claim against context
      3. Score = verified / total

    Returns:
      score           : float 0.0 – 1.0
      claims          : list of extracted claims
      verifications   : list of (claim, supported) pairs
      total_claims    : int
      supported_claims: int
    """
    if not answer or not context:
        return _empty_faithfulness("answer_or_context_missing")

    print("    Extracting claims...", end=" ", flush=True)
    claims = extract_claims(answer)
    print(f"{len(claims)} claims found.")

    if not claims:
        return _empty_faithfulness("judge_claim_extraction_failed")

    print("    Verifying claims in one batch...", end=" ", flush=True)
    verdicts = verify_claims(claims, context)
    if verdicts is None:
        return _empty_faithfulness("judge_claim_verification_failed")
    print("done.")

    verifications = []
    supported = 0
    for claim, is_supported in zip(claims, verdicts):
        verifications.append({"claim": claim, "supported": is_supported})
        if is_supported:
            supported += 1
        if verbose:
            safe_claim = claim.encode("ascii", errors="backslashreplace").decode("ascii")
            print(f"      {'SUPPORTED' if is_supported else 'NOT_SUPPORTED'} {safe_claim}")

    score = supported / len(claims) if claims else 0.0

    return {
        "score":             round(score, 4),
        "total_claims":      len(claims),
        "supported_claims":  supported,
        "claims":            claims,
        "verifications":     verifications,
        "error":             None,
    }


def _empty_faithfulness(error: Optional[str] = None) -> Dict:
    return {
        "score": 0.0, "total_claims": 0, "supported_claims": 0,
        "claims": [], "verifications": [], "error": error,
    }


# ─────────────────────────────────────────────
# 3.  RELEVANCY EVALUATION
# ─────────────────────────────────────────────

QUESTION_GEN_PROMPT = """Given the answer below, generate exactly 3 questions that this answer could be responding to.
Each question should be a standalone question that the answer addresses.
Return ONLY a numbered list of 3 questions, one per line.

Answer:
{answer}

3 questions this answer addresses:"""


def generate_questions_from_answer(answer: str) -> List[str]:
    """Ask the LLM to generate 3 questions that the answer responds to."""
    prompt   = QUESTION_GEN_PROMPT.format(answer=answer)
    response = call_judge(prompt, max_tokens=1000)

    questions = []
    for line in response.splitlines():
        line  = line.strip()
        # Match numbered list or bullet points ending in a question mark
        match = re.match(r"^(?:\d+[.)]|-|\*)\s+(.+\?)\s*$", line)
        if match:
            q = match.group(1).strip()
            if len(q) > 10:
                questions.append(q)

    # Fallback: any line ending in "?" if numbered parse failed
    if not questions:
        questions = [
            line.strip() for line in response.splitlines()
            if line.strip().endswith("?") and len(line.strip()) > 10
        ]

    return questions[:3]   # ensure max 3


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def evaluate_relevancy(
    original_query: str,
    answer: str,
    similarity_model: SentenceTransformer,
) -> Dict:
    """
    Full relevancy pipeline:
      1. Generate 3 questions from the answer
      2. Compute cosine similarity of each vs. the original query
      3. Score = mean of 3 similarities

    Returns:
      score              : float 0.0 – 1.0
      generated_questions: list of 3 generated questions
      similarities       : per-question similarity scores
    """
    if not answer:
        return _empty_relevancy("answer_missing")

    print("    Generating questions from answer...", end=" ", flush=True)
    questions = generate_questions_from_answer(answer)
    print(f"{len(questions)} questions generated.")

    if len(questions) != 3:
        return _empty_relevancy("judge_question_generation_failed")

    # Encode original query and generated questions
    all_texts    = [original_query] + questions
    embeddings   = similarity_model.encode(all_texts, normalize_embeddings=True)
    query_vec    = embeddings[0]
    question_vecs = embeddings[1:]

    similarities = []
    for q_vec in question_vecs:
        sim = cosine_similarity(query_vec, q_vec)
        similarities.append(round(float(sim), 4))

    score = float(np.mean(similarities)) if similarities else 0.0

    return {
        "score":               round(score, 4),
        "generated_questions": questions,
        "similarities":        similarities,
        "error":               None,
    }


def _empty_relevancy(error: Optional[str] = None) -> Dict:
    return {
        "score": 0.0,
        "generated_questions": [],
        "similarities": [],
        "error": error,
    }


# ─────────────────────────────────────────────
# 4.  FULL EVALUATION RUNNER
# ─────────────────────────────────────────────

def run_evaluation(
    test_queries: List[str],
    retriever_components: Dict,
    similarity_model: SentenceTransformer,
    config_label: str,
    semantic_only: bool = False,
    top_k: int = 5,
    temperature: float = 0.2,
    verbose: bool = False,
) -> Dict:
    """
    Run the full evaluation pipeline on all test queries.

    Args:
        config_label     : stable experiment label
        semantic_only    : if True, uses semantic-only retrieval
        top_k            : number of reranked chunks returned
        temperature      : answer-generation temperature
        verbose          : print per-claim verification

    Returns the configuration, per-query results, summary, and MLflow run ID.
    """
    results = []
    mode = "semantic_only" if semantic_only else "hybrid_rerank"

    if not test_queries:
        raise ValueError("Evaluation query set must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    print(f"\nRunning evaluation {config_label} — mode: {mode}, top_k: {top_k}")
    print(f"Queries: {len(test_queries)}")
    print("-" * 55)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] {query[:70]}")

        t_start = time.time()

        # Retrieval and generation share the same parent CHAIN span so one
        # evaluation query appears as one nested MLflow GenAI trace.
        print("  Running traced RAG chain...", end=" ", flush=True)
        rag_result = run_rag_query(
            query,
            retriever_components,
            top_k=top_k,
            semantic_only=semantic_only,
            temperature=temperature,
        )
        retrieval = rag_result["retrieval"]
        gen = rag_result["generation"]
        rag_timings = rag_result["timings"]
        retrieval_ms = rag_timings["retrieval_ms"]
        generation_ms = rag_timings["generation_ms"]
        chunks  = retrieval["chunks"]
        context = "\n\n".join(c["text"] for c in chunks)
        print(
            f"{retrieval_ms} ms retrieval, {generation_ms} ms generation "
            f"— {len(chunks)} chunks"
        )

        generation_error = (
            gen.get("error") if gen else "No context was retrieved for generation"
        )
        if generation_error:
            print(f"  Generation failed: {generation_error}")
            results.append({
                "query": query, "error": generation_error,
                "faithfulness": _empty_faithfulness(),
                "relevancy":    _empty_relevancy(),
                "timings": rag_timings,
            })
            time.sleep(EVALUATION_COOLDOWN_SECONDS)
            continue

        answer = gen["answer"]

        # 3. Faithfulness
        print("  Evaluating faithfulness...")
        faith = evaluate_faithfulness(answer, context, verbose=verbose)
        print(f"  Faithfulness: {faith['score']:.2%} "
              f"({faith['supported_claims']}/{faith['total_claims']} claims supported)")

        # 4. Relevancy
        print("  Evaluating relevancy...")
        rel = evaluate_relevancy(query, answer, similarity_model)
        print(f"  Relevancy: {rel['score']:.4f} "
              f"(similarities: {rel['similarities']})")

        total_ms = round((time.time() - t_start) * 1000)
        evaluation_errors = [
            error
            for error in (faith.get("error"), rel.get("error"))
            if error
        ]
        evaluation_error = "; ".join(evaluation_errors) or None
        if evaluation_error:
            print(f"  Evaluation failed: {evaluation_error}")

        results.append({
            "query":           query,
            "answer":          answer,
            "model_used":      gen["model_used"],
            "context_chunks":  len(chunks),
            "faithfulness":    faith,
            "relevancy":       rel,
            "timings": {
                "retrieval_ms":  retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms":      total_ms,
            },
            "config": {
                "mode": mode,
                "chunking_strategy": "recursive",
                "top_k": top_k,
                "temperature": temperature,
            },
            "error": evaluation_error,
        })

        # Keep provider token windows independent between comparable queries.
        time.sleep(EVALUATION_COOLDOWN_SECONDS)

    summary = summarize_results(results, config_label)
    run_id = log_evaluation_summary(
        run_name=f"evaluation-{config_label}",
        params={
            "config_label": config_label,
            "chunking_strategy": "recursive",
            "retrieval_mode": mode,
            "semantic_only": semantic_only,
            "top_k": top_k,
            "generation_temperature": temperature,
            "query_count": len(test_queries),
            "judge_model": JUDGE_MODEL,
            "judge_reasoning_effort": JUDGE_REASONING_EFFORT,
            "similarity_model": SIMILARITY_MODEL,
        },
        summary=summary,
        tags={
            "evaluation.config": config_label,
            "evaluation.status": (
                "passed" if summary["failed_queries"] == 0 else "partial"
            ),
        },
    )
    print(f"  MLflow run: {run_id}")
    return {
        "config": {
            "label": config_label,
            "retrieval_mode": mode,
            "semantic_only": semantic_only,
            "top_k": top_k,
            "generation_temperature": temperature,
            "chunking_strategy": "recursive",
        },
        "results": results,
        "summary": summary,
        "run_id": run_id,
    }


# ─────────────────────────────────────────────
# 5.  RAG CONFIGURATION EXPERIMENTS
# ─────────────────────────────────────────────

EXPERIMENT_CONFIGS = [
    {
        "label": "A-semantic-top5",
        "semantic_only": True,
        "top_k": 5,
        "temperature": 0.2,
    },
    {
        "label": "B-hybrid-top3",
        "semantic_only": False,
        "top_k": 3,
        "temperature": 0.2,
    },
    {
        "label": "C-hybrid-top5",
        "semantic_only": False,
        "top_k": 5,
        "temperature": 0.2,
    },
]


def run_experiment_suite(
    test_queries: List[str],
    retriever_components: Dict,
    similarity_model: SentenceTransformer,
) -> Dict:
    """Run the same evaluation set across three genuine RAG configurations."""
    if not test_queries:
        raise ValueError("Evaluation query set must not be empty")

    experiments = {}
    for config in EXPERIMENT_CONFIGS:
        config_name = config["label"]
        print(f"\n{'='*55}")
        print(f"  RAG experiment {config_name}")
        print(f"{'='*55}")

        experiments[config_name] = run_evaluation(
            test_queries,
            retriever_components,
            similarity_model,
            config_label=config_name,
            semantic_only=config["semantic_only"],
            top_k=config["top_k"],
            temperature=config["temperature"],
            verbose=False,
        )

    best_label = select_best_configuration(experiments)
    best_experiment = experiments[best_label]
    selection_reason = (
        "Highest mean faithfulness, then mean relevancy, then lower average "
        "total latency."
    )
    mark_best_evaluation_run(best_experiment["run_id"], selection_reason)

    return {
        "query_count": len(test_queries),
        "judge_model": JUDGE_MODEL,
        "judge_reasoning_effort": JUDGE_REASONING_EFFORT,
        "similarity_model": SIMILARITY_MODEL,
        "experiments": experiments,
        "best_configuration": best_label,
        "best_run_id": best_experiment["run_id"],
        "selection_reason": selection_reason,
    }


def select_best_configuration(experiments: Dict) -> str:
    """Select the best successful configuration using the documented ordering."""
    candidates = {
        label: experiment
        for label, experiment in experiments.items()
        if experiment["summary"].get("successful_queries", 0) > 0
        and experiment["summary"].get("failed_queries", 0) == 0
    }
    if not candidates:
        raise ValueError("No evaluation configuration completed without failures")

    return max(
        candidates,
        key=lambda label: (
            candidates[label]["summary"]["faithfulness_mean"],
            candidates[label]["summary"]["relevancy_mean"],
            -candidates[label]["summary"]["avg_total_ms"],
        ),
    )


def summarize_results(results: List[Dict], label: str = "") -> Dict:
    """Compute mean faithfulness and relevancy from a list of query results."""
    valid = [r for r in results if not r.get("error")]
    failed_queries = len(results) - len(valid)
    if not valid:
        return {
            "label": label,
            "n_queries": len(results),
            "successful_queries": 0,
            "failed_queries": failed_queries,
            "success_rate": 0.0,
            "faithfulness_mean": 0.0,
            "faithfulness_std": 0.0,
            "relevancy_mean": 0.0,
            "relevancy_std": 0.0,
            "avg_retrieval_ms": 0.0,
            "avg_generation_ms": 0.0,
            "avg_total_ms": 0.0,
        }

    faith_scores = [r["faithfulness"]["score"] for r in valid]
    rel_scores   = [r["relevancy"]["score"]    for r in valid]
    ret_times    = [r["timings"]["retrieval_ms"]  for r in valid]
    gen_times    = [r["timings"]["generation_ms"] for r in valid]
    total_times  = [r["timings"]["total_ms"] for r in valid]

    return {
        "label":             label,
        "n_queries":         len(results),
        "successful_queries": len(valid),
        "failed_queries":    failed_queries,
        "success_rate":      round(len(valid) / len(results), 4),
        "faithfulness_mean": round(float(np.mean(faith_scores)), 4),
        "faithfulness_std":  round(float(np.std(faith_scores)),  4),
        "relevancy_mean":    round(float(np.mean(rel_scores)),   4),
        "relevancy_std":     round(float(np.std(rel_scores)),    4),
        "avg_retrieval_ms":  round(float(np.mean(ret_times)),    1),
        "avg_generation_ms": round(float(np.mean(gen_times)),    1),
        "avg_total_ms":      round(float(np.mean(total_times)),  1),
    }


# ─────────────────────────────────────────────
# 6.  REPORT HELPERS
# ─────────────────────────────────────────────

def print_experiment_table(suite: Dict):
    """Pretty-print the comparable RAG experiment summaries."""
    print("\n" + "=" * 75)
    print("  RAG EXPERIMENT RESULTS")
    print("=" * 75)
    header = (
        f"{'Config':<24} {'Faith':>7} {'Relev':>7} "
        f"{'Ret ms':>8} {'Gen ms':>8} {'OK/All':>7}"
    )
    print(header)
    print("-" * 75)
    for key, experiment in suite["experiments"].items():
        r = experiment["summary"]
        row = (
            f"{key:<24} "
            f"{r['faithfulness_mean']:>6.2%} "
            f"{r['relevancy_mean']:>7.4f} "
            f"{r['avg_retrieval_ms']:>8.0f} "
            f"{r['avg_generation_ms']:>8.0f} "
            f"{r['successful_queries']:>2}/{r['n_queries']:<4}"
        )
        print(row)
    print("=" * 75)
    print(f"Best configuration: {suite['best_configuration']}")
    print(f"Best MLflow run: {suite['best_run_id']}")


def print_example_faithfulness(results: List[Dict], n: int = 3):
    """
    Print claim extraction + verification for 14bn example queries.
    Required by the assignment: show at least 3 example queries.
    """
    print("\n" + "=" * 55)
    print("  FAITHFULNESS EXAMPLES (for your report)")
    print("=" * 55)

    shown = 0
    for r in results:
        if r.get("error") or shown >= n:
            continue
        faith = r["faithfulness"]
        if not faith["claims"]:
            continue

        print(f"\nQuery: {r['query']}")
        answer_snippet = r["answer"][:200].encode(
            "ascii", errors="backslashreplace"
        ).decode("ascii")
        print(f"Answer snippet: {answer_snippet}...")
        print(f"Faithfulness score: {faith['score']:.2%} "
              f"({faith['supported_claims']}/{faith['total_claims']})")
        print("Claims:")
        for v in faith["verifications"]:
            mark = "SUPPORTED" if v["supported"] else "NOT_SUPPORTED"
            safe_claim = v["claim"].encode(
                "ascii", errors="backslashreplace"
            ).decode("ascii")
            print(f"  {mark}: {safe_claim}")
        shown += 1


def save_results(suite: Dict):
    """Save detailed ignored output plus a compact reviewable summary artifact."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    detail_path = os.path.join(OUTPUT_DIR, "rag_evaluation_details.json")
    summary_path = os.path.join(ARTIFACT_DIR, "rag_evaluation_summary.json")

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed evaluation results saved: {detail_path}")

    compact_summary = {
        "query_count": suite["query_count"],
        "judge_model": suite["judge_model"],
        "judge_reasoning_effort": suite["judge_reasoning_effort"],
        "similarity_model": suite["similarity_model"],
        "best_configuration": suite["best_configuration"],
        "best_run_id": suite["best_run_id"],
        "selection_reason": suite["selection_reason"],
        "experiments": {
            label: {
                "config": experiment["config"],
                "summary": experiment["summary"],
                "run_id": experiment["run_id"],
            }
            for label, experiment in suite["experiments"].items()
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(compact_summary, f, ensure_ascii=False, indent=2)
    print(f"Reviewable summary saved: {summary_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Legal Inheritance RAG — Step 5: Evaluation")
    print("=" * 55)

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required for generation fallback and judging")

    # Load similarity model (used for relevancy scoring)
    print(f"\nLoading similarity model: {SIMILARITY_MODEL}")
    sim_model = SentenceTransformer(SIMILARITY_MODEL)

    try:
        from src.retrieval import build_retriever
    except ImportError:  # supports `python src/evaluation.py`
        from retrieval import build_retriever

    print("\nBuilding retriever from the recursive prepared corpus...")
    components = build_retriever()
    evaluation_queries = TEST_QUERIES[:EVALUATION_QUERY_LIMIT]
    suite = run_experiment_suite(evaluation_queries, components, sim_model)

    print_experiment_table(suite)
    best_results = suite["experiments"][suite["best_configuration"]]["results"]
    print_example_faithfulness(best_results, n=3)
    save_results(suite)

    print("\nDone!")
    print("=" * 55)


if __name__ == "__main__":
    main()
