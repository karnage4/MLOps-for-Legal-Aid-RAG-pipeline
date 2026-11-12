"""Define and register versioned MLflow GenAI judges.

Step 2 is registration-only: importing or running this module does not execute
the RAG pipeline and does not call any judge model. The registered scorers are
consumed by the evaluation runner in Step 3.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Literal, Sequence

from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
    list_scorers,
)

from src.mlflow_config import configure_mlflow


DEFAULT_GROQ_JUDGE_MODEL = "openai/gpt-oss-20b"
JUDGE_NAMES = (
    "relevance_to_query",
    "retrieval_groundedness",
    "retrieval_relevance",
    "fact_coverage",
    "correctness",
    "retrieval_sufficiency",
    "legal_answer_quality",
)

# These four numeric/binary scorers form the reliable comparison gate. The
# remaining registered judges are retained as optional diagnostics: strict
# Correctness is intentionally all-or-nothing, RetrievalSufficiency overlaps
# the retrieval metrics, and legal_answer_quality is qualitative.
CORE_JUDGE_NAMES = (
    "relevance_to_query",
    "retrieval_groundedness",
    "retrieval_relevance",
    "fact_coverage",
)

LEGAL_ANSWER_QUALITY_INSTRUCTIONS = """
Evaluate the response using only {{ inputs }} and {{ outputs }}.
Return "pass" only when all of the following are true:
1. The answer directly addresses the user's legal-information question.
2. The answer distinguishes conditional rules from universal rules and avoids
   presenting an unsupported guarantee or personalized conclusion.
3. The answer communicates material uncertainty or fact dependence where needed.
4. The response is framed as general legal information and does not pretend to
   be personalized legal advice or guarantee a legal outcome.
Otherwise return "fail" and explain the most important problem in the rationale.
Do not use external legal knowledge, tools, or a reference answer. Groundedness
against retrieved documents is evaluated by a separate registered judge.
""".strip()

FACT_COVERAGE_INSTRUCTIONS = """
Compare {{ outputs }} with the `expected_facts` list in {{ expectations }}.
Evaluate every expected fact independently. Return a floating-point value equal
to `supported fact count / total expected fact count`, bounded from 0.0 to 1.0.

A fact is supported when the response explicitly states it or unambiguously
entails it. A partially stated, contradicted, or omitted fact is unsupported.
Do not make the whole score zero merely because one fact is missing.

In the rationale, enumerate every expected fact in order as SUPPORTED or
UNSUPPORTED, briefly explain the decision, and finish with the exact fraction
used for the score (for example, `3/4 = 0.75`). Do not use external knowledge.
""".strip()


def resolve_judge_model(model: str | None = None) -> str:
    """Return an MLflow direct-provider URI for the configured Groq model."""
    configured = (model or os.getenv("GROQ_JUDGE_MODEL") or DEFAULT_GROQ_JUDGE_MODEL)
    configured = configured.strip()
    if not configured:
        raise ValueError("GROQ_JUDGE_MODEL must not be empty")
    if ":/" in configured and not configured.startswith("groq:/"):
        raise ValueError("Step 2 judges must use a groq:/ model URI")
    return configured if configured.startswith("groq:/") else f"groq:/{configured}"


def build_judges(model: str | None = None) -> list[Any]:
    """Build the registered judges without invoking them."""
    model_uri = resolve_judge_model(model)
    return [
        RelevanceToQuery(model=model_uri),
        RetrievalGroundedness(model=model_uri),
        RetrievalRelevance(model=model_uri),
        make_judge(
            name="fact_coverage",
            instructions=FACT_COVERAGE_INSTRUCTIONS,
            feedback_value_type=float,
            model=model_uri,
        ),
        Correctness(model=model_uri),
        RetrievalSufficiency(model=model_uri),
        make_judge(
            name="legal_answer_quality",
            instructions=LEGAL_ANSWER_QUALITY_INSTRUCTIONS,
            feedback_value_type=Literal["pass", "fail"],
            model=model_uri,
        ),
    ]


def _judge_definition(judge: Any) -> dict[str, Any]:
    """Extract the stable serialized definition used for idempotence checks."""
    definition = judge.model_dump()
    # This records the serializer runtime, not a change to the judge definition.
    definition.pop("mlflow_version", None)
    return definition


def _judge_model(judge: Any) -> str | None:
    definition = judge.model_dump()
    builtin = definition.get("builtin_scorer_pydantic_data") or {}
    custom = definition.get("instructions_judge_pydantic_data") or {}
    return builtin.get("model") or custom.get("model")


def ensure_registered_judges(
    *,
    experiment: Any | None = None,
    judges: Sequence[Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Register changed judges and reuse identical latest versions."""
    experiment = experiment or configure_mlflow()
    experiment_id = str(experiment.experiment_id)
    desired = list(judges) if judges is not None else build_judges(model)

    names = [judge.name for judge in desired]
    if len(names) != len(set(names)):
        raise ValueError("Judge names must be unique")

    latest_by_name: dict[str, Any] = {}
    for scorer in list_scorers(experiment_id=experiment_id):
        current = latest_by_name.get(scorer.name)
        if current is None or (scorer.scorer_version or 0) > (
            current.scorer_version or 0
        ):
            latest_by_name[scorer.name] = scorer
    judge_reports = []
    for judge in desired:
        existing = latest_by_name.get(judge.name)
        if existing is not None and _judge_definition(existing) == _judge_definition(judge):
            registered = existing
            action = "reused"
        else:
            registered = judge.register(experiment_id=experiment_id)
            action = "registered" if existing is None else "versioned"

        judge_reports.append(
            {
                "name": registered.name,
                "version": registered.scorer_version,
                "model": _judge_model(registered),
                "action": action,
            }
        )

    return {
        "status": "ok",
        "experiment_id": experiment_id,
        "experiment_name": experiment.name,
        "judge_count": len(judge_reports),
        "inference_calls": 0,
        "judges": judge_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Define and register the versioned MLflow GenAI judges."
    )
    parser.add_argument(
        "--model",
        help="Groq model name or groq:/ URI (defaults to GROQ_JUDGE_MODEL).",
    )
    args = parser.parse_args()
    print(json.dumps(ensure_registered_judges(model=args.model), indent=2))


if __name__ == "__main__":
    main()
