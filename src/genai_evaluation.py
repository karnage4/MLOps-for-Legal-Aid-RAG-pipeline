"""Run the controlled MLflow GenAI evaluation for fixed-hybrid-top3."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import mlflow
from dotenv import load_dotenv
from mlflow.genai import evaluate as mlflow_genai_evaluate
from mlflow.genai.datasets import search_datasets
from mlflow.genai.scorers import get_scorer

from src.evaluation_expectations import EXPECTED_FACTS_BY_QUERY, EXPECTATION_VERSION
from src.genai_dataset import DATASET_NAME
from src.genai_judges import JUDGE_NAMES
from src.mlflow_config import configure_mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "output" / "genai_evaluation_fixed_hybrid_top3.json"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "genai_evaluation_fixed_hybrid_top3.json"

@dataclass(frozen=True)
class EvaluationProfile:
    """Lightweight fixed profile compatible with the existing RAG boundary."""

    chunking_strategy: str = "fixed"
    retrieval_mode: str = "hybrid"
    top_k: int = 3
    pinecone_namespace: str = "fixed"

    @property
    def semantic_only(self) -> bool:
        return self.retrieval_mode == "semantic"


PROFILE = EvaluationProfile()
CONFIG_LABEL = "fixed-hybrid-top3"
GENERATION_TEMPERATURE = 0.2


def run_rag_query(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy adapter that avoids model imports during setup and unit tests."""
    from src.pipeline import run_rag_query as implementation

    return implementation(*args, **kwargs)


def build_retriever(chunking_strategy: str) -> dict[str, Any]:
    """Lazy adapter that loads local retrieval models only for a live run."""
    from src.retrieval import build_retriever as implementation

    return implementation(chunking_strategy)


def _decode_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def validate_managed_dataset(dataset: Any) -> dict[str, Any]:
    """Require the exact approved inputs and version-1 expected facts."""
    frame = dataset.to_df()
    persisted: dict[str, list[str]] = {}
    for row in frame.to_dict(orient="records"):
        inputs = _decode_mapping(row.get("inputs"))
        expectations = _decode_mapping(row.get("expectations"))
        query = inputs.get("query")
        facts = expectations.get("expected_facts")
        if isinstance(query, str) and isinstance(facts, list):
            persisted[query.strip()] = facts

    expected = {query: list(facts) for query, facts in EXPECTED_FACTS_BY_QUERY.items()}
    if persisted != expected:
        missing = sorted(set(expected).difference(persisted))
        unexpected = sorted(set(persisted).difference(expected))
        raise ValueError(
            "Managed dataset does not match the approved expectations "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_name": dataset.name,
        "dataset_digest": dataset.digest,
        "record_count": len(persisted),
        "expectation_version": EXPECTATION_VERSION,
    }


def get_managed_dataset(dataset_name: str = DATASET_NAME) -> Any:
    matches = search_datasets(filter_string=f'name = "{dataset_name}"', max_results=2)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one dataset named {dataset_name!r}; found {len(matches)}"
        )
    validate_managed_dataset(matches[0])
    return matches[0]


def load_registered_judges(
    experiment_id: str,
    judge_names: Sequence[str] = JUDGE_NAMES,
) -> list[Any]:
    """Load the latest registered versions used as the evaluation lineage."""
    judges = [
        get_scorer(name=name, experiment_id=experiment_id)
        for name in judge_names
    ]
    missing_versions = [judge.name for judge in judges if judge.scorer_version is None]
    if missing_versions:
        raise RuntimeError(
            "Judges are not registered: " + ", ".join(missing_versions)
        )
    return judges


def build_predict_fn(
    retriever_components: Mapping[str, Any],
    *,
    call_records: list[dict[str, Any]],
    profile: EvaluationProfile = PROFILE,
    temperature: float = GENERATION_TEMPERATURE,
) -> Callable[[str], str]:
    """Adapt the existing traced RAG chain to MLflow's predict_fn contract."""

    def predict(query: str) -> str:
        result = run_rag_query(
            query,
            dict(retriever_components),
            temperature=temperature,
            profile=profile,
        )
        generation = result.get("generation")
        error = generation.get("error") if generation else "No generation result"
        answer = generation.get("answer") if generation else None
        if error or not answer:
            raise RuntimeError(f"RAG generation failed for {query!r}: {error}")

        call_records.append(
            {
                "query": query,
                "provider": generation.get("provider"),
                "model": generation.get("model_used"),
                "fallback_used": bool(generation.get("fallback_used")),
                "retrieved_chunks": len(result["retrieval"]["chunks"]),
                **result["timings"],
            }
        )
        return answer

    return predict


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _compact_evaluation_rows(result_df: Any) -> list[dict[str, Any]]:
    if result_df is None:
        return []
    rows = []
    for row in result_df.to_dict(orient="records"):
        compact = {
            key: _json_value(value)
            for key, value in row.items()
            if key in {"trace_id", "state", "execution_duration"}
            or key.endswith(("/value", "/rationale", "/error_message"))
        }
        inputs = _decode_mapping(row.get("inputs"))
        if query := inputs.get("query"):
            compact["query"] = query
        rows.append(compact)
    return rows


def _scorer_failures(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    failures = []
    for row_index, row in enumerate(rows, 1):
        for judge_name in JUDGE_NAMES:
            value = row.get(f"{judge_name}/value")
            error = row.get(f"{judge_name}/error_message")
            if error or value is None:
                failures.append(
                    f"row {row_index} {judge_name}: {error or 'missing value'}"
                )
    return failures


def _generation_summary(call_records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not call_records:
        return {
            "fallback_rate": 0.0,
            "avg_retrieval_ms": 0.0,
            "avg_generation_ms": 0.0,
            "avg_total_ms": 0.0,
        }
    count = len(call_records)
    return {
        "fallback_rate": sum(bool(row["fallback_used"]) for row in call_records) / count,
        "avg_retrieval_ms": sum(float(row["retrieval_ms"]) for row in call_records) / count,
        "avg_generation_ms": sum(float(row["generation_ms"]) for row in call_records) / count,
        "avg_total_ms": sum(float(row["total_ms"]) for row in call_records) / count,
    }


def run_controlled_evaluation(
    *,
    dataset: Any | None = None,
    retriever_components: Mapping[str, Any] | None = None,
    evaluate_fn: Callable[..., Any] = mlflow_genai_evaluate,
) -> dict[str, Any]:
    """Execute and verify the labeled three-query, six-judge evaluation."""
    load_dotenv()
    experiment = configure_mlflow()
    experiment_id = str(experiment.experiment_id)
    dataset = dataset or get_managed_dataset()
    dataset_report = validate_managed_dataset(dataset)
    judges = load_registered_judges(experiment_id)
    components = retriever_components or build_retriever("fixed")

    # The direct Groq judge provider is quota-sensitive. Keep both prediction
    # and scoring deterministic and sequential for this controlled milestone run.
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
    os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"
    call_records: list[dict[str, Any]] = []
    predict_fn = build_predict_fn(components, call_records=call_records)

    run_tags = {
        "evaluation.type": "mlflow-genai-managed-dataset",
        "evaluation.config": CONFIG_LABEL,
        "evaluation.dataset_id": dataset.dataset_id,
        "evaluation.expectation_version": EXPECTATION_VERSION,
        "evaluation.judge_count": str(len(judges)),
    }
    with mlflow.start_run(
        run_name=f"genai-evaluation-{CONFIG_LABEL}",
        tags=run_tags,
    ) as active_run:
        mlflow.log_params(
            {
                "config_label": CONFIG_LABEL,
                "chunking_strategy": PROFILE.chunking_strategy,
                "retrieval_mode": PROFILE.retrieval_mode,
                "top_k": PROFILE.top_k,
                "pinecone_namespace": PROFILE.pinecone_namespace,
                "generation_temperature": GENERATION_TEMPERATURE,
                "dataset_name": dataset.name,
                "dataset_digest": dataset.digest,
                "expectation_version": EXPECTATION_VERSION,
                "judge_versions": ",".join(
                    f"{judge.name}:v{judge.scorer_version}" for judge in judges
                ),
                "evaluation_workers": 1,
                "trace_validation_skipped": True,
            }
        )
        result = evaluate_fn(
            data=dataset,
            predict_fn=predict_fn,
            scorers=judges,
        )
        if result.run_id != active_run.info.run_id:
            raise RuntimeError("MLflow GenAI evaluation did not reuse the named run")

        providers = sorted(
            {record["provider"] for record in call_records if record.get("provider")}
        )
        models = sorted(
            {record["model"] for record in call_records if record.get("model")}
        )
        fallback_count = sum(record["fallback_used"] for record in call_records)
        generation_summary = _generation_summary(call_records)
        mlflow.log_metrics(
            {
                "generation_fallback_rate": generation_summary["fallback_rate"],
                "avg_retrieval_ms": generation_summary["avg_retrieval_ms"],
                "avg_generation_ms": generation_summary["avg_generation_ms"],
                "avg_total_ms": generation_summary["avg_total_ms"],
            }
        )
        mlflow.set_tags(
            {
                "generation.providers": ",".join(providers),
                "generation.models": ",".join(models),
                "generation.fallback_count": str(fallback_count),
                "evaluation.status": "completed",
            }
        )

        rows = _compact_evaluation_rows(result.result_df)
        scorer_failures = _scorer_failures(rows)
        if len(call_records) != dataset_report["record_count"]:
            scorer_failures.append(
                "prediction count mismatch: "
                f"expected {dataset_report['record_count']}, got {len(call_records)}"
            )

        report = {
            "status": "completed" if not scorer_failures else "partial",
            "run_id": result.run_id,
            "experiment_id": experiment_id,
            "config": {
                "label": CONFIG_LABEL,
                "chunking_strategy": PROFILE.chunking_strategy,
                "retrieval_mode": PROFILE.retrieval_mode,
                "top_k": PROFILE.top_k,
                "pinecone_namespace": PROFILE.pinecone_namespace,
                "generation_temperature": GENERATION_TEMPERATURE,
            },
            "dataset": dataset_report,
            "judges": [
                {"name": judge.name, "version": judge.scorer_version}
                for judge in judges
            ],
            "metrics": {
                key: _json_value(value) for key, value in result.metrics.items()
            },
            "generation": {
                "providers": providers,
                "models": models,
                "fallback_count": fallback_count,
                "summary": generation_summary,
                "calls": call_records,
            },
            "rows": rows,
            "scorer_failures": scorer_failures,
        }
        mlflow.set_tag("evaluation.status", report["status"])
        mlflow.log_dict(report, "genai_evaluation_summary.json")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["scorer_failures"]:
        raise RuntimeError(
            "MLflow GenAI evaluation was partial: "
            + "; ".join(report["scorer_failures"])
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLflow GenAI evaluation for fixed-hybrid-top3."
    )
    parser.parse_args()
    print(json.dumps(run_controlled_evaluation(), indent=2))


if __name__ == "__main__":
    main()

