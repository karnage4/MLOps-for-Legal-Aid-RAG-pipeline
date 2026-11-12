"""Run one controlled MLflow GenAI evaluation profile."""

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


@dataclass(frozen=True)
class EvaluationProfile:
    """Lightweight evaluation profile compatible with the existing RAG boundary."""

    chunking_strategy: str = "fixed"
    retrieval_mode: str = "hybrid"
    top_k: int = 3

    def __post_init__(self) -> None:
        if self.chunking_strategy not in {"recursive", "fixed"}:
            raise ValueError("chunking_strategy must be 'recursive' or 'fixed'")
        if self.retrieval_mode not in {"semantic", "hybrid"}:
            raise ValueError("retrieval_mode must be 'semantic' or 'hybrid'")
        if self.top_k not in {3, 5}:
            raise ValueError("Step 4 evaluation top_k must be 3 or 5")

    @property
    def pinecone_namespace(self) -> str | None:
        return "fixed" if self.chunking_strategy == "fixed" else None

    @property
    def semantic_only(self) -> bool:
        return self.retrieval_mode == "semantic"

    @property
    def label(self) -> str:
        return f"{self.chunking_strategy}-{self.retrieval_mode}-top{self.top_k}"


PROFILE = EvaluationProfile()
GENERATION_TEMPERATURE = 0.2


def evaluation_output_path(profile: EvaluationProfile) -> Path:
    return PROJECT_ROOT / "output" / f"genai_evaluation_{profile.label.replace('-', '_')}.json"


def evaluation_artifact_path(profile: EvaluationProfile) -> Path:
    return PROJECT_ROOT / "artifacts" / f"genai_evaluation_{profile.label.replace('-', '_')}.json"


def run_rag_query(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy adapter that avoids model imports during setup and unit tests."""
    from src.pipeline import run_rag_query as implementation

    return implementation(*args, **kwargs)


def build_retriever(chunking_strategy: str) -> dict[str, Any]:
    """Lazy adapter that loads local retrieval models only for a live run."""
    from src.retrieval import build_retriever as implementation

    return implementation(chunking_strategy)


def build_retrievers(
    chunking_strategies: Sequence[str] = ("recursive", "fixed"),
) -> dict[str, dict[str, Any]]:
    """Lazy adapter that shares dense resources across evaluation profiles."""
    from src.retrieval import build_retrievers as implementation

    return implementation(chunking_strategies)


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
    generation_config: Mapping[str, Any] | None = None,
) -> Callable[[str], str]:
    """Adapt the existing traced RAG chain to MLflow's predict_fn contract."""

    def predict(query: str) -> str:
        run_options = {
            "temperature": temperature,
            "profile": profile,
        }
        if generation_config:
            run_options["generation_config"] = dict(generation_config)
        result = run_rag_query(query, dict(retriever_components), **run_options)
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
                "prompt_variant": generation.get("prompt_variant"),
                "requested_provider": generation.get("requested_provider"),
                "requested_model": generation.get("requested_model"),
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
        request = _decode_mapping(row.get("request"))
        query = (
            inputs.get("query")
            or request.get("query")
            or row.get("inputs/query")
            or row.get("inputs.query")
            or row.get("query")
        )
        if query:
            compact["query"] = query
        rows.append(compact)
    return rows


def _scorer_failures(
    rows: Sequence[Mapping[str, Any]],
    judge_names: Sequence[str] = JUDGE_NAMES,
) -> list[str]:
    failures = []
    for row_index, row in enumerate(rows, 1):
        for judge_name in judge_names:
            value = row.get(f"{judge_name}/value")
            error = row.get(f"{judge_name}/error_message")
            if error or value is None:
                failures.append(
                    f"row {row_index} {judge_name}: {error or 'missing value'}"
                )
    return failures


def _generation_summary(call_records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    stage_names = (
        "bm25_ms",
        "embedding_ms",
        "pinecone_ms",
        "semantic_total_ms",
        "fusion_ms",
        "rerank_ms",
        "retrieval_total_ms",
    )
    if not call_records:
        empty = {
            "fallback_rate": 0.0,
            "avg_retrieval_ms": 0.0,
            "avg_generation_ms": 0.0,
            "avg_total_ms": 0.0,
        }
        empty.update({f"avg_{name}": 0.0 for name in stage_names})
        return empty
    count = len(call_records)
    summary = {
        "fallback_rate": sum(bool(row["fallback_used"]) for row in call_records) / count,
        "avg_retrieval_ms": sum(float(row["retrieval_ms"]) for row in call_records) / count,
        "avg_generation_ms": sum(float(row["generation_ms"]) for row in call_records) / count,
        "avg_total_ms": sum(float(row["total_ms"]) for row in call_records) / count,
    }
    for stage_name in stage_names:
        summary[f"avg_{stage_name}"] = sum(
            float(row.get("retrieval_stages_ms", {}).get(stage_name, 0.0))
            for row in call_records
        ) / count
    return summary


def run_controlled_evaluation(
    *,
    profile: EvaluationProfile = PROFILE,
    dataset: Any | None = None,
    judges: Sequence[Any] | None = None,
    evaluation_scorers: Sequence[Any] | None = None,
    feedback_names: Sequence[str] | None = None,
    scorer_max_attempts: int = 1,
    scorer_timeout_seconds: int = 300,
    retriever_components: Mapping[str, Any] | None = None,
    evaluate_fn: Callable[..., Any] = mlflow_genai_evaluate,
    temperature: float = GENERATION_TEMPERATURE,
    generation_config: Mapping[str, Any] | None = None,
    evaluation_label: str | None = None,
    output_path: Path | None = None,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Execute and verify the labeled three-query, six-judge evaluation."""
    load_dotenv()
    experiment = configure_mlflow()
    experiment_id = str(experiment.experiment_id)
    dataset = dataset or get_managed_dataset()
    dataset_report = validate_managed_dataset(dataset)
    judges = list(judges) if judges is not None else load_registered_judges(experiment_id)
    scorers = list(evaluation_scorers) if evaluation_scorers is not None else judges
    expected_feedback_names = list(feedback_names or [judge.name for judge in judges])
    if feedback_names is None and [scorer.name for scorer in scorers] != [judge.name for judge in judges]:
        raise ValueError("Evaluation scorer names must match the registered judge lineage")
    if set(expected_feedback_names) != {judge.name for judge in judges}:
        raise ValueError("Expected feedback names must match the registered judge lineage")
    components = retriever_components or build_retriever(profile.chunking_strategy)
    config_label = evaluation_label or profile.label
    generation_options = dict(generation_config or {})
    prompt_variant = generation_options.get("prompt_variant", "strict_grounded_v1")
    requested_provider = generation_options.get("provider_mode", "groq")
    requested_model = generation_options.get("groq_model")

    # The direct Groq judge provider is quota-sensitive. Keep both prediction
    # and scoring deterministic and sequential for this controlled milestone run.
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
    os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "1"
    os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"
    call_records: list[dict[str, Any]] = []
    predict_fn = build_predict_fn(
        components,
        call_records=call_records,
        profile=profile,
        temperature=temperature,
        generation_config=generation_options,
    )

    run_tags = {
        "evaluation.type": "mlflow-genai-managed-dataset",
        "evaluation.config": config_label,
        "evaluation.dataset_id": dataset.dataset_id,
        "evaluation.expectation_version": EXPECTATION_VERSION,
        "evaluation.judge_count": str(len(judges)),
        "evaluation.prompt_variant": str(prompt_variant),
        "evaluation.requested_model": str(requested_model or "auto"),
        "evaluation.temperature": str(temperature),
    }
    with mlflow.start_run(
        run_name=f"genai-evaluation-{config_label}",
        tags=run_tags,
    ) as active_run:
        mlflow.log_params(
            {
                "config_label": config_label,
                "chunking_strategy": profile.chunking_strategy,
                "retrieval_mode": profile.retrieval_mode,
                "top_k": profile.top_k,
                "pinecone_namespace": profile.pinecone_namespace or "__default__",
                "generation_temperature": temperature,
                "prompt_variant": prompt_variant,
                "requested_generation_provider": requested_provider,
                "requested_generation_model": requested_model or "auto",
                "dataset_name": dataset.name,
                "dataset_digest": dataset.digest,
                "expectation_version": EXPECTATION_VERSION,
                "judge_versions": ",".join(
                    f"{judge.name}:v{judge.scorer_version}" for judge in judges
                ),
                "evaluation_workers": 1,
                "scorer_workers": 1,
                "scorer_max_attempts": scorer_max_attempts,
                "scorer_timeout_seconds": scorer_timeout_seconds,
                "trace_validation_skipped": True,
            }
        )
        result = evaluate_fn(
            data=dataset,
            predict_fn=predict_fn,
            scorers=scorers,
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
                **{
                    key: value
                    for key, value in generation_summary.items()
                    if key.startswith("avg_")
                    and key
                    not in {"avg_retrieval_ms", "avg_generation_ms", "avg_total_ms"}
                },
            }
        )
        mlflow.set_tags(
            {
                "generation.providers": ",".join(providers),
                "generation.models": ",".join(models),
                "generation.fallback_count": str(fallback_count),
                "generation.prompt_variant": str(prompt_variant),
                "generation.requested_provider": str(requested_provider),
                "generation.requested_model": str(requested_model or "auto"),
                "evaluation.status": "completed",
            }
        )

        rows = _compact_evaluation_rows(result.result_df)
        scorer_failures = _scorer_failures(rows, expected_feedback_names)
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
                "label": config_label,
                "chunking_strategy": profile.chunking_strategy,
                "retrieval_mode": profile.retrieval_mode,
                "top_k": profile.top_k,
                "pinecone_namespace": profile.pinecone_namespace or "__default__",
                "generation_temperature": temperature,
                "prompt_variant": prompt_variant,
                "requested_generation_provider": requested_provider,
                "requested_generation_model": requested_model,
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

    resolved_output_path = output_path or evaluation_output_path(profile)
    resolved_artifact_path = artifact_path or evaluation_artifact_path(profile)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    resolved_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_artifact_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["scorer_failures"]:
        raise RuntimeError(
            "MLflow GenAI evaluation was partial: "
            + "; ".join(report["scorer_failures"])
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one controlled MLflow GenAI evaluation profile."
    )
    parser.add_argument("--chunking", choices=("recursive", "fixed"), default="fixed")
    parser.add_argument("--retrieval", choices=("semantic", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", choices=(3, 5), type=int, default=3)
    args = parser.parse_args()
    profile = EvaluationProfile(args.chunking, args.retrieval, args.top_k)
    print(json.dumps(run_controlled_evaluation(profile=profile), indent=2))


if __name__ == "__main__":
    main()

