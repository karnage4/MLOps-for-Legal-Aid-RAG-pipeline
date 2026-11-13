"""Controlled prompt, model, and temperature inference comparisons in MLflow."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlflow
from dotenv import load_dotenv

from src.genai_comparison import (
    SCORER_MAX_ATTEMPTS,
    SCORER_TIMEOUT_SECONDS,
    rank_score_cards,
    score_card,
)
from src.genai_evaluation import (
    EvaluationProfile,
    build_retriever,
    get_managed_dataset,
    load_registered_judges,
    run_controlled_evaluation,
    validate_managed_dataset,
)
from src.generation import DEFAULT_PROMPT_VARIANT, PROMPT_VARIANTS
from src.genai_judges import CORE_JUDGE_NAMES
from src.core_genai_scorer import core_rag_quality
from src.mlflow_config import configure_mlflow


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_LINEAGE = "core_v2"
OUTPUT_DIR = PROJECT_ROOT / "output" / "inference_ablation" / EVALUATION_LINEAGE
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "inference_ablation" / EVALUATION_LINEAGE
COMPARISON_OUTPUT_PATH = OUTPUT_DIR / "comparison.json"
COMPARISON_ARTIFACT_PATH = ARTIFACT_DIR / "comparison.json"

BASELINE_PROFILE = EvaluationProfile("fixed", "hybrid", 5)
MODEL_A = os.getenv("INFERENCE_MODEL_A", "qwen/qwen3.8-27b")
MODEL_B = os.getenv("INFERENCE_MODEL_B", "openai/gpt-oss-120b")


@dataclass(frozen=True)
class InferenceConfig:
    """One isolated generation-variable comparison configuration."""

    label: str
    dimension: str
    prompt_variant: str
    model: str
    temperature: float
    provider: str = "groq"

    def __post_init__(self) -> None:
        if not self.label or any(character.isspace() for character in self.label):
            raise ValueError("label must be a non-empty slug")
        if self.dimension not in {"baseline", "prompt", "model", "temperature"}:
            raise ValueError("unsupported comparison dimension")
        if self.prompt_variant not in PROMPT_VARIANTS:
            raise ValueError("unknown prompt variant")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be between 0 and 1")
        if self.provider != "groq":
            raise ValueError("controlled inference comparisons require direct Groq")

    @property
    def generation_config(self) -> dict[str, Any]:
        return {
            "provider_mode": self.provider,
            "groq_model": self.model,
            "prompt_variant": self.prompt_variant,
        }


INFERENCE_CONFIGS = (
    InferenceConfig(
        "baseline-low-temp",
        "baseline",
        DEFAULT_PROMPT_VARIANT,
        MODEL_A,
        0.1,
    ),
    InferenceConfig(
        "prompt-structured",
        "prompt",
        "structured_legal_v2",
        MODEL_A,
        0.1,
    ),
    InferenceConfig(
        "model-b",
        "model",
        DEFAULT_PROMPT_VARIANT,
        MODEL_B,
        0.1,
    ),
    InferenceConfig(
        "temperature-high",
        "temperature",
        DEFAULT_PROMPT_VARIANT,
        MODEL_A,
        0.7,
    ),
)
CONFIG_BY_LABEL = {config.label: config for config in INFERENCE_CONFIGS}


def report_path(config: InferenceConfig, *, tracked: bool) -> Path:
    directory = ARTIFACT_DIR if tracked else OUTPUT_DIR
    return directory / f"{config.label}.json"


def _lineage(judges: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {"name": judge.name, "version": judge.scorer_version}
        for judge in judges
    ]


def is_reusable_report(
    report: Mapping[str, Any],
    *,
    config: InferenceConfig,
    dataset_report: Mapping[str, Any],
    judges: Sequence[Any],
) -> bool:
    stored = report.get("config", {})
    return (
        report.get("status") == "completed"
        and not report.get("scorer_failures")
        and stored.get("label") == config.label
        and stored.get("prompt_variant") == config.prompt_variant
        and stored.get("requested_generation_provider") == config.provider
        and stored.get("requested_generation_model") == config.model
        and float(stored.get("generation_temperature", -1)) == config.temperature
        and report.get("dataset", {}).get("dataset_digest")
        == dataset_report.get("dataset_digest")
        and report.get("dataset", {}).get("expectation_version")
        == dataset_report.get("expectation_version")
        and report.get("judges") == _lineage(judges)
        and len(report.get("rows", [])) == dataset_report.get("record_count")
    )


def load_reusable_report(
    config: InferenceConfig,
    *,
    dataset_report: Mapping[str, Any],
    judges: Sequence[Any],
) -> dict[str, Any] | None:
    path = report_path(config, tracked=True)
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return (
        report
        if is_reusable_report(
            report,
            config=config,
            dataset_report=dataset_report,
            judges=judges,
        )
        else None
    )


def _effect_cards(cards: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_label = {str(card["label"]): card for card in cards}
    baseline = by_label.get("baseline-low-temp")
    if baseline is None:
        return []
    dimensions = {
        "prompt-structured": "prompt",
        "model-b": "model",
        "temperature-high": "temperature",
    }
    metrics = (
        "fact_coverage",
        "retrieval_groundedness",
        "retrieval_relevance",
        "relevance_to_query",
        "avg_total_ms",
    )
    effects = []
    for label, dimension in dimensions.items():
        candidate = by_label.get(label)
        if candidate is None:
            continue
        effects.append(
            {
                "dimension": dimension,
                "baseline": "baseline-low-temp",
                "candidate": label,
                "deltas": {
                    metric: float(candidate[metric]) - float(baseline[metric])
                    for metric in metrics
                },
            }
        )
    return effects


def _comparison_report(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    dataset_report: Mapping[str, Any],
    judges: Sequence[Any],
    error: str | None = None,
) -> dict[str, Any]:
    cards = rank_score_cards([score_card(report) for report in reports.values()])
    complete = set(reports) == set(CONFIG_BY_LABEL)
    return {
        "status": "completed" if complete else ("interrupted" if error else "in_progress"),
        "objective": "Controlled prompt, model, and temperature inference ablation",
        "evaluation_lineage": EVALUATION_LINEAGE,
        "fixed_retrieval_profile": {
            "chunking_strategy": BASELINE_PROFILE.chunking_strategy,
            "retrieval_mode": BASELINE_PROFILE.retrieval_mode,
            "top_k": BASELINE_PROFILE.top_k,
            "pinecone_namespace": BASELINE_PROFILE.pinecone_namespace,
        },
        "dataset": dict(dataset_report),
        "judges": _lineage(judges),
        "requested_configs": list(CONFIG_BY_LABEL),
        "completed_configs": sorted(reports),
        "remaining_configs": sorted(set(CONFIG_BY_LABEL).difference(reports)),
        "best_config": cards[0] if complete and cards else None,
        "ranking": cards,
        "effects_vs_baseline": _effect_cards(cards),
        "error": error,
    }


def _write_comparison(report: Mapping[str, Any]) -> None:
    payload = json.dumps(report, indent=2)
    for path in (COMPARISON_OUTPUT_PATH, COMPARISON_ARTIFACT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def mark_best_config(cards: Sequence[Mapping[str, Any]]) -> None:
    if not cards:
        return
    client = mlflow.MlflowClient()
    for card in cards:
        client.set_tag(card["run_id"], "evaluation.best_inference_config", "false")
    best = cards[0]
    client.set_tag(best["run_id"], "evaluation.best_inference_config", "true")
    client.set_tag(
        best["run_id"],
        "evaluation.selection_reason",
        "Best controlled inference configuration by the documented quality-first rule.",
    )


def run_inference_ablation(
    requested_configs: Sequence[InferenceConfig] = INFERENCE_CONFIGS,
) -> dict[str, Any]:
    """Resume compatible reports and evaluate missing configurations sequentially."""
    experiment = configure_mlflow()
    dataset = get_managed_dataset()
    dataset_report = validate_managed_dataset(dataset)
    judges = load_registered_judges(
        str(experiment.experiment_id), CORE_JUDGE_NAMES
    )
    scorers = [core_rag_quality]

    reports: dict[str, dict[str, Any]] = {}
    for config in INFERENCE_CONFIGS:
        reusable = load_reusable_report(
            config,
            dataset_report=dataset_report,
            judges=judges,
        )
        if reusable is not None:
            reports[config.label] = reusable

    pending = [config for config in requested_configs if config.label not in reports]
    retriever = build_retriever(BASELINE_PROFILE.chunking_strategy) if pending else {}

    for config in pending:
        try:
            reports[config.label] = run_controlled_evaluation(
                profile=BASELINE_PROFILE,
                dataset=dataset,
                judges=judges,
                evaluation_scorers=scorers,
                feedback_names=CORE_JUDGE_NAMES,
                scorer_max_attempts=SCORER_MAX_ATTEMPTS,
                scorer_timeout_seconds=SCORER_TIMEOUT_SECONDS,
                retriever_components=retriever,
                temperature=config.temperature,
                generation_config=config.generation_config,
                evaluation_label=config.label,
                output_path=report_path(config, tracked=False),
                artifact_path=report_path(config, tracked=True),
            )
        except Exception as exc:
            checkpoint = _comparison_report(
                reports,
                dataset_report=dataset_report,
                judges=judges,
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_comparison(checkpoint)
            raise
        _write_comparison(
            _comparison_report(
                reports,
                dataset_report=dataset_report,
                judges=judges,
            )
        )

    comparison = _comparison_report(
        reports,
        dataset_report=dataset_report,
        judges=judges,
    )
    if comparison["status"] == "completed":
        mark_best_config(comparison["ranking"])
    _write_comparison(comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        choices=tuple(CONFIG_BY_LABEL),
        help="Evaluate only this configuration (repeatable); completed results resume.",
    )
    args = parser.parse_args()
    configs = (
        [CONFIG_BY_LABEL[label] for label in args.config]
        if args.config
        else INFERENCE_CONFIGS
    )
    print(json.dumps(run_inference_ablation(configs), indent=2))


if __name__ == "__main__":
    main()
