"""Resumable eight-profile MLflow GenAI evaluation comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlflow
from mlflow.genai.scorers import scorer

from src.genai_evaluation import (
    PROJECT_ROOT,
    EvaluationProfile,
    build_retrievers,
    evaluation_artifact_path,
    get_managed_dataset,
    load_registered_judges,
    run_controlled_evaluation,
    validate_managed_dataset,
)
from src.mlflow_config import configure_mlflow
from src.genai_judges import CORE_JUDGE_NAMES
from src.core_genai_scorer import core_rag_quality


COMPARISON_OUTPUT_PATH = PROJECT_ROOT / "output" / "genai_evaluation_comparison_core_v2.json"
COMPARISON_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "genai_evaluation_comparison_core_v2.json"

PROFILE_MATRIX = tuple(
    EvaluationProfile(chunking, retrieval, top_k)
    for chunking in ("recursive", "fixed")
    for retrieval in ("semantic", "hybrid")
    for top_k in (3, 5)
)
PROFILE_BY_LABEL = {profile.label: profile for profile in PROFILE_MATRIX}

METRIC_PRIORITY = (
    "fact_coverage",
    "retrieval_groundedness",
    "retrieval_relevance",
    "relevance_to_query",
)
SCORER_MAX_ATTEMPTS = 3
# Healthy judge calls complete well inside this bound. If the provider is
# throttling for longer, fail the profile and resume after quota recovery
# instead of holding the comparison process for hours.
SCORER_TIMEOUT_SECONDS = 60


def core_evaluation_output_path(profile: EvaluationProfile) -> Path:
    stem = profile.label.replace("-", "_")
    return PROJECT_ROOT / "output" / f"genai_evaluation_{stem}_core_v2.json"


def core_evaluation_artifact_path(profile: EvaluationProfile) -> Path:
    stem = profile.label.replace("-", "_")
    return PROJECT_ROOT / "artifacts" / f"genai_evaluation_{stem}_core_v2.json"


def build_retrying_scorers(
    judges: Sequence[Any],
    *,
    max_attempts: int = SCORER_MAX_ATTEMPTS,
    timeout_seconds: int = SCORER_TIMEOUT_SECONDS,
) -> list[Any]:
    """Wrap registered judges with bounded retries for transient model formatting errors."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be at least 1")

    def wrap(judge: Any) -> Any:
        @scorer(
            name=judge.name,
            description=judge.description,
            aggregations=judge.aggregations,
            pass_if=judge.pass_if,
            timeout=timeout_seconds,
        )
        def retrying_scorer(
            inputs=None,
            outputs=None,
            expectations=None,
            trace=None,
        ):
            for attempt in range(1, max_attempts + 1):
                try:
                    return judge.run(
                        inputs=inputs,
                        outputs=outputs,
                        expectations=expectations,
                        trace=trace,
                    )
                except Exception:
                    if attempt == max_attempts:
                        raise
                    print(
                        f"Retrying judge {judge.name!r} after transient failure "
                        f"({attempt}/{max_attempts})...",
                        flush=True,
                    )
                    time.sleep(attempt)

        return retrying_scorer

    return [wrap(judge) for judge in judges]


def _judge_versions(judges: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {"name": judge.name, "version": judge.scorer_version}
        for judge in judges
    ]


def is_reusable_report(
    report: Mapping[str, Any],
    *,
    profile: EvaluationProfile,
    dataset_report: Mapping[str, Any],
    judges: Sequence[Any],
) -> bool:
    """Only reuse a fully successful report with identical evaluation lineage."""
    return (
        report.get("status") == "completed"
        and not report.get("scorer_failures")
        and report.get("config", {}).get("label") == profile.label
        and report.get("dataset", {}).get("dataset_id") == dataset_report.get("dataset_id")
        and report.get("dataset", {}).get("dataset_digest") == dataset_report.get("dataset_digest")
        and report.get("dataset", {}).get("expectation_version")
        == dataset_report.get("expectation_version")
        and report.get("judges") == _judge_versions(judges)
        and len(report.get("rows", [])) == dataset_report.get("record_count")
    )


def load_reusable_report(
    profile: EvaluationProfile,
    *,
    dataset_report: Mapping[str, Any],
    judges: Sequence[Any],
) -> dict[str, Any] | None:
    path = core_evaluation_artifact_path(profile)
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not is_reusable_report(
        report,
        profile=profile,
        dataset_report=dataset_report,
        judges=judges,
    ):
        return None
    return report


def _row_rate(report: Mapping[str, Any], judge_name: str) -> float:
    values = [
        row.get(f"{judge_name}/value")
        for row in report.get("rows", [])
        if row.get(f"{judge_name}/value") is not None
    ]
    if not values:
        return 0.0
    positive = {"yes", "pass", "true", "1"}
    return sum(str(value).strip().lower() in positive for value in values) / len(values)


def score_card(report: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize every judge result plus latency for deterministic ranking."""
    metrics = report.get("metrics", {})

    def metric_or_rate(name: str) -> float:
        value = metrics.get(f"{name}/mean")
        return float(value) if value is not None else _row_rate(report, name)

    generation_summary = report.get("generation", {}).get("summary", {})
    return {
        "label": report["config"]["label"],
        "run_id": report["run_id"],
        "fact_coverage": metric_or_rate("fact_coverage"),
        "correctness": metric_or_rate("correctness"),
        "retrieval_sufficiency": metric_or_rate("retrieval_sufficiency"),
        "legal_answer_quality": _row_rate(report, "legal_answer_quality"),
        "retrieval_groundedness": metric_or_rate("retrieval_groundedness"),
        "retrieval_relevance": metric_or_rate("retrieval_relevance"),
        "retrieval_relevance_precision": float(
            metrics.get("retrieval_relevance/precision/mean", 0.0)
        ),
        "relevance_to_query": metric_or_rate("relevance_to_query"),
        "avg_total_ms": float(generation_summary.get("avg_total_ms", 0.0)),
        "fallback_rate": float(generation_summary.get("fallback_rate", 0.0)),
    }


def rank_score_cards(cards: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank quality first and use lower latency then label as tie-breakers."""
    return sorted(
        (dict(card) for card in cards),
        key=lambda card: (
            *(-float(card[metric]) for metric in METRIC_PRIORITY),
            -float(card["retrieval_relevance_precision"]),
            float(card["avg_total_ms"]),
            str(card["label"]),
        ),
    )


def _write_comparison(report: Mapping[str, Any]) -> None:
    payload = json.dumps(report, indent=2)
    for path in (COMPARISON_OUTPUT_PATH, COMPARISON_ARTIFACT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def _comparison_report(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    dataset_report: Mapping[str, Any],
    judges: Sequence[Any],
    requested_labels: Sequence[str],
    error: str | None = None,
) -> dict[str, Any]:
    cards = rank_score_cards([score_card(report) for report in reports.values()])
    all_complete = set(reports) == set(PROFILE_BY_LABEL)
    best = cards[0] if all_complete and cards else None
    return {
        "status": "completed" if all_complete else ("interrupted" if error else "in_progress"),
        "dataset": dict(dataset_report),
        "judges": _judge_versions(judges),
        "selection_rule": (
            "Highest per-fact coverage, retrieval groundedness, retrieval relevance, "
            "and relevance to query in that order; then lowest average total latency."
        ),
        "requested_profiles": list(requested_labels),
        "completed_profiles": sorted(reports),
        "remaining_profiles": sorted(set(PROFILE_BY_LABEL).difference(reports)),
        "best_profile": best,
        "ranking": cards,
        "error": error,
    }


def mark_best_profile(cards: Sequence[Mapping[str, Any]]) -> None:
    if not cards:
        return
    client = mlflow.MlflowClient()
    for card in cards:
        client.set_tag(card["run_id"], "evaluation.best_run", "false")
    best = cards[0]
    client.set_tag(best["run_id"], "evaluation.best_run", "true")
    client.set_tag(
        best["run_id"],
        "evaluation.selection_reason",
        "Best of eight profiles by the documented quality-first ranking rule.",
    )


def run_profile_comparison(
    requested_profiles: Sequence[EvaluationProfile] = PROFILE_MATRIX,
) -> dict[str, Any]:
    """Resume compatible results and evaluate missing profiles sequentially."""
    experiment = configure_mlflow()
    dataset = get_managed_dataset()
    dataset_report = validate_managed_dataset(dataset)
    judges = load_registered_judges(
        str(experiment.experiment_id), CORE_JUDGE_NAMES
    )
    evaluation_scorers = [core_rag_quality]

    reports: dict[str, dict[str, Any]] = {}
    for profile in PROFILE_MATRIX:
        reusable = load_reusable_report(
            profile,
            dataset_report=dataset_report,
            judges=judges,
        )
        if reusable is not None:
            reports[profile.label] = reusable

    pending = [profile for profile in requested_profiles if profile.label not in reports]
    retrievers = build_retrievers(("recursive", "fixed")) if pending else {}
    requested_labels = [profile.label for profile in requested_profiles]

    for profile in pending:
        try:
            reports[profile.label] = run_controlled_evaluation(
                profile=profile,
                dataset=dataset,
                judges=judges,
                evaluation_scorers=evaluation_scorers,
                feedback_names=CORE_JUDGE_NAMES,
                scorer_max_attempts=SCORER_MAX_ATTEMPTS,
                scorer_timeout_seconds=SCORER_TIMEOUT_SECONDS,
                retriever_components=retrievers[profile.chunking_strategy],
                output_path=core_evaluation_output_path(profile),
                artifact_path=core_evaluation_artifact_path(profile),
            )
        except Exception as exc:
            checkpoint = _comparison_report(
                reports,
                dataset_report=dataset_report,
                judges=judges,
                requested_labels=requested_labels,
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_comparison(checkpoint)
            raise
        checkpoint = _comparison_report(
            reports,
            dataset_report=dataset_report,
            judges=judges,
            requested_labels=requested_labels,
        )
        _write_comparison(checkpoint)

    comparison = _comparison_report(
        reports,
        dataset_report=dataset_report,
        judges=judges,
        requested_labels=requested_labels,
    )
    if comparison["status"] == "completed":
        mark_best_profile(comparison["ranking"])
    _write_comparison(comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        choices=tuple(PROFILE_BY_LABEL),
        help="Evaluate only this profile (repeatable); compatible results still resume.",
    )
    args = parser.parse_args()
    profiles = (
        [PROFILE_BY_LABEL[label] for label in args.profile]
        if args.profile
        else PROFILE_MATRIX
    )
    print(json.dumps(run_profile_comparison(profiles), indent=2))


if __name__ == "__main__":
    main()
