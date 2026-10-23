"""Central MLflow setup for the original Legal-Aid RAG application.

No new environment variable is required. By default MLflow uses a SQLite
database in this project folder. Existing MLFLOW_* variables are respected
when the team later points the app at a shared tracking server.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = "legal-aid-rag"


def default_tracking_uri() -> str:
    database_path = (PROJECT_ROOT / "mlflow.db").resolve().as_posix()
    return f"sqlite:///{database_path}"


def configure_mlflow():
    """Configure the tracking backend and return the active experiment."""
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(default_tracking_uri())
        # SQLite trace writes are kept synchronous so a completed root span
        # cannot be read before its children/parent relationship is persisted.
        os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")

    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
    if experiment_id:
        mlflow.set_experiment(experiment_id=experiment_id)
        return mlflow.get_experiment(experiment_id)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)
    return mlflow.set_experiment(experiment_name)


def log_evaluation_summary(
    run_name: str,
    params: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> str:
    """Log one offline evaluation/ablation configuration as an MLflow run."""
    configure_mlflow()
    metric_names = (
        "faithfulness_mean",
        "faithfulness_std",
        "relevancy_mean",
        "relevancy_std",
        "avg_retrieval_ms",
        "avg_generation_ms",
    )
    metrics = {
        key: float(summary[key])
        for key in metric_names
        if key in summary and summary[key] is not None
    }
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(params))
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"pipeline": "legal-aid-rag", "stage": "offline-evaluation"})
        return run.info.run_id
