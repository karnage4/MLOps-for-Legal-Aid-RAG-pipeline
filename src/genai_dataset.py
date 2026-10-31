"""Create and validate the MLflow GenAI evaluation dataset.

This is intentionally separate from the evaluation runner. Step 1 only
registers the shared held-out inputs; later steps can attach judges and run
profiles against the same dataset without changing the RAG pipeline.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Iterable

from mlflow.genai.datasets import (
    add_dataset_to_experiments,
    create_dataset,
    delete_dataset,
    get_dataset,
    search_datasets,
    set_dataset_tags,
)

from src.evaluation_queries import TEST_QUERIES
from src.mlflow_config import configure_mlflow


DATASET_NAME = "legal-aid-rag-held-out-v1"
DATASET_QUERY_COUNT = 3
BASE_DATASET_TAGS = {
    "project": "legal-aid-rag",
    "purpose": "rag-regression-evaluation",
    "version": "1",
    "label_status": "unlabeled",
    "source": "curated-held-out-queries",
}


def build_dataset_records(
    queries: Iterable[str],
    *,
    creator: str | None = None,
) -> list[dict[str, Any]]:
    """Validate held-out queries and convert them to MLflow dataset records."""
    normalized = [query.strip() for query in queries]
    if not normalized or any(not query for query in normalized):
        raise ValueError("Evaluation queries must be non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Evaluation queries must be unique")
    creator = creator.strip() if creator else None
    return [
        {
            "inputs": {"query": query},
            **({"tags": {"mlflow.user": creator}} if creator else {}),
        }
        for query in normalized
    ]


def _queries_from_dataset(dataset: Any) -> list[str]:
    frame = dataset.to_df()
    queries: list[str] = []
    for row in frame.to_dict(orient="records"):
        inputs = row.get("inputs")
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except json.JSONDecodeError:
                inputs = None
        if isinstance(inputs, dict) and isinstance(inputs.get("query"), str):
            queries.append(inputs["query"].strip())
    return queries


def ensure_evaluation_dataset(
    queries: Iterable[str],
    *,
    experiment: Any | None = None,
    dataset_name: str = DATASET_NAME,
    creator: str | None = None,
    recreate: bool = False,
) -> dict[str, Any]:
    """Create or update the managed dataset and verify the expected records."""
    creator = creator.strip() if creator else None
    records = build_dataset_records(queries, creator=creator)
    expected_queries = [record["inputs"]["query"] for record in records]
    experiment = experiment or configure_mlflow()
    experiment_id = str(experiment.experiment_id)

    matches = search_datasets(
        filter_string=f'name = "{dataset_name}"',
        max_results=2,
    )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple MLflow datasets are named {dataset_name!r}; use a unique name"
        )

    if matches and creator and matches[0].created_by != creator and not recreate:
        raise ValueError(
            f"Dataset creator is {matches[0].created_by!r}; MLflow does not "
            "support updating created_by in place. Rerun with recreate=True."
        )
    if matches and recreate:
        delete_dataset(dataset_id=matches[0].dataset_id)
        matches = []

    dataset_tags = dict(BASE_DATASET_TAGS)
    if creator:
        dataset_tags["mlflow.user"] = creator

    created = not matches
    dataset = (
        create_dataset(
            name=dataset_name,
            experiment_id=[experiment_id],
            tags=dataset_tags,
        )
        if created
        else matches[0]
    )

    if experiment_id not in {str(value) for value in dataset.experiment_ids}:
        dataset = add_dataset_to_experiments(
            dataset_id=dataset.dataset_id,
            experiment_ids=[experiment_id],
        )

    set_dataset_tags(dataset_id=dataset.dataset_id, tags=dataset_tags)

    existing_queries = set(_queries_from_dataset(dataset))
    missing_records = [
        record
        for record in records
        if record["inputs"]["query"] not in existing_queries
    ]
    if missing_records:
        dataset.merge_records(missing_records)

    dataset = get_dataset(dataset_id=dataset.dataset_id)
    persisted_queries = _queries_from_dataset(dataset)
    missing_after_merge = sorted(set(expected_queries).difference(persisted_queries))
    if missing_after_merge:
        raise RuntimeError(
            "MLflow dataset verification failed; missing queries: "
            + ", ".join(missing_after_merge)
        )

    return {
        "status": "ok",
        "dataset_name": dataset.name,
        "dataset_id": dataset.dataset_id,
        "dataset_digest": dataset.digest,
        "experiment_id": experiment_id,
        "created": created,
        "records_added": len(missing_records),
        "expected_query_count": len(expected_queries),
        "persisted_record_count": len(persisted_queries),
        "label_status": BASE_DATASET_TAGS["label_status"],
        "created_by": dataset.created_by,
        "last_updated_by": dataset.last_updated_by,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or validate the managed MLflow GenAI dataset."
    )
    parser.add_argument(
        "--creator",
        help="Creator name recorded through MLflow's reserved mlflow.user tag.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the named dataset (required to change created_by).",
    )
    args = parser.parse_args()
    report = ensure_evaluation_dataset(
        TEST_QUERIES[:DATASET_QUERY_COUNT],
        creator=args.creator,
        recreate=args.recreate,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
