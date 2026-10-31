"""Attach approved human expectations to the managed MLflow dataset."""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping, Sequence

from mlflow.genai.datasets import get_dataset, search_datasets, set_dataset_tags

from src.evaluation_expectations import EXPECTATION_VERSION, EXPECTED_FACTS_BY_QUERY
from src.genai_dataset import DATASET_NAME
from src.mlflow_config import configure_mlflow


def build_expectation_records(
    expectations_by_query: Mapping[str, Sequence[str]],
    *,
    reviewer: str,
) -> list[dict[str, Any]]:
    """Validate approved facts and build input-keyed MLflow upsert records."""
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("Reviewer name must not be empty")
    if not expectations_by_query:
        raise ValueError("At least one query expectation is required")

    records = []
    for raw_query, raw_facts in expectations_by_query.items():
        query = raw_query.strip()
        facts = [fact.strip() for fact in raw_facts]
        if not query or not facts or any(not fact for fact in facts):
            raise ValueError("Queries and expected facts must be non-empty")
        if len(facts) != len(set(facts)):
            raise ValueError(f"Expected facts must be unique for query: {query}")
        records.append(
            {
                "inputs": {"query": query},
                "expectations": {"expected_facts": facts},
                "tags": {
                    "review_status": "human_reviewed",
                    "reviewed_by": reviewer,
                    "expectation_version": EXPECTATION_VERSION,
                    "mlflow.user": reviewer,
                },
            }
        )
    return records


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


def _facts_by_query(dataset: Any) -> dict[str, list[str]]:
    facts_by_query: dict[str, list[str]] = {}
    for row in dataset.to_df().to_dict(orient="records"):
        inputs = _decode_mapping(row.get("inputs"))
        expectations = _decode_mapping(row.get("expectations"))
        query = inputs.get("query")
        facts = expectations.get("expected_facts")
        if isinstance(query, str) and isinstance(facts, list):
            facts_by_query[query.strip()] = facts
    return facts_by_query


def ensure_dataset_expectations(
    *,
    reviewer: str,
    expectations_by_query: Mapping[str, Sequence[str]] = EXPECTED_FACTS_BY_QUERY,
    dataset_name: str = DATASET_NAME,
) -> dict[str, Any]:
    """Upsert changed expectations and verify their persisted values."""
    experiment = configure_mlflow()
    matches = search_datasets(filter_string=f'name = "{dataset_name}"', max_results=2)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one dataset named {dataset_name!r}; found {len(matches)}"
        )

    dataset = matches[0]
    records = build_expectation_records(expectations_by_query, reviewer=reviewer)
    existing_facts = _facts_by_query(dataset)
    changed_records = [
        record
        for record in records
        if existing_facts.get(record["inputs"]["query"])
        != record["expectations"]["expected_facts"]
    ]
    if changed_records:
        dataset.merge_records(changed_records)

    dataset_tags = {
        "label_status": "human-reviewed",
        "expectation_version": EXPECTATION_VERSION,
        "reviewed_by": reviewer.strip(),
        "mlflow.user": reviewer.strip(),
    }
    set_dataset_tags(dataset_id=dataset.dataset_id, tags=dataset_tags)

    dataset = get_dataset(dataset_id=dataset.dataset_id)
    persisted = _facts_by_query(dataset)
    expected = {
        record["inputs"]["query"]: record["expectations"]["expected_facts"]
        for record in records
    }
    if any(persisted.get(query) != facts for query, facts in expected.items()):
        raise RuntimeError("MLflow expectation verification failed after dataset upsert")

    return {
        "status": "ok",
        "dataset_name": dataset.name,
        "dataset_id": dataset.dataset_id,
        "dataset_digest": dataset.digest,
        "experiment_id": str(experiment.experiment_id),
        "reviewer": reviewer.strip(),
        "expectation_version": EXPECTATION_VERSION,
        "expected_record_count": len(expected),
        "persisted_expectation_count": len(persisted),
        "records_updated": len(changed_records),
        "label_status": "human-reviewed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach approved expected facts to the MLflow GenAI dataset."
    )
    parser.add_argument("--reviewer", required=True, help="Human reviewer name.")
    args = parser.parse_args()
    print(json.dumps(ensure_dataset_expectations(reviewer=args.reviewer), indent=2))


if __name__ == "__main__":
    main()

