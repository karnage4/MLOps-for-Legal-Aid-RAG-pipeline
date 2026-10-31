"""Tests for the managed MLflow GenAI evaluation dataset setup."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src import genai_dataset


QUERIES = [
    "Who inherits property when a person dies without a will?",
    "What is the share of a daughter in Islamic inheritance?",
    "Can a non-Muslim inherit from a Muslim's estate?",
]


class FakeDataset:
    def __init__(self, queries=(), experiment_ids=("7",)):
        self.name = genai_dataset.DATASET_NAME
        self.dataset_id = "d-test"
        self.digest = "digest-test"
        self.experiment_ids = list(experiment_ids)
        self.queries = list(queries)
        self.created_by = "test-user"
        self.last_updated_by = "test-user"

    def to_df(self):
        return pd.DataFrame(
            [{"inputs": {"query": query}} for query in self.queries]
        )

    def merge_records(self, records):
        self.queries.extend(record["inputs"]["query"] for record in records)
        return self


def test_build_dataset_records_validates_missing_and_duplicate_queries():
    assert genai_dataset.build_dataset_records(QUERIES) == [
        {"inputs": {"query": query}} for query in QUERIES
    ]

    with pytest.raises(ValueError, match="non-empty"):
        genai_dataset.build_dataset_records(["valid", " "])
    with pytest.raises(ValueError, match="unique"):
        genai_dataset.build_dataset_records(["same", "same"])


def test_ensure_dataset_creates_and_verifies_records(monkeypatch):
    dataset = FakeDataset(queries=())
    created_calls = []
    tag_calls = []

    monkeypatch.setattr(genai_dataset, "search_datasets", lambda **_: [])

    def fake_create_dataset(**kwargs):
        created_calls.append(kwargs)
        return dataset

    monkeypatch.setattr(genai_dataset, "create_dataset", fake_create_dataset)
    monkeypatch.setattr(
        genai_dataset,
        "set_dataset_tags",
        lambda **kwargs: tag_calls.append(kwargs),
    )
    monkeypatch.setattr(
        genai_dataset,
        "get_dataset",
        lambda **_: dataset,
    )

    report = genai_dataset.ensure_evaluation_dataset(
        QUERIES,
        experiment=SimpleNamespace(experiment_id="7"),
    )

    assert dataset.queries == QUERIES
    assert report["created"] is True
    assert report["records_added"] == 3
    assert report["persisted_record_count"] == 3
    assert created_calls[0]["experiment_id"] == ["7"]
    assert created_calls[0]["tags"]["label_status"] == "unlabeled"
    assert tag_calls[0]["dataset_id"] == "d-test"


def test_ensure_dataset_is_idempotent(monkeypatch):
    dataset = FakeDataset(queries=QUERIES)
    merge_calls = []
    dataset.merge_records = lambda records: merge_calls.append(records)

    monkeypatch.setattr(
        genai_dataset,
        "search_datasets",
        lambda **_: [dataset],
    )
    monkeypatch.setattr(genai_dataset, "set_dataset_tags", lambda **_: None)
    monkeypatch.setattr(genai_dataset, "get_dataset", lambda **_: dataset)

    report = genai_dataset.ensure_evaluation_dataset(
        QUERIES,
        experiment=SimpleNamespace(experiment_id="7"),
    )

    assert merge_calls == []
    assert report["created"] is False
    assert report["records_added"] == 0
    assert report["persisted_record_count"] == 3


def test_ensure_dataset_links_existing_dataset_to_experiment(monkeypatch):
    dataset = FakeDataset(queries=QUERIES, experiment_ids=("old",))
    link_calls = []

    monkeypatch.setattr(
        genai_dataset,
        "search_datasets",
        lambda **_: [dataset],
    )

    def fake_link(**kwargs):
        link_calls.append(kwargs)
        dataset.experiment_ids.append("7")
        return dataset

    monkeypatch.setattr(genai_dataset, "add_dataset_to_experiments", fake_link)
    monkeypatch.setattr(genai_dataset, "set_dataset_tags", lambda **_: None)
    monkeypatch.setattr(genai_dataset, "get_dataset", lambda **_: dataset)

    genai_dataset.ensure_evaluation_dataset(
        QUERIES,
        experiment=SimpleNamespace(experiment_id="7"),
    )

    assert link_calls == [
        {"dataset_id": "d-test", "experiment_ids": ["7"]}
    ]


def test_recreate_changes_creator_through_supported_mlflow_tags(monkeypatch):
    old_dataset = FakeDataset(queries=QUERIES)
    old_dataset.created_by = "SHEIKHANI LAPTOP"
    new_dataset = FakeDataset(queries=())
    new_dataset.created_by = "Karam Hussain"
    new_dataset.last_updated_by = "Karam Hussain"
    delete_calls = []

    monkeypatch.setattr(
        genai_dataset,
        "search_datasets",
        lambda **_: [old_dataset],
    )
    monkeypatch.setattr(
        genai_dataset,
        "delete_dataset",
        lambda **kwargs: delete_calls.append(kwargs),
    )
    monkeypatch.setattr(
        genai_dataset,
        "create_dataset",
        lambda **_: new_dataset,
    )
    monkeypatch.setattr(genai_dataset, "set_dataset_tags", lambda **_: None)
    monkeypatch.setattr(genai_dataset, "get_dataset", lambda **_: new_dataset)

    report = genai_dataset.ensure_evaluation_dataset(
        QUERIES,
        experiment=SimpleNamespace(experiment_id="7"),
        creator="Karam Hussain",
        recreate=True,
    )

    assert delete_calls == [{"dataset_id": "d-test"}]
    assert new_dataset.queries == QUERIES
    assert report["created_by"] == "Karam Hussain"
