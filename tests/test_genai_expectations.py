"""Tests for human-reviewed MLflow dataset expectations."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src import genai_expectations


class FakeDataset:
    def __init__(self, facts_by_query=None):
        self.name = "legal-aid-rag-held-out-v1"
        self.dataset_id = "d-test"
        self.digest = "digest-test"
        self.facts_by_query = dict(facts_by_query or {})
        self.merge_calls = []

    def to_df(self):
        return pd.DataFrame(
            [
                {
                    "inputs": {"query": query},
                    "expectations": {"expected_facts": facts},
                }
                for query, facts in self.facts_by_query.items()
            ]
        )

    def merge_records(self, records):
        self.merge_calls.append(records)
        for record in records:
            self.facts_by_query[record["inputs"]["query"]] = record[
                "expectations"
            ]["expected_facts"]
        return self


def test_build_expectation_records_adds_review_metadata():
    records = genai_expectations.build_expectation_records(
        {"Question?": ["Fact one.", "Fact two."]},
        reviewer="Karam Hussain",
    )

    assert records[0]["expectations"] == {
        "expected_facts": ["Fact one.", "Fact two."]
    }
    assert records[0]["tags"]["review_status"] == "human_reviewed"
    assert records[0]["tags"]["reviewed_by"] == "Karam Hussain"
    assert records[0]["tags"]["mlflow.user"] == "Karam Hussain"


def test_build_expectation_records_validates_reviewer_and_facts():
    with pytest.raises(ValueError, match="Reviewer"):
        genai_expectations.build_expectation_records({"Q": ["F"]}, reviewer=" ")
    with pytest.raises(ValueError, match="non-empty"):
        genai_expectations.build_expectation_records({"Q": [" "]}, reviewer="Reviewer")
    with pytest.raises(ValueError, match="unique"):
        genai_expectations.build_expectation_records(
            {"Q": ["same", "same"]}, reviewer="Reviewer"
        )


def test_ensure_expectations_updates_and_verifies_records(monkeypatch):
    dataset = FakeDataset()
    tags = []
    monkeypatch.setattr(
        genai_expectations,
        "configure_mlflow",
        lambda: SimpleNamespace(experiment_id="7"),
    )
    monkeypatch.setattr(genai_expectations, "search_datasets", lambda **_: [dataset])
    monkeypatch.setattr(genai_expectations, "get_dataset", lambda **_: dataset)
    monkeypatch.setattr(
        genai_expectations,
        "set_dataset_tags",
        lambda **kwargs: tags.append(kwargs),
    )

    report = genai_expectations.ensure_dataset_expectations(
        reviewer="Karam Hussain",
        expectations_by_query={"Question?": ["Expected fact."]},
    )

    assert report["records_updated"] == 1
    assert report["persisted_expectation_count"] == 1
    assert len(dataset.merge_calls) == 1
    assert tags[0]["tags"]["label_status"] == "human-reviewed"


def test_ensure_expectations_is_idempotent(monkeypatch):
    facts = {"Question?": ["Expected fact."]}
    dataset = FakeDataset(facts)
    monkeypatch.setattr(
        genai_expectations,
        "configure_mlflow",
        lambda: SimpleNamespace(experiment_id="7"),
    )
    monkeypatch.setattr(genai_expectations, "search_datasets", lambda **_: [dataset])
    monkeypatch.setattr(genai_expectations, "get_dataset", lambda **_: dataset)
    monkeypatch.setattr(genai_expectations, "set_dataset_tags", lambda **_: None)

    report = genai_expectations.ensure_dataset_expectations(
        reviewer="Karam Hussain",
        expectations_by_query=facts,
    )

    assert report["records_updated"] == 0
    assert dataset.merge_calls == []

