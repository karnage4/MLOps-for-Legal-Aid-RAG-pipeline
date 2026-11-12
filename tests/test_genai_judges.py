"""Tests for versioned MLflow GenAI judge registration."""

from types import SimpleNamespace

import pytest

from src import genai_judges


class FakeJudge:
    def __init__(self, name, definition, version=None):
        self.name = name
        self.definition = definition
        self.scorer_version = version
        self.register_calls = []

    def model_dump(self, mode="json"):
        return {
            "name": self.name,
            "mlflow_version": "test-version",
            "instructions_judge_pydantic_data": self.definition,
        }

    def register(self, *, experiment_id):
        self.register_calls.append(experiment_id)
        return FakeJudge(self.name, self.definition, version=1)


def test_resolve_judge_model_normalizes_groq_uri(monkeypatch):
    monkeypatch.setenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")
    assert genai_judges.resolve_judge_model() == "groq:/openai/gpt-oss-20b"
    assert (
        genai_judges.resolve_judge_model("groq:/llama/model")
        == "groq:/llama/model"
    )
    with pytest.raises(ValueError, match="groq"):
        genai_judges.resolve_judge_model("openai:/gpt-4o")


def test_build_judges_use_configured_model_and_limit_expectation_usage():
    judges = genai_judges.build_judges("openai/gpt-oss-20b")

    assert tuple(judge.name for judge in judges) == genai_judges.JUDGE_NAMES
    assert all(
        genai_judges._judge_model(judge) == "groq:/openai/gpt-oss-20b"
        for judge in judges
    )
    expectation_dependent = {
        "fact_coverage",
        "correctness",
        "retrieval_sufficiency",
    }
    assert expectation_dependent.issubset({judge.name for judge in judges})
    assert "{{ inputs }}" in genai_judges.LEGAL_ANSWER_QUALITY_INSTRUCTIONS
    assert "{{ outputs }}" in genai_judges.LEGAL_ANSWER_QUALITY_INSTRUCTIONS
    assert "{{ trace }}" not in genai_judges.LEGAL_ANSWER_QUALITY_INSTRUCTIONS
    assert "{{ expectations }}" in genai_judges.FACT_COVERAGE_INSTRUCTIONS
    assert "supported fact count / total expected fact count" in (
        genai_judges.FACT_COVERAGE_INSTRUCTIONS
    )


def test_registration_reuses_identical_latest_version(monkeypatch):
    existing = FakeJudge("judge-a", {"model": "groq:/model"}, version=3)
    desired = FakeJudge("judge-a", {"model": "groq:/model"})
    monkeypatch.setattr(genai_judges, "list_scorers", lambda **_: [existing])

    report = genai_judges.ensure_registered_judges(
        experiment=SimpleNamespace(experiment_id="7", name="experiment"),
        judges=[desired],
    )

    assert desired.register_calls == []
    assert report["inference_calls"] == 0
    assert report["judges"] == [
        {
            "name": "judge-a",
            "version": 3,
            "model": "groq:/model",
            "action": "reused",
        }
    ]


def test_registration_selects_highest_version_regardless_of_list_order(monkeypatch):
    older = FakeJudge("judge-a", {"model": "groq:/old"}, version=1)
    latest = FakeJudge("judge-a", {"model": "groq:/model"}, version=2)
    desired = FakeJudge("judge-a", {"model": "groq:/model"})
    monkeypatch.setattr(
        genai_judges,
        "list_scorers",
        lambda **_: [latest, older],
    )

    report = genai_judges.ensure_registered_judges(
        experiment=SimpleNamespace(experiment_id="7", name="experiment"),
        judges=[desired],
    )

    assert desired.register_calls == []
    assert report["judges"][0]["version"] == 2
    assert report["judges"][0]["action"] == "reused"


def test_registration_creates_or_versions_changed_judges(monkeypatch):
    old = FakeJudge("changed", {"model": "groq:/old"}, version=2)
    changed = FakeJudge("changed", {"model": "groq:/new"})
    new = FakeJudge("new", {"model": "groq:/new"})
    monkeypatch.setattr(genai_judges, "list_scorers", lambda **_: [old])

    report = genai_judges.ensure_registered_judges(
        experiment=SimpleNamespace(experiment_id="7", name="experiment"),
        judges=[changed, new],
    )

    assert changed.register_calls == ["7"]
    assert new.register_calls == ["7"]
    assert [item["action"] for item in report["judges"]] == [
        "versioned",
        "registered",
    ]


def test_duplicate_judge_names_are_rejected(monkeypatch):
    monkeypatch.setattr(genai_judges, "list_scorers", lambda **_: [])
    duplicate = [FakeJudge("same", {}), FakeJudge("same", {})]
    with pytest.raises(ValueError, match="unique"):
        genai_judges.ensure_registered_judges(
            experiment=SimpleNamespace(experiment_id="7", name="experiment"),
            judges=duplicate,
        )
