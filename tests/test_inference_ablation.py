"""Tests for the controlled prompt/model/temperature inference matrix."""

from types import SimpleNamespace

import pytest

from src import inference_ablation


def test_four_run_matrix_changes_one_generation_dimension_at_a_time():
    configs = {config.label: config for config in inference_ablation.INFERENCE_CONFIGS}
    baseline = configs["baseline-low-temp"]

    assert len(configs) == 4
    assert inference_ablation.BASELINE_PROFILE.label == "fixed-hybrid-top5"

    prompt = configs["prompt-structured"]
    assert prompt.prompt_variant != baseline.prompt_variant
    assert prompt.model == baseline.model
    assert prompt.temperature == baseline.temperature

    model = configs["model-b"]
    assert model.prompt_variant == baseline.prompt_variant
    assert model.model != baseline.model
    assert model.temperature == baseline.temperature

    temperature = configs["temperature-high"]
    assert temperature.prompt_variant == baseline.prompt_variant
    assert temperature.model == baseline.model
    assert temperature.temperature > baseline.temperature


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label": "bad label"},
        {"dimension": "unknown"},
        {"prompt_variant": "unknown"},
        {"temperature": 1.1},
        {"provider": "auto"},
    ],
)
def test_inference_config_validation(kwargs):
    values = {
        "label": "valid",
        "dimension": "baseline",
        "prompt_variant": "strict_grounded_v1",
        "model": "model-a",
        "temperature": 0.1,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        inference_ablation.InferenceConfig(**values)


def test_reusable_report_requires_exact_generation_and_lineage():
    config = inference_ablation.INFERENCE_CONFIGS[0]
    judges = [SimpleNamespace(name="judge", scorer_version=1)]
    dataset = {"dataset_digest": "digest"}
    report = {
        "status": "completed",
        "scorer_failures": [],
        "config": {
            "label": config.label,
            "prompt_variant": config.prompt_variant,
            "requested_generation_provider": config.provider,
            "requested_generation_model": config.model,
            "generation_temperature": config.temperature,
        },
        "dataset": dataset,
        "judges": [{"name": "judge", "version": 1}],
    }

    assert inference_ablation.is_reusable_report(
        report,
        config=config,
        dataset_report=dataset,
        judges=judges,
    )
    report["config"]["generation_temperature"] = 0.7
    assert not inference_ablation.is_reusable_report(
        report,
        config=config,
        dataset_report=dataset,
        judges=judges,
    )


def test_effect_cards_report_deltas_from_shared_baseline():
    baseline = {
        "label": "baseline-low-temp",
        "fact_coverage": 0.5,
        "correctness": 0.5,
        "retrieval_sufficiency": 0.5,
        "legal_answer_quality": 0.5,
        "retrieval_groundedness": 0.5,
        "retrieval_relevance": 0.5,
        "relevance_to_query": 0.5,
        "avg_total_ms": 100.0,
    }
    prompt = {**baseline, "label": "prompt-structured", "correctness": 0.75}

    effects = inference_ablation._effect_cards([baseline, prompt])

    assert effects == [
        {
            "dimension": "prompt",
            "baseline": "baseline-low-temp",
            "candidate": "prompt-structured",
            "deltas": {
                "fact_coverage": 0.0,
                "correctness": 0.25,
                "retrieval_sufficiency": 0.0,
                "legal_answer_quality": 0.0,
                "retrieval_groundedness": 0.0,
                "retrieval_relevance": 0.0,
                "relevance_to_query": 0.0,
                "avg_total_ms": 0.0,
            },
        }
    ]
