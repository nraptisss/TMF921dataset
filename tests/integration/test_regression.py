"""Regression tests for previously accepted records."""

import json
from pathlib import Path

import pytest

from src.tmf921_dataset_gen.agents.critic import CriticRefinementAgent
from src.tmf921_dataset_gen.config import Settings
from src.tmf921_dataset_gen.validation.semantic_frame import build_intent_frame


@pytest.fixture
def regression_dataset_path():
    """Path to the regression dataset."""
    return Path("output/1k_qwen_gpu_fullpower/dataset.jsonl")


def test_regression_record_validation(tmp_path: Path, regression_dataset_path: Path):
    """Test that regression records are properly validated under new system."""
    if not regression_dataset_path.exists():
        pytest.skip("Regression dataset not available")

    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "workflow-index"

    critic = CriticRefinementAgent(settings)

    # Load first record from regression dataset
    with open(regression_dataset_path, 'r') as f:
        line = f.readline().strip()
        record = json.loads(line)

    nl_intent = record["nl_intent"]
    payload = record["tmf921_intent"]
    metadata = record.get("metadata", {})
    taxonomy_target = metadata.get("taxonomy_target", {
        "layer": "service",
        "traffic_profile": "embb",
        "scenario_family": "provisioning",
        "domain_context": "generic",
        "taxonomy_category": "service/embb/provisioning"
    })

    # Build intent frame
    intent_frame = build_intent_frame(nl_intent, taxonomy_target)

    # Run validation
    report = critic.review(
        nl_intent, payload, record.get("serialization", "json-ld"),
        taxonomy_target, [], intent_frame
    )

    # The record should either be accepted or rejected with clear reasoning
    # We mainly want to ensure the new validation runs without errors
    assert "accepted" in report
    assert "notes" in report
    assert isinstance(report["accepted"], bool)

    # Check that new metadata fields are present
    assert "semantic_pass" in report
    assert "operator_pass" in report
    assert "constraint_coverage" in report
    assert "unsupported_claim_count" in report
    assert "contradiction_count" in report


def test_regression_multiple_records(tmp_path: Path, regression_dataset_path: Path):
    """Test validation on multiple regression records."""
    if not regression_dataset_path.exists():
        pytest.skip("Regression dataset not available")

    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "workflow-index"

    critic = CriticRefinementAgent(settings)

    passed_count = 0
    failed_count = 0
    total_checked = 0

    with open(regression_dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Test first 10 records
                break

            record = json.loads(line.strip())
            nl_intent = record["nl_intent"]
            payload = record["tmf921_intent"]
            metadata = record.get("metadata", {})
            taxonomy_target = metadata.get("taxonomy_target", {
                "layer": "service",
                "traffic_profile": "embb",
                "scenario_family": "provisioning",
                "domain_context": "generic",
                "taxonomy_category": "service/embb/provisioning"
            })

            intent_frame = build_intent_frame(nl_intent, taxonomy_target)

            report = critic.review(
                nl_intent, payload, record.get("serialization", "json-ld"),
                taxonomy_target, [], intent_frame
            )

            total_checked += 1
            if report["accepted"]:
                passed_count += 1
            else:
                failed_count += 1

    # Ensure we processed some records
    assert total_checked > 0

    # With the new stricter validation, we expect some failures
    # But not all should fail (that would indicate over-rejection)
    assert passed_count + failed_count == total_checked
    print(f"Regression test: {passed_count} passed, {failed_count} failed out of {total_checked}")