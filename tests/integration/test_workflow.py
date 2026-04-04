from pathlib import Path

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.graph.workflow import run_generation
from tmf921_dataset_gen.agents.critic import CriticRefinementAgent
from tmf921_dataset_gen.validation.semantic_frame import build_intent_frame


def test_run_generation_produces_exportable_records(tmp_path: Path) -> None:
    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "workflow-index"
    settings.repo.output_dir = tmp_path / "output"
    records = run_generation(settings, count=3, output_dir=tmp_path / "dataset")
    assert len(records) == 3
    assert (tmp_path / "dataset" / "dataset.jsonl").exists()
    assert (tmp_path / "dataset" / "manifest.json").exists()
    assert (tmp_path / "dataset" / "release_audit.json").exists()


def test_critic_rejects_missing_constraints(tmp_path: Path) -> None:
    """Test that critic rejects payloads with missing constraints."""
    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "workflow-index"

    critic = CriticRefinementAgent(settings)

    nl_intent = "Ensure throughput of at least 500 Mbps and latency below 10 ms"
    taxonomy_target = {
        "layer": "service",
        "traffic_profile": "embb",
        "scenario_family": "provisioning",
        "domain_context": "generic",
        "taxonomy_category": "service/embb/provisioning"
    }

    # Build intent frame with both constraints
    intent_frame = build_intent_frame(nl_intent, taxonomy_target)

    # Create payload missing latency constraint
    payload = {
        "@type": "Intent",
        "name": "Test Intent",
        "description": nl_intent,
        "expression": {
            "@type": "JsonLdExpression",
            "expressionValue": {
                "@graph": [{
                    "@type": "icm:DeliveryExpectation",
                    "icm:params": {
                        "met:throughput": [{"icm:atLeast": "500 Mbps"}]
                        # Missing latency
                    }
                }]
            }
        }
    }

    report = critic.review(
        nl_intent, payload, "json-ld", taxonomy_target, [], intent_frame
    )

    assert report["accepted"] is False
    assert report["semantic_pass"] is False
    assert "missing constraint: latency_ms" in str(report["notes"])


def test_critic_rejects_wrong_operators(tmp_path: Path) -> None:
    """Test that critic rejects payloads with wrong operators."""
    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "workflow-index"

    critic = CriticRefinementAgent(settings)

    nl_intent = "Ensure throughput of at least 500 Mbps"
    taxonomy_target = {
        "layer": "service",
        "traffic_profile": "embb",
        "scenario_family": "provisioning",
        "domain_context": "generic",
        "taxonomy_category": "service/embb/provisioning"
    }

    intent_frame = build_intent_frame(nl_intent, taxonomy_target)

    # Create payload with wrong operator (atMost instead of atLeast)
    payload = {
        "@type": "Intent",
        "name": "Test Intent",
        "description": nl_intent,
        "expression": {
            "@type": "JsonLdExpression",
            "expressionValue": {
                "@graph": [{
                    "@type": "icm:DeliveryExpectation",
                    "icm:params": {
                        "met:throughput": [{"icm:atMost": "500 Mbps"}]  # Wrong operator
                    }
                }]
            }
        }
    }

    report = critic.review(
        nl_intent, payload, "json-ld", taxonomy_target, [], intent_frame
    )

    assert report["accepted"] is False
    assert report["operator_pass"] is False
    assert "operator mismatch: throughput_mbps" in str(report["notes"])


def test_critic_rejects_contradictory_names(tmp_path: Path) -> None:
    """Test that critic rejects payloads with contradictory names/context."""
    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "workflow-index"

    critic = CriticRefinementAgent(settings)

    nl_intent = "Provision eMBB service"
    taxonomy_target = {
        "layer": "service",
        "traffic_profile": "embb",
        "scenario_family": "provisioning",
        "domain_context": "generic",
        "taxonomy_category": "service/embb/provisioning"
    }

    intent_frame = build_intent_frame(nl_intent, taxonomy_target)

    # Create payload with contradictory name
    payload = {
        "@type": "Intent",
        "name": "Monitor URLLC service",  # Contradictory - monitoring vs provisioning
        "description": nl_intent,
        "expression": {
            "@type": "JsonLdExpression",
            "expressionValue": {
                "@graph": [{
                    "@type": "icm:DeliveryExpectation",
                    "icm:params": {}
                }]
            }
        }
    }

    report = critic.review(
        nl_intent, payload, "json-ld", taxonomy_target, [], intent_frame
    )

    assert report["accepted"] is False
    assert report["contradiction_count"] > 0
    assert any("contradictory scenario" in note for note in report["notes"])


def test_critic_accepts_valid_payload(tmp_path: Path) -> None:
    """Test that critic accepts valid payloads."""
    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "workflow-index"

    critic = CriticRefinementAgent(settings)

    nl_intent = "Ensure throughput of at least 500 Mbps"
    taxonomy_target = {
        "layer": "service",
        "traffic_profile": "embb",
        "scenario_family": "provisioning",
        "domain_context": "generic",
        "taxonomy_category": "service/embb/provisioning"
    }

    intent_frame = build_intent_frame(nl_intent, taxonomy_target)

    # Create valid payload with proper schema compliance
    payload = {
        "@type": "Intent",
        "name": "EMBB Delivery Intent",
        "description": nl_intent,
        "priority": "high",
        "context": "generic context",
        "version": "1.0",
        "lifecycleStatus": "active",
        "expression": {
            "@type": "JsonLdExpression",
            "@baseType": "IntentExpression",
            "iri": "https://tmf921.dataset.local/expressions/test-intent",
            "expressionValue": {
                "@context": {
                    "icm": "http://www.models.tmforum.org/tio/v1.0.0/IntentCommonModel#",
                    "idan": "http://www.idan-tmforum-catalyst.org/IntentDrivenAutonomousNetworks#",
                    "sli": "http://io.irc.huawei.com/Io/v1.0.0/SliceExtensionModel#",
                    "met": "http://www.sdo2.org/TelecomMetrics/Version_1.0#",
                    "xsd": "http://www.w3.org/2001/XMLSchema#",
                    "t": "http://www.w3.org/2006/time#",
                },
                "@graph": [
                    {
                        "@id": "idan:test-intent",
                        "@type": "icm:Intent",
                        "icm:intentOwner": "idan:DatasetGenerator",
                        "icm:hasExpectation": [{"@id": "idan:test-intent:expectation"}],
                    },
                    {
                        "@id": "idan:test-intent:expectation",
                        "@type": "icm:DeliveryExpectation",
                        "icm:target": {"@id": "_:test-target"},
                        "icm:params": {
                            "icm:targetDescription": "6G service embb intent",
                            "met:throughput": [{"icm:atLeast": "500 Mbps"}],
                        },
                    },
                ]
            }
        }
    }

    report = critic.review(
        nl_intent, payload, "json-ld", taxonomy_target, [], intent_frame
    )

    assert report["accepted"] is True
    assert report["semantic_pass"] is True
    assert report["operator_pass"] is True
    assert report["contradiction_count"] == 0
