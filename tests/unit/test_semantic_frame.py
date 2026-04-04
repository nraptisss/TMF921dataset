"""Unit tests for semantic frame validation and symbolic verification."""

import pytest
from src.tmf921_dataset_gen.validation.semantic_frame import (
    build_intent_frame,
    verify_semantic_alignment,
    payload_constraints,
    infer_operator,
    METRIC_SPECS,
)


class TestSemanticFrame:
    """Test semantic frame building and validation."""

    def test_build_intent_frame_basic(self):
        """Test basic intent frame building."""
        nl_intent = "Ensure throughput of at least 500 Mbps for eMBB services"
        taxonomy_target = {
            "layer": "service",
            "traffic_profile": "embb",
            "scenario_family": "provisioning",
            "domain_context": "generic",
            "taxonomy_category": "service/embb/provisioning"
        }

        frame = build_intent_frame(nl_intent, taxonomy_target)

        assert frame["layer"] == "service"
        assert frame["traffic_profile"] == "embb"
        assert frame["scenario_family"] == "provisioning"
        assert len(frame["constraints"]) > 0

        # Check throughput constraint
        throughput_constraint = next((c for c in frame["constraints"] if c["metric"] == "throughput_mbps"), None)
        assert throughput_constraint is not None
        assert throughput_constraint["operator"] == "at_least"
        assert throughput_constraint["value"] == 500

    def test_infer_operator_at_least(self):
        """Test operator inference for 'at least'."""
        text = "throughput of at least 500 Mbps"
        operator = infer_operator(text, "throughput_mbps", "provisioning")
        assert operator == "at_least"

    def test_infer_operator_at_most(self):
        """Test operator inference for 'at most'."""
        text = "latency below 10 ms"
        operator = infer_operator(text, "latency_ms", "provisioning")
        assert operator == "at_most"

    def test_infer_operator_trigger_within(self):
        """Test operator inference for trigger scenarios."""
        text = "failover within 50 ms"
        operator = infer_operator(text, "reaction_time_ms", "predictive_assurance")
        assert operator == "trigger_within"

    def test_verify_semantic_alignment_pass(self):
        """Test successful semantic alignment verification."""
        nl_intent = "Ensure throughput of at least 500 Mbps"
        intent_frame = {
            "constraints": [{
                "metric": "throughput_mbps",
                "value": 500,
                "operator": "at_least",
                "source": "nl_intent"
            }],
            "scenario_family": "provisioning",
            "expectation_type": "icm:DeliveryExpectation"
        }
        payload = {
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {
                    "@graph": [{
                        "@type": "icm:DeliveryExpectation",
                        "icm:params": {
                            "met:throughput": [{"icm:atLeast": "500 Mbps"}]
                        }
                    }]
                }
            }
        }

        result = verify_semantic_alignment(nl_intent, intent_frame, payload)

        assert result["semantic_pass"] is True
        assert result["operator_pass"] is True
        assert result["constraint_coverage"] == 1.0
        assert result["contradiction_count"] == 0

    def test_verify_semantic_alignment_missing_constraint(self):
        """Test detection of missing constraints."""
        nl_intent = "Ensure throughput of at least 500 Mbps and latency below 10 ms"
        intent_frame = {
            "constraints": [
                {
                    "metric": "throughput_mbps",
                    "value": 500,
                    "operator": "at_least",
                    "source": "nl_intent"
                },
                {
                    "metric": "latency_ms",
                    "value": 10,
                    "operator": "at_most",
                    "source": "nl_intent"
                }
            ],
            "scenario_family": "provisioning",
            "expectation_type": "icm:DeliveryExpectation"
        }
        payload = {
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {
                    "@graph": [{
                        "icm:params": {
                            "met:throughput": [{"icm:atLeast": "500 Mbps"}]
                            # Missing latency constraint
                        }
                    }]
                }
            }
        }

        result = verify_semantic_alignment(nl_intent, intent_frame, payload)

        assert result["semantic_pass"] is False
        assert "missing constraint: latency_ms" in result["notes"]
        assert result["constraint_coverage"] < 1.0

    def test_verify_semantic_alignment_operator_mismatch(self):
        """Test detection of operator mismatches."""
        nl_intent = "Ensure throughput of at least 500 Mbps"
        intent_frame = {
            "constraints": [{
                "metric": "throughput_mbps",
                "value": 500,
                "operator": "at_least",
                "source": "nl_intent"
            }],
            "scenario_family": "provisioning",
            "expectation_type": "icm:DeliveryExpectation"
        }
        payload = {
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {
                    "@graph": [{
                        "icm:params": {
                            "met:throughput": [{"icm:atMost": "500 Mbps"}]  # Wrong operator
                        }
                    }]
                }
            }
        }

        result = verify_semantic_alignment(nl_intent, intent_frame, payload)

        assert result["semantic_pass"] is False
        assert result["operator_pass"] is False
        assert "operator mismatch: throughput_mbps" in result["notes"]

    def test_verify_semantic_alignment_contradictory_scenario(self):
        """Test detection of contradictory scenario keywords."""
        nl_intent = "Provision eMBB service"
        intent_frame = {
            "constraints": [],
            "scenario_family": "provisioning",
            "expectation_type": "icm:DeliveryExpectation"
        }
        payload = {
            "name": "Monitor URLLC service",  # Contradictory name
            "context": "Monitoring context",
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {"@graph": []}
            }
        }

        result = verify_semantic_alignment(nl_intent, intent_frame, payload)

        assert result["semantic_pass"] is False
        assert result["contradiction_count"] > 0
        assert any("contradictory scenario" in note for note in result["notes"])

    def test_verify_semantic_alignment_reporting_scenario(self):
        """Test reporting scenario validation."""
        nl_intent = "Monitor latency every 60 seconds"
        intent_frame = {
            "constraints": [{
                "metric": "reporting_interval_seconds",
                "value": 60,
                "operator": "periodic_every",
                "source": "nl_intent"
            }],
            "scenario_family": "reporting",
            "expectation_type": "icm:ReportingExpectation"
        }
        payload = {
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {
                    "@graph": [{
                        "@type": "icm:ReportingExpectation"
                    }]
                }
            }
        }

        result = verify_semantic_alignment(nl_intent, intent_frame, payload)

        # Should pass if expectation type is correct
        assert "wrong expectation type" not in str(result["notes"])

    def test_payload_constraints_jsonld(self):
        """Test constraint extraction from JSON-LD payload."""
        payload = {
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {
                    "@graph": [{
                        "icm:params": {
                            "met:throughput": [{"icm:atLeast": "500 Mbps"}],
                            "met:latency": [{"icm:atMost": "10 ms"}]
                        }
                    }]
                }
            }
        }

        constraints = payload_constraints(payload)

        assert len(constraints) == 2
        throughput = next(c for c in constraints if c["metric"] == "throughput_mbps")
        latency = next(c for c in constraints if c["metric"] == "latency_ms")

        assert throughput["operator"] == "at_least"
        assert throughput["value"] == 500
        assert latency["operator"] == "at_most"
        assert latency["value"] == 10

    def test_round_trip_constraint_extraction(self):
        """Test that constraints can be extracted from generated payloads."""
        # This test ensures the payload rendering and constraint extraction are consistent
        nl_intent = "Ensure throughput >= 1000 Mbps and latency <= 5 ms"
        taxonomy_target = {
            "layer": "service",
            "traffic_profile": "embb",
            "scenario_family": "provisioning",
            "domain_context": "generic",
            "taxonomy_category": "service/embb/provisioning"
        }

        intent_frame = build_intent_frame(nl_intent, taxonomy_target)

        # Simulate payload (this would normally be generated)
        payload = {
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {
                    "@graph": [{
                        "icm:params": {
                            "met:throughput": [{"icm:atLeast": "1000 Mbps"}],
                            "met:latency": [{"icm:atMost": "5 ms"}]
                        }
                    }]
                }
            }
        }

        result = verify_semantic_alignment(nl_intent, intent_frame, payload)

        # Should detect the constraints correctly
        assert len(result["payload_constraints"]) == 2