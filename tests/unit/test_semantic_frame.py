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


class TestTriggerThresholdOperatorInference:
    """Test operator inference for trigger-threshold language.

    E.g., 'if throughput drops below 428 Mbps' means throughput should be
    maintained at_least 428 Mbps (428 is the floor, not a cap).
    """

    def test_drops_below_means_at_least(self):
        """'if throughput drops below 428 Mbps' → throughput at_least 428."""
        text = (
            "if throughput drops below 428 Mbps or reliability falls under 99.917%, "
            "the system automatically triggers corrective resource reallocation within 265 ms"
        )
        assert infer_operator(text, "throughput_mbps", "predictive_assurance") == "at_least"
        assert infer_operator(text, "reliability_percent", "predictive_assurance") == "at_least"

    def test_falls_under_means_at_least(self):
        """'reliability falls under 99.9%' → reliability at_least 99.9."""
        text = "if reliability falls under 99.9%, trigger corrective action"
        assert infer_operator(text, "reliability_percent", "predictive_assurance") == "at_least"

    def test_dips_below_means_at_least(self):
        """'throughput dips below 500 Mbps' → throughput at_least 500."""
        text = "when throughput dips below 500 Mbps, initiate remediation"
        assert infer_operator(text, "throughput_mbps", "predictive_assurance") == "at_least"

    def test_goes_below_means_at_least(self):
        """'latency goes below 5 ms' → latency at_least 5 (unusual but consistent)."""
        text = "if latency goes below 5 ms, alert the operator"
        assert infer_operator(text, "latency_ms", "predictive_assurance") == "at_least"

    def test_exceeds_means_at_most(self):
        """'if latency exceeds 10 ms' → latency at_most 10."""
        text = "if latency exceeds 10 ms, trigger failover"
        assert infer_operator(text, "latency_ms", "predictive_assurance") == "at_most"

    def test_breach_means_at_least(self):
        """'if throughput breaches below 428 Mbps' → throughput at_least 428."""
        text = "if throughput breaches below 428 Mbps, trigger remediation"
        assert infer_operator(text, "throughput_mbps", "predictive_assurance") == "at_least"

    def test_normal_below_still_at_most(self):
        """'latency below 10 ms' (no trigger context) → latency at_most 10."""
        text = "Ensure latency below 10 ms for the service"
        assert infer_operator(text, "latency_ms", "provisioning") == "at_most"

    def test_normal_under_still_at_most(self):
        """'energy under 200 kWh' (no trigger context) → energy at_most 200."""
        text = "Keep energy consumption under 200 kWh per day"
        assert infer_operator(text, "energy_kwh", "energy") == "at_most"


class TestMetaphoricalOperatorInference:
    """Test operator inference for metaphorical constraint language.

    E.g., 'throughput floor of 500 Mbps' → at_least 500,
    'reliability ceiling of 99.9%' → at_most 99.9.
    """

    def test_floor_means_at_least(self):
        """'555 Mbps throughput floor' → throughput at_least 555."""
        text = "ensure a throughput floor of 555 Mbps"
        assert infer_operator(text, "throughput_mbps", "reporting") == "at_least"

    def test_ceiling_means_at_most(self):
        """'99.981% reliability ceiling' → reliability at_most 99.981."""
        text = "maintain a reliability ceiling of 99.981%"
        assert infer_operator(text, "reliability_percent", "reporting") == "at_most"

    def test_minimum_means_at_least(self):
        """'minimum throughput of 220 Mbps' → throughput at_least 220."""
        text = "guarantee a minimum throughput of 220 Mbps"
        assert infer_operator(text, "throughput_mbps", "resilience") == "at_least"

    def test_maximum_means_at_most(self):
        """'maximum latency of 5 ms' → latency at_most 5."""
        text = "enforce a maximum latency of 5 ms"
        assert infer_operator(text, "latency_ms", "provisioning") == "at_most"

    def test_cap_means_at_most(self):
        """'latency capped at 3.41 ms' → latency at_most 3.41."""
        text = "strict end-to-end latency capped at 3.41 ms"
        assert infer_operator(text, "latency_ms", "energy") == "at_most"

    def test_guarantee_means_at_least(self):
        """'reliability guarantee of 99.9984%' → reliability at_least 99.9984."""
        text = "a reliability guarantee of 99.9984%"
        assert infer_operator(text, "reliability_percent", "energy") == "at_least"