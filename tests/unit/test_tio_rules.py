"""Tests for tio_rules.py — TIO structural compliance validation."""
from tmf921_dataset_gen.validation.tio_rules import evaluate_tio_compliance


class TestJsonLdTioCompliance:
    def test_full_compliance(self):
        payload = {
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {
                    "@context": {
                        "icm": "http://www.models.tmforum.org/tio/v1.0.0/IntentCommonModel#",
                        "idan": "http://www.idan-tmforum-catalyst.org/IntentDrivenAutonomousNetworks#",
                    },
                    "@graph": [
                        {"@id": "idan:test", "@type": "icm:Intent"},
                        {
                            "@id": "idan:test:expectation",
                            "@type": "icm:DeliveryExpectation",
                            "icm:params": {"met:latency": [{"icm:atMost": "10 ms"}]},
                        },
                    ],
                },
            }
        }
        result = evaluate_tio_compliance(payload)
        assert result["score"] == 1.0
        assert result["notes"] == []

    def test_missing_icm_prefix(self):
        payload = {
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {
                    "@context": {},
                    "@graph": [
                        {"@id": "test", "@type": "icm:Intent"},
                        {"@id": "test:expectation", "@type": "icm:DeliveryExpectation"},
                    ],
                },
            }
        }
        result = evaluate_tio_compliance(payload)
        assert result["score"] < 1.0
        assert any("icm" in note for note in result["notes"])

    def test_missing_intent(self):
        payload = {
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {
                    "@context": {"icm": "http://www.models.tmforum.org/tio/v1.0.0/IntentCommonModel#"},
                    "@graph": [
                        {"@id": "test:expectation", "@type": "icm:DeliveryExpectation"},
                    ],
                },
            }
        }
        result = evaluate_tio_compliance(payload)
        assert any("Intent" in note for note in result["notes"])

    def test_missing_expectation(self):
        payload = {
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {
                    "@context": {"icm": "http://www.models.tmforum.org/tio/v1.0.0/IntentCommonModel#"},
                    "@graph": [
                        {"@id": "test", "@type": "icm:Intent"},
                    ],
                },
            }
        }
        result = evaluate_tio_compliance(payload)
        assert any("expectation" in note.lower() for note in result["notes"])

    def test_missing_kpi_constraints(self):
        payload = {
            "expression": {
                "@type": "JsonLdExpression",
                "expressionValue": {
                    "@context": {
                        "icm": "http://www.models.tmforum.org/tio/v1.0.0/IntentCommonModel#",
                        "idan": "http://www.idan-tmforum-catalyst.org/IntentDrivenAutonomousNetworks#",
                    },
                    "@graph": [
                        {"@id": "idan:test", "@type": "icm:Intent"},
                        {"@id": "idan:test:expectation", "@type": "icm:DeliveryExpectation"},
                    ],
                },
            }
        }
        result = evaluate_tio_compliance(payload)
        assert any("KPI" in note for note in result["notes"])


class TestTurtleTioCompliance:
    def test_valid_turtle(self):
        ttl = """@prefix icm: <http://www.models.tmforum.org/tio/v1.0.0/IntentCommonModel#> .
@prefix met: <http://www.sdo2.org/TelecomMetrics/Version_1.0#> .
@prefix idan: <http://www.idan-tmforum-catalyst.org/IntentDrivenAutonomousNetworks#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

idan:test a icm:Intent ;
    icm:hasExpectation idan:test:expectation .

idan:test:expectation a icm:DeliveryExpectation ;
    met:latency [ icm:atMost "10 ms" ] .
"""
        payload = {"expression": {"@type": "TurtleExpression", "expressionValue": ttl}}
        result = evaluate_tio_compliance(payload)
        assert result["score"] == 1.0
        assert result["notes"] == []

    def test_invalid_turtle(self):
        payload = {"expression": {"@type": "TurtleExpression", "expressionValue": "not valid turtle"}}
        result = evaluate_tio_compliance(payload)
        assert result["score"] < 1.0
        assert any("turtle parse" in note.lower() for note in result["notes"])


class TestUnsupportedExpression:
    def test_unknown_type(self):
        payload = {"expression": {"@type": "UnknownType"}}
        result = evaluate_tio_compliance(payload)
        assert result["score"] == 0.0
        assert any("unsupported" in note.lower() for note in result["notes"])
