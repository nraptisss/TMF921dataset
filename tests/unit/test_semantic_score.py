"""Tests for semantic_score.py — KPI extraction and semantic faithfulness scoring."""
from tmf921_dataset_gen.validation.semantic_score import (
    extract_kpis,
    _nlp_kpi_extraction,
    _token_overlap,
    _numeric_match,
    _kpi_match_score,
    intent_payload_to_text,
)


class TestNlpKpiExtraction:
    def test_latency(self):
        result = _nlp_kpi_extraction("Ensure latency below 5 ms for the service")
        assert result["latency_ms"] == 5.0

    def test_throughput_mbps(self):
        result = _nlp_kpi_extraction("Throughput of at least 500 Mbps required")
        assert result["throughput_mbps"] == 500.0

    def test_throughput_gbps(self):
        result = _nlp_kpi_extraction("Need throughput of 10 Gbps")
        assert result["throughput_gbps"] == 10.0

    def test_energy(self):
        result = _nlp_kpi_extraction("Energy consumption under 200 kWh per day")
        assert result["energy_kwh"] == 200.0

    def test_reliability(self):
        result = _nlp_kpi_extraction("Reliability of 99.9%")
        assert result["reliability_percent"] == 99.9

    def test_device_count(self):
        result = _nlp_kpi_extraction("Support for 1000 devices")
        assert result["device_count"] == 1000

    def test_device_count_with_thousands_separator(self):
        result = _nlp_kpi_extraction("Support for 37,606 devices across the campus")
        assert result["device_count"] == 37606

    def test_device_count_no_false_positive(self):
        """Regression: 'for latency below 0.69 ms' should NOT produce device_count."""
        result = _nlp_kpi_extraction("looking for latency below 0.69 ms")
        assert "device_count" not in result

    def test_reporting_interval_seconds(self):
        result = _nlp_kpi_extraction("Reports every 120 seconds")
        assert result["reporting_interval_seconds"] == 120

    def test_reporting_interval_minutes(self):
        result = _nlp_kpi_extraction("Reporting interval of 5 minutes")
        assert result["reporting_interval_seconds"] == 300

    def test_reaction_time_ms(self):
        result = _nlp_kpi_extraction("Failover to backup resources within 50 ms")
        assert result["reaction_time_ms"] == 50.0

    def test_multiple_kpis(self):
        result = _nlp_kpi_extraction(
            "Throughput of at least 428 Mbps, reliability of 99.9%, energy consumption under 481 kWh"
        )
        assert result["throughput_mbps"] == 428.0
        assert result["reliability_percent"] == 99.9
        assert result["energy_kwh"] == 481.0

    def test_no_kpis(self):
        result = _nlp_kpi_extraction("Deploy a network slice for the service")
        assert result == {}


class TestExtractKpis:
    def test_plausibility_negative_latency(self):
        """Regex captures absolute value only; negative sign is lost."""
        result = extract_kpis("latency of -5 ms")
        assert result["latency_ms"] == 5.0

    def test_plausibility_zero_latency(self):
        result = extract_kpis("latency of 0 ms")
        assert "latency_ms" not in result

    def test_plausibility_invalid_reliability(self):
        result = extract_kpis("reliability of 150%")
        assert "reliability_percent" not in result

    def test_valid_kpis_pass(self):
        result = extract_kpis("latency of 10 ms, throughput of 500 Mbps")
        assert result["latency_ms"] == 10.0
        assert result["throughput_mbps"] == 500.0


class TestTokenOverlap:
    def test_identical(self):
        assert _token_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _token_overlap("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        score = _token_overlap("hello world foo", "hello world bar")
        assert 0.0 < score < 1.0

    def test_empty(self):
        assert _token_overlap("", "") == 0.0


class TestNumericMatch:
    def test_exact_match(self):
        assert _numeric_match(100, 100) == 1.0

    def test_close_values(self):
        score = _numeric_match(100, 105)
        assert score > 0.5

    def test_different_values(self):
        score = _numeric_match(100, 1000)
        assert score < 0.5

    def test_string_fallback(self):
        assert _numeric_match("abc", "abc") == 1.0
        assert _numeric_match("abc", "def") == 0.0


class TestKpiMatchScore:
    def test_perfect_match(self):
        nl = {"latency_ms": 10.0, "throughput_mbps": 500.0}
        payload = {"latency_ms": 10.0, "throughput_mbps": 500.0}
        assert _kpi_match_score(nl, payload) == 1.0

    def test_missing_key(self):
        nl = {"latency_ms": 10.0, "throughput_mbps": 500.0}
        payload = {"latency_ms": 10.0}
        score = _kpi_match_score(nl, payload)
        assert score < 1.0

    def test_empty_nl(self):
        assert _kpi_match_score({}, {"latency_ms": 10.0}) == 0.0


class TestIntentPayloadToText:
    def test_uses_description_and_context(self):
        payload = {"description": "test desc", "context": "test context", "name": "test name"}
        text = intent_payload_to_text(payload)
        assert "test desc" in text
        assert "test context" in text
        assert "test name" in text

    def test_empty_payload(self):
        assert intent_payload_to_text({}) == ""
