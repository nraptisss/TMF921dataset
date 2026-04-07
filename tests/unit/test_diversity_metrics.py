"""Tests for diversity_metrics.py."""
from tmf921_dataset_gen.validation.diversity_metrics import calculate_diversity_score, detect_bias


class TestCalculateDiversityScore:
    def test_empty_records(self):
        assert calculate_diversity_score([]) == 0.0

    def test_all_unique(self):
        records = [
            {"nl_intent": "intent one", "tmf921_intent": {"name": "a"}},
            {"nl_intent": "intent two", "tmf921_intent": {"name": "b"}},
        ]
        assert calculate_diversity_score(records) == 1.0

    def test_all_duplicates(self):
        records = [
            {"nl_intent": "same", "tmf921_intent": {"name": "same"}},
            {"nl_intent": "same", "tmf921_intent": {"name": "same"}},
        ]
        assert calculate_diversity_score(records) == 0.5

    def test_mixed(self):
        records = [
            {"nl_intent": "a", "tmf921_intent": {"name": "x"}},
            {"nl_intent": "a", "tmf921_intent": {"name": "x"}},
            {"nl_intent": "b", "tmf921_intent": {"name": "y"}},
        ]
        score = calculate_diversity_score(records)
        assert 0.0 < score < 1.0


class TestDetectBias:
    def test_empty_records(self):
        result = detect_bias([], [])
        assert result["bias_score"] == 0.0
        assert result["notes"] == []

    def test_balanced_distribution(self):
        records = [
            {"metadata": {"taxonomy_target": {"layer": "service", "traffic_profile": "embb"}}},
            {"metadata": {"taxonomy_target": {"layer": "service", "traffic_profile": "urllc"}}},
            {"metadata": {"taxonomy_target": {"layer": "resource", "traffic_profile": "embb"}}},
            {"metadata": {"taxonomy_target": {"layer": "resource", "traffic_profile": "urllc"}}},
        ]
        result = detect_bias(records, [])
        assert result["bias_score"] == 0.0
        assert result["notes"] == []

    def test_underrepresented_layer(self):
        records = []
        for _ in range(20):
            records.append({"metadata": {"taxonomy_target": {"layer": "service", "traffic_profile": "embb"}}})
        records.append({"metadata": {"taxonomy_target": {"layer": "business", "traffic_profile": "embb"}}})
        result = detect_bias(records, [])
        assert result["bias_score"] > 0.0
        assert any("business" in note for note in result["notes"])

    def test_falls_back_to_taxonomy_category(self):
        records = [
            {"metadata": {"taxonomy_category": "service/embb/energy"}},
            {"metadata": {"taxonomy_category": "resource/urllc/reporting"}},
        ]
        result = detect_bias(records, [])
        assert "bias_score" in result
