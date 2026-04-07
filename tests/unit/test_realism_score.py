"""Tests for realism_score.py."""
from tmf921_dataset_gen.validation.realism_score import score_realism


class TestScoreRealism:
    def test_no_context(self):
        result = score_realism({"name": "test"}, [])
        assert result["score"] == 0.5
        assert "no retrieved context" in result["notes"][0]

    def test_with_context(self):
        context = [
            {
                "text": "telecom network slice with low latency requirements",
                "metadata": {"source_type": "seed"},
            }
        ]
        payload = {"name": "test slice", "context": "low latency telecom"}
        result = score_realism(payload, context)
        assert 0.0 <= result["score"] <= 1.0
        assert result["notes"] == []

    def test_anchor_source_present(self):
        context = [
            {
                "text": "intent management specification",
                "metadata": {"source_type": "oas_example"},
            }
        ]
        payload = {"description": "test"}
        result = score_realism(payload, context)
        assert 0.0 <= result["score"] <= 1.0
