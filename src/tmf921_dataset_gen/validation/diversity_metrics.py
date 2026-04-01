from __future__ import annotations

from typing import Any


def calculate_diversity_score(records: list[dict[str, Any]]) -> float:
    """Calculate diversity score based on unique intents and payloads."""
    if not records:
        return 0.0

    nl_intents = set()
    payload_hashes = set()

    for record in records:
        nl_intents.add(record.get("nl_intent", ""))
        # Simple hash of payload structure
        payload_str = str(record.get("tmf921_intent", {}))
        payload_hashes.add(hash(payload_str))

    # Diversity as ratio of unique to total
    nl_diversity = len(nl_intents) / len(records)
    payload_diversity = len(payload_hashes) / len(records)

    return (nl_diversity + payload_diversity) / 2


def detect_bias(records: list[dict[str, Any]], taxonomy_targets: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect bias in layer/traffic profile distribution."""
    layer_counts = {}
    traffic_counts = {}

    for record in records:
        target = record.get("metadata", {}).get("taxonomy_target", {})
        layer = target.get("layer", "unknown")
        traffic = target.get("traffic_profile", "unknown")

        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        traffic_counts[traffic] = traffic_counts.get(traffic, 0) + 1

    total = len(records)
    bias_score = 0.0
    notes = []

    # Check for imbalance (simple threshold)
    for layer, count in layer_counts.items():
        if count / total < 0.1:  # Less than 10%
            bias_score += 0.1
            notes.append(f"underrepresented layer: {layer}")

    for traffic, count in traffic_counts.items():
        if count / total < 0.1:
            bias_score += 0.1
            notes.append(f"underrepresented traffic profile: {traffic}")

    return {"bias_score": min(bias_score, 1.0), "notes": notes}