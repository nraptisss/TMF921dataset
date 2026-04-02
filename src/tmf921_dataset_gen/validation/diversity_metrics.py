from __future__ import annotations

import json
from typing import Any


def calculate_diversity_score(records: list[dict[str, Any]]) -> float:
    """Calculate diversity score based on unique intents and payloads."""
    if not records:
        return 0.0

    nl_intents = set()
    payload_hashes = set()

    for record in records:
        nl_intents.add(record.get("nl_intent", ""))
        payload_str = json.dumps(record.get("tmf921_intent", {}), sort_keys=True)
        payload_hashes.add(payload_str)

    # Diversity as ratio of unique to total
    nl_diversity = len(nl_intents) / len(records)
    payload_diversity = len(payload_hashes) / len(records)

    return (nl_diversity + payload_diversity) / 2


def _coerce_taxonomy_target(record: dict[str, Any], taxonomy_targets: list[dict[str, Any]], index: int) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    target = metadata.get("taxonomy_target")
    if isinstance(target, dict) and target:
        return target
    if index < len(taxonomy_targets) and isinstance(taxonomy_targets[index], dict):
        return taxonomy_targets[index]
    category = metadata.get("taxonomy_category", "")
    parts = category.split("/")
    if len(parts) >= 3:
        return {
            "layer": parts[0],
            "traffic_profile": parts[1],
            "scenario_family": parts[2],
        }
    return {}


def detect_bias(records: list[dict[str, Any]], taxonomy_targets: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect bias in layer/traffic profile distribution."""
    layer_counts = {}
    traffic_counts = {}

    if not records:
        return {"bias_score": 0.0, "notes": []}

    for index, record in enumerate(records):
        target = _coerce_taxonomy_target(record, taxonomy_targets, index)
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
