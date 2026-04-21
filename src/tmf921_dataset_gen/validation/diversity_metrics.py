from __future__ import annotations

import json
import re
from typing import Any


def calculate_diversity_score(records: list[dict[str, Any]]) -> float:
    """Calculate diversity score across text, payload, and taxonomy coverage."""
    if not records:
        return 0.0

    nl_intents = set()
    payload_hashes = set()
    normalized_nl = set()
    scenario_families = set()
    domain_contexts = set()
    traffic_profiles = set()
    has_taxonomy = False

    for record in records:
        nl_intent = record.get("nl_intent", "")
        nl_intents.add(nl_intent)
        normalized_nl.add(re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", nl_intent.lower())).strip())
        payload_str = json.dumps(record.get("tmf921_intent", {}), sort_keys=True)
        payload_hashes.add(payload_str)
        target = _coerce_taxonomy_target(record, [], 0)
        if target:
            has_taxonomy = True
            scenario_families.add(target.get("scenario_family", ""))
            domain_contexts.add(target.get("domain_context", ""))
            traffic_profiles.add(target.get("traffic_profile", ""))

    nl_diversity = len(nl_intents) / len(records)
    normalized_nl_diversity = len(normalized_nl) / len(records)
    payload_diversity = len(payload_hashes) / len(records)
    if not has_taxonomy:
        return round((nl_diversity + payload_diversity) / 2, 4)
    scenario_coverage = len({value for value in scenario_families if value}) / 8
    context_coverage = len({value for value in domain_contexts if value}) / 4
    profile_coverage = len({value for value in traffic_profiles if value}) / 3

    return round(
        (
            (0.2 * nl_diversity)
            + (0.2 * normalized_nl_diversity)
            + (0.2 * payload_diversity)
            + (0.2 * scenario_coverage)
            + (0.1 * context_coverage)
            + (0.1 * profile_coverage)
        ),
        4,
    )


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
    """Detect bias in layer, traffic, scenario, and context distribution."""
    layer_counts = {}
    traffic_counts = {}
    scenario_counts = {}
    context_counts = {}

    if not records:
        return {"bias_score": 0.0, "notes": []}

    for index, record in enumerate(records):
        target = _coerce_taxonomy_target(record, taxonomy_targets, index)
        layer = target.get("layer", "unknown")
        traffic = target.get("traffic_profile", "unknown")
        scenario = target.get("scenario_family", "unknown")
        context = target.get("domain_context", "unknown")

        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        traffic_counts[traffic] = traffic_counts.get(traffic, 0) + 1
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        context_counts[context] = context_counts.get(context, 0) + 1

    total = len(records)
    bias_score = 0.0
    notes = []

    def flag_imbalance(label: str, counts: dict[str, int], minimum_ratio: float, score_increment: float) -> None:
        nonlocal bias_score
        for key, count in counts.items():
            if count / total < minimum_ratio:
                bias_score += score_increment
                notes.append(f"underrepresented {label}: {key}")

    for layer, count in layer_counts.items():
        if count / total < 0.15:
            bias_score += 0.1
            notes.append(f"underrepresented layer: {layer}")

    flag_imbalance("traffic profile", traffic_counts, minimum_ratio=0.15, score_increment=0.15)
    flag_imbalance("scenario family", scenario_counts, minimum_ratio=0.08, score_increment=0.08)
    flag_imbalance("domain context", context_counts, minimum_ratio=0.15, score_increment=0.05)

    return {"bias_score": min(bias_score, 1.0), "notes": notes}
