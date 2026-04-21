#!/usr/bin/env python3
"""Prepare an independent manual review packet for a TMF921 dataset release.

This script does not auto-approve a dataset. It selects a stratified review
sample and writes pending review items for human judgment.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


RUBRIC = [
    "semantic faithfulness",
    "tmf/tio correctness",
    "grounding support",
    "linguistic naturalness",
]


def _load_records(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _bucket_name(record: dict) -> str:
    metadata = record.get("metadata", {})
    if not metadata.get("semantic_pass", False):
        return "semantic_failures"
    if not metadata.get("operator_pass", False):
        return "operator_failures"
    if metadata.get("unsupported_claim_count", 0) > 0:
        return "grounding_failures"
    return "accepted_passes"


def _select_review_sample(records: list[dict], sample_size: int) -> list[tuple[int, dict]]:
    indexed = list(enumerate(records))
    buckets: dict[str, list[tuple[int, dict]]] = {
        "semantic_failures": [],
        "operator_failures": [],
        "grounding_failures": [],
        "accepted_passes": [],
    }
    for item in indexed:
        buckets[_bucket_name(item[1])].append(item)

    selected: list[tuple[int, dict]] = []
    seen_indices: set[int] = set()
    rng = random.Random(42)

    bucket_targets = {
        "semantic_failures": max(3, sample_size // 6),
        "operator_failures": max(2, sample_size // 8),
        "grounding_failures": max(4, sample_size // 5),
        "accepted_passes": max(6, sample_size // 3),
    }

    for bucket_name, target_count in bucket_targets.items():
        candidates = list(buckets[bucket_name])
        rng.shuffle(candidates)
        for index, record in candidates:
            if index in seen_indices:
                continue
            selected.append((index, record))
            seen_indices.add(index)
            if len([1 for chosen_index, chosen_record in selected if _bucket_name(chosen_record) == bucket_name]) >= target_count:
                break

    remaining = [item for item in indexed if item[0] not in seen_indices]
    rng.shuffle(remaining)
    for item in remaining:
        if len(selected) >= sample_size:
            break
        selected.append(item)
        seen_indices.add(item[0])

    return selected[:sample_size]


def _review_item(index: int, record: dict) -> dict:
    metadata = record.get("metadata", {})
    taxonomy = metadata.get("taxonomy_target", {})
    return {
        "record_index": index,
        "bucket": _bucket_name(record),
        "layer": taxonomy.get("layer"),
        "traffic_profile": taxonomy.get("traffic_profile"),
        "scenario_family": taxonomy.get("scenario_family"),
        "domain_context": taxonomy.get("domain_context"),
        "serialization": record.get("serialization"),
        "nl_intent": record.get("nl_intent"),
        "tmf921_intent": record.get("tmf921_intent"),
        "metadata_snapshot": {
            "semantic_pass": metadata.get("semantic_pass"),
            "operator_pass": metadata.get("operator_pass"),
            "unsupported_claim_count": metadata.get("unsupported_claim_count"),
            "supported_claim_ratio": metadata.get("supported_claim_ratio"),
            "validation_notes": metadata.get("validation_notes", []),
        },
        "human_review": {
            criterion: {
                "status": "pending",
                "comment": "",
            }
            for criterion in RUBRIC
        },
        "overall_decision": "pending",
    }


def main() -> int:
    dataset_path = Path("thousand_records_dataset/dataset.jsonl")
    output_path = Path("thousand_records_dataset/manual_review_results.json")
    records = _load_records(dataset_path)
    sample_size = min(50, max(12, len(records) // 20))
    sample = _select_review_sample(records, sample_size)

    results = {
        "sample_size": len(sample),
        "total_records": len(records),
        "status": "pending_human_review",
        "instructions": {
            "summary": "Review every item independently. Do not trust dataset metadata as ground truth.",
            "rubric": RUBRIC,
            "pass_rule": "Dataset remains blocked until every sampled item has a human decision and release approver sign-off.",
        },
        "evaluations": [_review_item(index, record) for index, record in sample],
    }

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared pending manual review packet with {len(sample)} items at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
