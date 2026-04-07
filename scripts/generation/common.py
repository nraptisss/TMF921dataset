"""Shared utilities for batch generation scripts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_jsonl(target_path: Path, source_path: Path) -> None:
    """Append contents of source_path to target_path, ensuring proper newlines."""
    if not source_path.exists():
        return
    payload = source_path.read_text(encoding="utf-8")
    if not payload:
        return
    if target_path.exists() and target_path.stat().st_size > 0:
        with target_path.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                with target_path.open("a", encoding="utf-8") as writer:
                    writer.write("\n")
    with target_path.open("a", encoding="utf-8") as writer:
        writer.write(payload)
        if not payload.endswith("\n"):
            writer.write("\n")


def build_combined_manifest(output_base: Path, total_generated: int, generation_method: str) -> dict[str, Any]:
    """Aggregate batch manifests into a combined manifest."""
    batch_manifests: list[dict[str, Any]] = []
    quality_scores: list[float] = []
    semantic_scores: list[float] = []
    tio_scores: list[float] = []
    diversity_scores: list[float] = []

    for manifest_path in sorted(output_base.glob("batch_*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        batch_manifests.append({"path": str(manifest_path), "record_count": payload.get("record_count", 0)})
        metrics = payload.get("quality_metrics", {})
        if "average_quality_score" in metrics:
            quality_scores.append(metrics["average_quality_score"])
        if "average_semantic_score" in metrics:
            semantic_scores.append(metrics["average_semantic_score"])
        if "average_tio_compliance" in metrics:
            tio_scores.append(metrics["average_tio_compliance"])
        if "diversity_score" in metrics:
            diversity_scores.append(metrics["diversity_score"])

    def average(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    return {
        "generation_summary": {
            "total_samples": total_generated,
            "generation_method": generation_method,
            "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "batch_count": len(batch_manifests),
        },
        "quality_metrics": {
            "average_quality_score": average(quality_scores),
            "average_semantic_score": average(semantic_scores),
            "average_tio_compliance": average(tio_scores),
            "average_diversity_score": average(diversity_scores),
        },
        "batches": batch_manifests,
    }
