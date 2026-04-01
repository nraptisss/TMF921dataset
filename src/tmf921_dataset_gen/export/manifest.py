from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..models.dataset import DatasetRecord
from ..validation.diversity_metrics import calculate_diversity_score, detect_bias


def build_manifest(settings: Settings, records: list[DatasetRecord], hf_dataset_path: Path | None, jsonl_path: Path) -> dict:
    # Calculate quality metrics
    payloads = [record.model_dump(mode="json") for record in records]
    diversity_score = calculate_diversity_score(payloads)
    bias_report = detect_bias(payloads, [record.metadata.taxonomy_target.model_dump() for record in records])

    avg_quality = sum(record.metadata.quality_score for record in records) / len(records) if records else 0
    avg_semantic = sum(record.metadata.semantic_score for record in records) / len(records) if records else 0
    avg_tio = sum(record.metadata.tio_compliance for record in records) / len(records) if records else 0

    return {
        "project": "tmf921-dataset-gen",
        "record_count": len(records),
        "hf_dataset_path": str(hf_dataset_path) if hf_dataset_path else None,
        "jsonl_path": str(jsonl_path),
        "embedding_model": settings.embedding_model,
        "reasoning_model": settings.reasoning_model,
        "bulk_model": settings.bulk_model,
        "inference_backend": settings.inference_backend,
        "quality_metrics": {
            "average_quality_score": round(avg_quality, 4),
            "average_semantic_score": round(avg_semantic, 4),
            "average_tio_compliance": round(avg_tio, 4),
            "diversity_score": round(diversity_score, 4),
            "bias_report": bias_report,
        },
    }
