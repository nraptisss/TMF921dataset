from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..models.dataset import DatasetRecord
from ..validation.diversity_metrics import calculate_diversity_score, detect_bias


def build_manifest(settings: Settings, records: list[DatasetRecord], hf_dataset_path: Path | None, jsonl_path: Path) -> dict:
    # Calculate quality metrics
    payloads = [record.model_dump(mode="json") for record in records]
    diversity_score = calculate_diversity_score(payloads)
    bias_report = detect_bias(payloads, [record.metadata.taxonomy_target for record in records])

    avg_quality = sum(record.metadata.quality_score for record in records) / len(records) if records else 0
    avg_semantic = sum(record.metadata.semantic_score for record in records) / len(records) if records else 0
    avg_tio = sum(record.metadata.tio_compliance for record in records) / len(records) if records else 0

    effective_embedding_model = settings.effective_embedding_model or settings.embedding_model
    effective_embedding_backend = settings.effective_embedding_backend
    if effective_embedding_backend is None:
        effective_embedding_backend = "not-used" if settings.inference_backend == "mock" else "unknown"

    return {
        "project": "tmf921-dataset-gen",
        "record_count": len(records),
        "hf_dataset_path": str(hf_dataset_path) if hf_dataset_path else None,
        "jsonl_path": str(jsonl_path),
        "embedding_model": effective_embedding_model,
        "configured_embedding_model": settings.embedding_model,
        "effective_embedding_backend": effective_embedding_backend,
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
