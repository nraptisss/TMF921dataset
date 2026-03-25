from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..models.dataset import DatasetRecord


def build_manifest(settings: Settings, records: list[DatasetRecord], hf_dataset_path: Path | None, jsonl_path: Path) -> dict:
    return {
        "project": "tmf921-dataset-gen",
        "record_count": len(records),
        "hf_dataset_path": str(hf_dataset_path) if hf_dataset_path else None,
        "jsonl_path": str(jsonl_path),
        "embedding_model": settings.embedding_model,
        "reasoning_model": settings.reasoning_model,
        "bulk_model": settings.bulk_model,
        "inference_backend": settings.inference_backend,
    }
