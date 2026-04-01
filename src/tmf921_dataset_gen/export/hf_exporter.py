from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings
from ..models.dataset import DatasetRecord
from .manifest import build_manifest


@dataclass(slots=True)
class ExportReport:
    output_dir: Path
    record_count: int
    hf_dataset_path: Path | None
    jsonl_path: Path
    manifest_path: Path



def export_dataset_records(settings: Settings, records: list[DatasetRecord], output_dir: Path) -> ExportReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [record.model_dump(mode="json") for record in records]
    jsonl_path = output_dir / "dataset.jsonl"
    jsonl_path.write_text("\n".join(__import__("json").dumps(row) for row in payload), encoding="utf-8")

    hf_dataset_path: Path | None = None
    try:
        from datasets import Dataset

        hf_dataset_path = output_dir / "hf_dataset"
        dataset = Dataset.from_list(payload)
        dataset.save_to_disk(str(hf_dataset_path))
    except Exception:
        hf_dataset_path = None

    manifest = build_manifest(settings, records, hf_dataset_path, jsonl_path)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    return ExportReport(
        output_dir=output_dir,
        record_count=len(records),
        hf_dataset_path=hf_dataset_path,
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
    )
