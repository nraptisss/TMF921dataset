from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

from .config import Settings


@dataclass(slots=True)
class PreflightReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_preflight(settings: Settings) -> PreflightReport:
    repo = settings.repo
    report = PreflightReport()
    required_files = {
        "Postman collection": repo.postman_collection,
        "TMF921 OAS": repo.oas_spec,
        "TR290 PDF": repo.tr290_pdf,
        "TR290 DOCX": repo.tr290_docx,
        "Seeds": repo.seeds_path,
    }
    for label, path in required_files.items():
        if not path.exists():
            report.errors.append(f"Missing required resource: {label} at {path}")

    if not repo.idan_reference_dir.exists():
        report.warnings.append(
            "Optional but strongly recommended resource missing: idan-reference/. "
            "Large sample generation is blocked until it is available."
        )

    if settings.is_local_model_backend:
        if not _module_available("torch"):
            report.errors.append("Local transformers backend requires the 'torch' package.")
        if not _module_available("transformers"):
            report.errors.append("Local transformers backend requires the 'transformers' package.")
        if settings.local_reasoning_model_path and not settings.local_reasoning_model_path.exists():
            report.errors.append(f"Missing local reasoning model path: {settings.local_reasoning_model_path}")
        if settings.local_bulk_model_path and not settings.local_bulk_model_path.exists():
            report.errors.append(f"Missing local bulk model path: {settings.local_bulk_model_path}")
        if settings.local_embedding_model_path and not settings.local_embedding_model_path.exists():
            report.errors.append(f"Missing local embedding model path: {settings.local_embedding_model_path}")
        if settings.local_use_4bit and not _module_available("bitsandbytes"):
            report.warnings.append("LOCAL_USE_4BIT=true but bitsandbytes is not installed; 4-bit loading will not work.")
        if settings.local_device.startswith("cuda") and _module_available("torch"):
            import torch

            if not torch.cuda.is_available():
                report.warnings.append(
                    f"LOCAL_DEVICE is set to {settings.local_device}, but CUDA is not currently available on this machine."
                )
        if settings.local_files_only and not settings.local_reasoning_model_path:
            report.warnings.append(
                "LOCAL_FILES_ONLY=true and LOCAL_REASONING_MODEL_PATH is unset. The reasoning model must already exist in the local Hugging Face cache."
            )
        if settings.local_files_only and not settings.local_bulk_model_path:
            report.warnings.append(
                "LOCAL_FILES_ONLY=true and LOCAL_BULK_MODEL_PATH is unset. The bulk model must already exist in the local Hugging Face cache."
            )

    for path in (
        repo.artifacts_dir,
        repo.normalized_dir,
        repo.indexes_dir,
        repo.reports_dir,
        repo.output_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return report
