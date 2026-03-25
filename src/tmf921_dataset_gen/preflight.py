from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings


@dataclass(slots=True)
class PreflightReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


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

    for path in (
        repo.artifacts_dir,
        repo.normalized_dir,
        repo.indexes_dir,
        repo.reports_dir,
        repo.output_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return report
