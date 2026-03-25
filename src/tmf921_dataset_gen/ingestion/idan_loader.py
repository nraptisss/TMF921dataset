from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Settings


SUPPORTED_SUFFIXES = {".ttl", ".jsonld", ".json", ".md", ".rdf", ".owl"}


def load_idan_documents(settings: Settings) -> list[dict[str, Any]]:
    normalized_dir = settings.repo.normalized_dir / "idan"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    if not settings.repo.idan_reference_dir.exists():
        (normalized_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "missing",
                    "path": str(settings.repo.idan_reference_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return []

    corpus: list[dict[str, Any]] = []
    for path in sorted(settings.repo.idan_reference_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        relative_path = path.relative_to(settings.repo.idan_reference_dir)
        corpus.append(
            {
                "id": f"idan:{relative_path.as_posix()}",
                "source_type": "idan_reference",
                "title": relative_path.name,
                "text": text,
                "metadata": {"path": str(relative_path)},
            }
        )
    (normalized_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "loaded",
                "document_count": len(corpus),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return corpus
