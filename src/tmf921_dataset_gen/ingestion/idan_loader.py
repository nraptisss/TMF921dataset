from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Settings


SUPPORTED_SUFFIXES = {".ttl", ".jsonld", ".json", ".md", ".rdf", ".owl"}
ALLOWED_TOP_LEVEL = {"ontologies", "api", "serviceorders", "utils"}
EXCLUDED_FILENAMES = {"readme.md", "package.json", "package-lock.json", "kgconfig.json"}

# Keywords indicating intent-related content
INTENT_KEYWORDS = {"intent", "fvo", "expression", "deliveryexpectation", "constraint", "kpi", "slice", "network", "service"}


def _is_intent_related(text: str) -> bool:
    """Check if text contains intent-related keywords (case-insensitive)."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in INTENT_KEYWORDS)


def _is_allowed_idan_path(relative_path: Path) -> bool:
    if not relative_path.parts:
        return False
    if any(part.startswith(".") for part in relative_path.parts):
        return False
    if relative_path.name.lower() in EXCLUDED_FILENAMES:
        return False
    return relative_path.parts[0] in ALLOWED_TOP_LEVEL


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
        relative_path = path.relative_to(settings.repo.idan_reference_dir)
        if not _is_allowed_idan_path(relative_path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Filter for intent-related content
        if not _is_intent_related(text):
            continue
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
