from __future__ import annotations

from pathlib import Path

from ..config import Settings


def run_generation(settings: Settings, count: int, output_dir: Path | None) -> list[dict]:
    del settings, count, output_dir
    return []
