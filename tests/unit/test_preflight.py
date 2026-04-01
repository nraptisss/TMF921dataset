from pathlib import Path

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.preflight import run_preflight


def test_preflight_detects_required_assets() -> None:
    settings = Settings.from_env(Path.cwd())
    report = run_preflight(settings)
    assert report.ok is True
    assert report.errors == []


def test_preflight_warns_when_idan_reference_missing() -> None:
    settings = Settings.from_env(Path.cwd())
    report = run_preflight(settings)
    assert any("idan-reference" in warning for warning in report.warnings)
