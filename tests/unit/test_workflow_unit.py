from pathlib import Path

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.graph.workflow import run_generation


def test_run_generation_returns_requested_count(tmp_path: Path) -> None:
    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "cli-index"
    records = run_generation(settings, count=2, output_dir=None)
    assert len(records) == 2
