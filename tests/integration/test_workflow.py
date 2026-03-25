from pathlib import Path

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.graph.workflow import run_generation


def test_run_generation_produces_exportable_records(tmp_path: Path) -> None:
    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "workflow-index"
    settings.repo.output_dir = tmp_path / "output"
    records = run_generation(settings, count=3, output_dir=tmp_path / "dataset")
    assert len(records) == 3
    assert (tmp_path / "dataset" / "dataset.jsonl").exists()
    assert (tmp_path / "dataset" / "manifest.json").exists()
