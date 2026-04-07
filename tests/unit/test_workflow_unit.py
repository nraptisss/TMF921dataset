from pathlib import Path

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.graph.workflow import run_generation


def test_run_generation_returns_requested_count(tmp_path: Path) -> None:
    settings = Settings.from_env(Path.cwd())
    # Use mock backend for fast unit testing
    settings.inference_backend = "mock"
    settings.fast_mode = True
    settings.enable_llm_rewrite = False
    settings.enable_llm_translation_hints = False
    settings.enable_llm_semantic_review = False
    settings.grounding_mode = "synthetic_semantic"
    settings.vector_index_dir = tmp_path / "cli-index"
    records = run_generation(settings, count=2, output_dir=None)
    assert len(records) == 2
    assert all("intent_frame" in record["metadata"] for record in records)
    assert all("constraint_alignment" in record["metadata"] for record in records)
