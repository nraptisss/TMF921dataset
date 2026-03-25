from pathlib import Path

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.ingestion.postman_parser import extract_postman_assets


def test_postman_parser_extracts_create_intent_operation() -> None:
    settings = Settings.from_env(Path.cwd())
    corpus = extract_postman_assets(settings)
    assert any(doc["metadata"]["url"] == "{{baseUrl}}/intent?fields=<string>" for doc in corpus)
    operations_path = settings.repo.normalized_dir / "postman" / "operations.json"
    assert operations_path.exists()
