from pathlib import Path

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.ingestion.oas_parser import extract_oas_assets, get_example_values, load_oas_document
from tmf921_dataset_gen.ingestion.postman_parser import extract_postman_assets
from tmf921_dataset_gen.models.generated.tmf921_models import (
    Attachment,
    Hub,
    IntentCreateEvent,
    IntentFVO,
    IntentStatusChangeEvent,
)
from tmf921_dataset_gen.validation.jsonschema_validator import TMFJsonSchemaValidator


def test_oas_assets_extract_core_schemas_and_examples() -> None:
    settings = Settings.from_env(Path.cwd())
    corpus = extract_oas_assets(settings)
    assert any(doc["id"] == "oas-schema:Intent_FVO" for doc in corpus)
    assert any(doc["id"] == "oas-example:Intent_create_example_01_request" for doc in corpus)
    assert (settings.repo.artifacts_dir / "schemas" / "Intent_FVO.schema.json").exists()
    assert (settings.repo.normalized_dir / "oas" / "examples.json").exists()


def test_generated_models_and_schema_validator_accept_official_intent_examples() -> None:
    settings = Settings.from_env(Path.cwd())
    examples = get_example_values(load_oas_document(settings))
    validator = TMFJsonSchemaValidator(settings, "Intent_FVO")

    for key in ["Intent_create_example_01_request", "Intent_create_example_11_request"]:
        payload = examples[key]
        IntentFVO.model_validate(payload)
        result = validator.validate(payload)
        assert result.valid, result.errors


def test_event_models_accept_official_event_examples() -> None:
    settings = Settings.from_env(Path.cwd())
    examples = get_example_values(load_oas_document(settings))
    IntentCreateEvent.model_validate(examples["IntentCreateEvent_request"])
    IntentStatusChangeEvent.model_validate(examples["IntentStatusChangeEvent_request"])


def test_postman_hub_payload_and_attachment_model_validate() -> None:
    settings = Settings.from_env(Path.cwd())
    extract_postman_assets(settings)
    operations_path = settings.repo.normalized_dir / "postman" / "operations.json"
    operations = __import__("json").loads(operations_path.read_text(encoding="utf-8"))
    hub_operation = next(op for op in operations if op["method"] == "POST" and op["url"] == "{{baseUrl}}/hub")
    hub_payload = __import__("json").loads(hub_operation["body_raw"])
    Hub.model_validate(hub_payload)
    Attachment.model_validate({"@type": "Attachment", "id": "att-1", "href": "https://example.com/a", "name": "reference"})
