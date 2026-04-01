from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import Settings

CORE_SCHEMA_NAMES = [
    "Intent",
    "Intent_FVO",
    "IntentExpression",
    "IntentExpression_FVO",
    "JsonLdExpression",
    "JsonLdExpression_FVO",
    "TurtleExpression",
    "TurtleExpression_FVO",
    "RelatedPartyRefOrPartyRoleRef",
    "AttachmentRefOrValue",
    "Hub",
    "IntentCreateEvent",
    "IntentStatusChangeEvent",
    "IntentSpecification",
    "IntentSpecification_FVO",
    "IntentReport",
]


def load_oas_document(settings: Settings) -> dict[str, Any]:
    return yaml.safe_load(settings.repo.oas_spec.read_text(encoding="utf-8"))


def build_root_schema(oas_document: dict[str, Any], schema_name: str) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": f"#/components/schemas/{schema_name}",
        "components": {
            "schemas": oas_document.get("components", {}).get("schemas", {}),
        },
    }


def get_example_values(oas_document: dict[str, Any]) -> dict[str, Any]:
    raw_examples = oas_document.get("components", {}).get("examples", {})
    return {name: example.get("value") for name, example in raw_examples.items()}


def materialize_oas_artifacts(settings: Settings, oas_document: dict[str, Any]) -> None:
    schemas_dir = settings.repo.artifacts_dir / "schemas"
    normalized_dir = settings.repo.normalized_dir / "oas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    components = oas_document.get("components", {}).get("schemas", {})
    (schemas_dir / "tmf921_components.json").write_text(
        __import__("json").dumps(components, indent=2),
        encoding="utf-8",
    )

    for schema_name in CORE_SCHEMA_NAMES:
        if schema_name in components:
            (schemas_dir / f"{schema_name}.schema.json").write_text(
                __import__("json").dumps(build_root_schema(oas_document, schema_name), indent=2),
                encoding="utf-8",
            )

    examples_payload = get_example_values(oas_document)
    (normalized_dir / "examples.json").write_text(
        __import__("json").dumps(examples_payload, indent=2),
        encoding="utf-8",
    )


def extract_oas_assets(settings: Settings) -> list[dict[str, Any]]:
    oas_document = load_oas_document(settings)
    materialize_oas_artifacts(settings, oas_document)
    corpus: list[dict[str, Any]] = []
    schemas = oas_document.get("components", {}).get("schemas", {})
    for schema_name in CORE_SCHEMA_NAMES:
        schema = schemas.get(schema_name)
        if schema is None:
            continue
        corpus.append(
            {
                "id": f"oas-schema:{schema_name}",
                "source_type": "oas_schema",
                "title": schema_name,
                "text": __import__("json").dumps(schema, indent=2),
                "metadata": {"schema_name": schema_name},
            }
        )
    for example_name, value in get_example_values(oas_document).items():
        corpus.append(
            {
                "id": f"oas-example:{example_name}",
                "source_type": "oas_example",
                "title": example_name,
                "text": __import__("json").dumps(value, indent=2),
                "metadata": {"example_name": example_name},
            }
        )
    return corpus
