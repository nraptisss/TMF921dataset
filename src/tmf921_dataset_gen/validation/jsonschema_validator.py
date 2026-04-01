from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from ..config import Settings
from ..ingestion.oas_parser import load_oas_document


def _ref_to_schema_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _apply_discriminator_constraint(schema: dict[str, Any], property_name: str, allowed_values: list[str]) -> dict[str, Any]:
    constraint = {
        "type": "object",
        "properties": {
            property_name: {"enum": allowed_values},
        },
        "required": [property_name],
    }
    existing_all_of = schema.get("allOf")
    if existing_all_of:
        filtered_all_of = [item for item in existing_all_of if not (isinstance(item, dict) and item.get("x-codex-discriminator-constraint"))]
        schema["allOf"] = [*filtered_all_of, {**constraint, "x-codex-discriminator-constraint": True}]
        return schema
    if schema.get("x-codex-discriminator-wrapper"):
        schema["allOf"][-1] = {**constraint, "x-codex-discriminator-constraint": True}
        return schema
    return {
        "allOf": [schema, {**constraint, "x-codex-discriminator-constraint": True}],
        "x-codex-discriminator-wrapper": True,
    }


def _augment_discriminators(components: dict[str, Any]) -> dict[str, Any]:
    augmented = deepcopy(components)
    target_tokens: dict[tuple[str, str], set[str]] = {}

    for schema_name, schema in components.items():
        discriminator = schema.get("discriminator")
        if not discriminator:
            continue
        property_name = discriminator["propertyName"]
        mapping = discriminator.get("mapping", {})
        target_tokens.setdefault((schema_name, property_name), set()).update(mapping.keys())
        for token, ref in mapping.items():
            target_name = _ref_to_schema_name(ref)
            target_tokens.setdefault((target_name, property_name), set()).add(token)

    for (target_name, property_name), allowed_values in target_tokens.items():
        target_schema = augmented.get(target_name)
        if target_schema is None:
            continue
        augmented[target_name] = _apply_discriminator_constraint(target_schema, property_name, sorted(allowed_values))

    return augmented


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


class TMFJsonSchemaValidator:
    def __init__(self, settings: Settings, schema_name: str = "Intent_FVO") -> None:
        self.settings = settings
        self.schema_name = schema_name
        self.oas_document = load_oas_document(settings)
        components = self.oas_document.get("components", {}).get("schemas", {})
        self.schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": f"#/components/schemas/{schema_name}",
            "components": {
                "schemas": _augment_discriminators(components),
            },
        }
        self.validator = Draft202012Validator(self.schema, format_checker=FormatChecker())

    def validate(self, payload: dict[str, Any]) -> ValidationResult:
        errors = sorted(self.validator.iter_errors(payload), key=lambda err: list(err.path))
        return ValidationResult(
            valid=not errors,
            errors=[error.message for error in errors],
        )
