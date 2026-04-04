from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    taxonomy_category: str
    taxonomy_target: dict[str, Any] = Field(default_factory=dict)
    kpis: dict[str, Any] = Field(default_factory=dict)
    quality_score: float = 0.0
    tio_compliance: float = 0.0
    seed_id: str | None = None
    generation_timestamp: datetime | None = None
    schema_validity: float = 1.0
    realism_score: float = 0.0
    semantic_score: float = 0.0
    validation_notes: list[str] = Field(default_factory=list)
    retrieved_context_ids: list[str] = Field(default_factory=list)
    translation_backend: str | None = None
    generation_backend: str | None = None
    intent_frame: dict[str, Any] = Field(default_factory=dict)
    constraint_set: list[dict[str, Any]] = Field(default_factory=list)
    constraint_alignment: dict[str, Any] = Field(default_factory=dict)
    evidence_map: dict[str, list[str]] = Field(default_factory=dict)
    semantic_pass: bool = False
    operator_pass: bool = False
    constraint_coverage: float = 0.0
    unsupported_claim_count: int = 0
    contradiction_count: int = 0
    grounding_mode: str = "synthetic_semantic"
    grounding_pass: bool = True
    supported_claim_ratio: float = 0.0


class DatasetRecord(BaseModel):
    nl_intent: str
    tmf921_intent: dict[str, Any]
    serialization: Literal["json-ld", "turtle"]
    metadata: DatasetMetadata
