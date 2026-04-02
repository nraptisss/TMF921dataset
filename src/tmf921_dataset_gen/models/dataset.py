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


class DatasetRecord(BaseModel):
    nl_intent: str
    tmf921_intent: dict[str, Any]
    serialization: Literal["json-ld", "turtle"]
    metadata: DatasetMetadata
