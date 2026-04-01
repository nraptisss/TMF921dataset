from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    taxonomy_category: str
    kpis: dict[str, Any] = Field(default_factory=dict)
    quality_score: float = 0.0
    tio_compliance: float = 0.0
    seed_id: str | None = None
    generation_timestamp: datetime | None = None
    schema_validity: float = 1.0
    realism_score: float = 0.0
    semantic_score: float = 0.0
    validation_notes: list[str] = Field(default_factory=list)


class DatasetRecord(BaseModel):
    nl_intent: str
    tmf921_intent: dict[str, Any]
    serialization: Literal["json-ld", "turtle"]
    metadata: DatasetMetadata
