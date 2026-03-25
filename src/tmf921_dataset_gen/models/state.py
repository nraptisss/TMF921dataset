from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    taxonomy_target: dict[str, Any]
    retrieved_context: list[dict[str, Any]]
    seed_ids: list[str]
    nl_intent: str
    tmf921_intent: dict[str, Any]
    serialization: str
    metadata: dict[str, Any]
    critic_report: dict[str, Any]
    refinement_count: int
    accepted: bool
    generation_timestamp: datetime
