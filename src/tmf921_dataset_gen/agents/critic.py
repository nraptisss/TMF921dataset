from __future__ import annotations

from typing import Any

from ..config import Settings
from ..rag.embeddings import HashingEmbeddingModel
from ..validation.jsonschema_validator import TMFJsonSchemaValidator
from ..validation.realism_score import score_realism
from ..validation.semantic_score import score_semantic_faithfulness
from ..validation.tio_rules import evaluate_tio_compliance
from .translator import TranslatorAgent


class CriticRefinementAgent:
    def __init__(self, settings: Settings, translator: TranslatorAgent | None = None) -> None:
        self.settings = settings
        self.translator = translator or TranslatorAgent(settings)
        self.validator = TMFJsonSchemaValidator(settings, "Intent_FVO")
        self.embedder = HashingEmbeddingModel()

    def review(
        self,
        nl_intent: str,
        payload: dict[str, Any],
        serialization: str,
        taxonomy_target: dict[str, Any],
        retrieved_context: list[dict[str, Any]],
        refinement_count: int = 0,
    ) -> dict[str, Any]:
        schema_result = self.validator.validate(payload)
        semantic = score_semantic_faithfulness(nl_intent, payload, self.embedder)
        tio = evaluate_tio_compliance(payload)
        realism = score_realism(payload, retrieved_context)
        quality_score = (0.4 * semantic["score"]) + (0.35 * tio["score"]) + (0.25 * realism["score"])
        accepted = (
            schema_result.valid
            and semantic["score"] >= 0.85
            and tio["score"] >= 0.85
            and realism["score"] >= 0.85
            and quality_score >= 0.90
        )
        repaired_payload = payload
        notes = [*schema_result.errors, *tio["notes"], *realism["notes"]]
        if not accepted and refinement_count < self.settings.max_refinement_loops:
            repaired_payload = self.translator.repair_payload(payload, taxonomy_target, serialization)
            notes.append("applied heuristic repair")
        return {
            "accepted": accepted,
            "schema_validity": 1.0 if schema_result.valid else 0.0,
            "semantic": semantic,
            "tio": tio,
            "realism": realism,
            "quality_score": round(quality_score, 4),
            "notes": notes,
            "repaired_payload": repaired_payload,
        }
