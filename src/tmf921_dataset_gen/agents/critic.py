from __future__ import annotations

from typing import Any

from ..config import Settings
from ..llm import LLMRouter
from ..rag.embeddings import HashingEmbeddingModel, build_embedding_backend
from ..validation.jsonschema_validator import TMFJsonSchemaValidator
from ..validation.realism_score import score_realism
from ..validation.semantic_score import intent_payload_to_text, score_semantic_faithfulness
from ..validation.tio_rules import evaluate_tio_compliance
from .translator import TranslatorAgent


class CriticRefinementAgent:
    def __init__(self, settings: Settings, translator: TranslatorAgent | None = None) -> None:
        self.settings = settings
        self.translator = translator or TranslatorAgent(settings)
        self.validator = TMFJsonSchemaValidator(settings, "Intent_FVO")
        self.embedder = self._build_semantic_embedder()
        self.router = LLMRouter(settings)

    def _build_semantic_embedder(self):
        if self.settings.effective_embedding_backend == "sentence-transformers":
            model_name = self.settings.effective_embedding_model or self.settings.resolve_embedding_model()
            return build_embedding_backend(
                model_name,
                allow_fallback=True,
                prefer_hashing=False,
                local_files_only=self.settings.is_local_model_backend and self.settings.local_files_only,
            )
        if self.settings.effective_embedding_backend == "hashing-vectorizer":
            return HashingEmbeddingModel()
        return HashingEmbeddingModel()

    def _llm_semantic_review(self, nl_intent: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.enable_llm_semantic_review or not self.router.supports_generation():
            return {}
        prompt = f"""
You are judging whether a TMF921 intent payload is faithful to a natural-language telecom request.
Return JSON only with this shape:
{{"faithfulness": 0.0, "notes": ["..."]}}
Use a score from 0.0 to 1.0.

Natural-language intent:
{nl_intent}

Payload summary:
{intent_payload_to_text(payload)}
""".strip()
        try:
            response = self.router.generate_json(
                prompt,
                model=self.settings.reasoning_model,
                temperature=0.0,
                max_new_tokens=self.settings.local_planning_max_new_tokens,
            )
            if isinstance(response, dict) and "faithfulness" in response:
                return response
        except Exception:
            pass
        return {}

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
        llm_semantic = self._llm_semantic_review(nl_intent, payload)
        if llm_semantic and "faithfulness" in llm_semantic:
            semantic["llm_judge"] = float(llm_semantic["faithfulness"])
            semantic["score"] = max(semantic["score"], (0.6 * semantic["llm_judge"]) + (0.4 * semantic["cosine"]))
        tio = evaluate_tio_compliance(payload)
        realism = score_realism(payload, retrieved_context)
        quality_score = (0.4 * semantic["score"]) + (0.35 * tio["score"]) + (0.25 * realism["score"])
        semantic_threshold = 0.50
        tio_threshold = 0.50
        realism_threshold = 0.30
        quality_threshold = 0.50
        if self.settings.inference_backend == "mock":
            semantic_threshold = 0.30
            tio_threshold = 0.40
            realism_threshold = 0.0
            quality_threshold = 0.40
        accepted = (
            schema_result.valid
            and semantic["score"] >= semantic_threshold
            and tio["score"] >= tio_threshold
            and realism["score"] >= realism_threshold
            and quality_score >= quality_threshold
        )
        repaired_payload = payload
        notes = [*schema_result.errors, *tio["notes"], *realism["notes"]]
        if llm_semantic.get("notes"):
            notes.extend(llm_semantic["notes"])
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
