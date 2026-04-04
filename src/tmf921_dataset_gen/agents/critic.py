from __future__ import annotations

from typing import Any, cast

from ..config import Settings
from ..llm import LLMRouter
from ..rag.embeddings import HashingEmbeddingModel, build_embedding_backend
from ..validation.jsonschema_validator import TMFJsonSchemaValidator
from ..validation.realism_score import score_realism
from ..validation.semantic_score import intent_payload_to_text, score_semantic_faithfulness
from ..validation.semantic_frame import attribute_evidence, verify_semantic_alignment
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
        intent_frame: dict[str, Any] | None = None,
        refinement_count: int = 0,
    ) -> dict[str, Any]:
        # Layer 1: Schema validation
        schema_result = self.validator.validate(payload)

        # Layer 2: TIO structural validation
        tio = evaluate_tio_compliance(payload)

        # Layer 3: Build intent frame if not provided
        if not intent_frame:
            semantic = score_semantic_faithfulness(nl_intent, payload, self.embedder)
            intent_frame = self.translator.translate(
                nl_intent,
                taxonomy_target,
                retrieved_context,
                [],
                sample_index=0,
                source_kpis=semantic.get("nl_kpis", {}),
            )["intent_frame"]

        # Ensure intent_frame is not None for subsequent calls
        assert intent_frame is not None
        intent_frame_safe = cast(dict[str, Any], intent_frame)

        # Layer 3: Symbolic semantic equivalence checks
        symbolic = verify_semantic_alignment(nl_intent, intent_frame_safe, payload)

        # Layer 4: Evidence-grounding checks
        evidence = attribute_evidence(
            intent_frame_safe,
            retrieved_context,
            grounding_mode=self.settings.grounding_mode,
            similarity_threshold=self.settings.grounding_similarity_threshold,
        )

        # Layer 5: Optional LLM judge as tie-breaker/secondary reviewer
        llm_semantic = self._llm_semantic_review(nl_intent, payload)

        # Compute acceptance with symbolic checks as primary rejection mechanism
        schema_pass = schema_result.valid
        tio_pass = tio["score"] == 1.0
        symbolic_pass = symbolic["semantic_pass"]
        grounding_pass = evidence["grounding_pass"]

        # Accept only if all layers pass, with symbolic as mandatory
        accept = schema_pass and tio_pass and symbolic_pass and grounding_pass

        # If borderline (all but one layer passes), use LLM as tie-breaker
        if not accept and sum([schema_pass, tio_pass, symbolic_pass, grounding_pass]) == 3:
            if llm_semantic and "faithfulness" in llm_semantic:
                llm_score = float(llm_semantic["faithfulness"])
                if llm_score >= 0.9:  # High confidence override
                    accept = True
                    symbolic["notes"].append(f"LLM tie-breaker override: {llm_score}")

        # Legacy quality score for diagnostics (not used for acceptance)
        semantic = score_semantic_faithfulness(nl_intent, payload, self.embedder)  # Recompute for legacy score
        if llm_semantic and "faithfulness" in llm_semantic:
            semantic["llm_judge"] = float(llm_semantic["faithfulness"])
        realism = score_realism(payload, retrieved_context)  # Keep for diagnostics
        semantic_pass_score = 1.0 if symbolic_pass else 0.0
        grounding_score = evidence["supported_claim_ratio"]
        quality_score = (0.35 * semantic["score"]) + (0.25 * tio["score"]) + (0.25 * semantic_pass_score) + (0.15 * grounding_score)

        # Collect all validation notes
        notes = [*schema_result.errors, *tio["notes"], *realism["notes"], *symbolic["notes"], *evidence.get("notes", [])]
        if llm_semantic.get("notes"):
            notes.extend(llm_semantic["notes"])

        # Attempt repair if not accepted and within refinement limits
        repaired_payload = payload
        if not accept and refinement_count < self.settings.max_refinement_loops:
            repaired_payload = self.translator.repair_payload(payload, taxonomy_target, serialization, intent_frame=intent_frame_safe)
            notes.append("applied heuristic repair")

        return {
            "accepted": accept,
            "schema_validity": 1.0 if schema_pass else 0.0,
            "semantic": semantic,
            "symbolic": symbolic,
            "tio": tio,
            "realism": realism,
            "evidence": evidence,
            "quality_score": round(quality_score, 4),
            "notes": notes,
            "repaired_payload": repaired_payload,
            # New metadata fields for research-grade tracking
            "semantic_pass": symbolic_pass,
            "operator_pass": symbolic["operator_pass"],
            "constraint_coverage": symbolic["constraint_coverage"],
            "unsupported_claim_count": evidence["unsupported_claim_count"],
            "contradiction_count": symbolic["contradiction_count"],
            "grounding_mode": self.settings.grounding_mode,
            "grounding_pass": grounding_pass,
            "supported_claim_ratio": evidence["supported_claim_ratio"],
        }
