from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from ..config import Settings
from ..llm import LLMRouter
from ..models.generated.tmf921_models import IntentFVO
from ..validation.jsonschema_validator import TMFJsonSchemaValidator
from ..validation.semantic_score import extract_kpis
from ..validation.semantic_frame import build_intent_frame, canonical_context, canonical_name


ICM_CONTEXT = {
    "icm": "http://www.models.tmforum.org/tio/v1.0.0/IntentCommonModel#",
    "idan": "http://www.idan-tmforum-catalyst.org/IntentDrivenAutonomousNetworks#",
    "sli": "http://io.irc.huawei.com/Io/v1.0.0/SliceExtensionModel#",
    "met": "http://www.sdo2.org/TelecomMetrics/Version_1.0#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "t": "http://www.w3.org/2006/time#",
}


class TranslatorAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.validator = TMFJsonSchemaValidator(settings, "Intent_FVO")
        self.router = LLMRouter(settings)

    def choose_serialization(self, sample_index: int) -> str:
        threshold = int(self.settings.jsonld_ratio * 100)
        return "json-ld" if (sample_index * 37 + self.settings.random_seed) % 100 < threshold else "turtle"

    def _slugify(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:64]

    def _priority_for(self, taxonomy_target: dict[str, Any]) -> str:
        if taxonomy_target["traffic_profile"] == "urllc":
            return "critical"
        if taxonomy_target["scenario_family"] in {"energy", "reporting"}:
            return "medium"
        return "high"

    def _llm_translation_options(self, nl_intent: str, taxonomy_target: dict[str, Any], kpis: dict[str, Any], context: list[dict[str, Any]], sample_index: int) -> dict[str, Any]:
        # Restricted LLM role: only for priority and serialization hints, not name/context
        if not self.settings.enable_llm_translation_hints or not self.router.supports_generation():
            return {}
        default_serialization = self.choose_serialization(sample_index)
        context_text = "\n".join([f"- {chunk.get('text', '')[:200]}" for chunk in context[:3]])  # Top 3 chunks, truncated
        prompt = f"""
Context from knowledge base:
{context_text}

Planning TMF921 translation hints:

You are assisting a TMF921 translation system. Return JSON only with optional keys for priority and serialization hints.
Do not provide name or context - these will be derived from the normalized frame.

Optional keys: {{"priority": "critical|high|medium|low", "serialization": "json-ld|turtle"}}

Taxonomy target:
{json.dumps(taxonomy_target, indent=2)}

KPI summary:
{json.dumps(kpis, indent=2)}

Default serialization: {default_serialization}

Natural-language intent:
{nl_intent}
""".strip()
        try:
            response = self.router.generate_json(
                prompt,
                model=self.settings.bulk_model,
                temperature=0.1,
                max_new_tokens=self.settings.local_planning_max_new_tokens,
            )
            return response if isinstance(response, dict) else {}
        except Exception:
            return {}

    def _operator_key(self, operator: str, metric_key: str) -> str:
        if operator == "trigger_within":
            return "icm:atMost"
        if operator in {"at_most", "within"}:
            return "icm:atMost"
        if operator == "at_least":
            return "icm:atLeast"
        if operator in {"periodic_every", "exactly"} or metric_key == "device_count":
            return "icm:value"
        return "icm:value"

    def _expression_params(self, intent_frame: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "icm:targetDescription": f"6G {intent_frame['layer']} {intent_frame['traffic_profile']} intent",
        }
        unit_suffixes = {
            "latency_ms": "ms",
            "throughput_mbps": "Mbps",
            "throughput_gbps": "Gbps",
            "reliability_percent": "%",
            "availability_percent": "%",
            "energy_kwh": "kWh",
            "delivery_ratio_percent": "%",
            "reaction_time_ms": "ms",
            "reporting_interval_seconds": "",
            "device_count": "",
        }
        for constraint in intent_frame.get("constraints", []):
            payload_key = constraint["payload_key"]
            metric_key = constraint["metric"]
            operator_key = self._operator_key(constraint["operator"], metric_key)
            suffix = unit_suffixes.get(metric_key, "")
            rendered_value = f"{constraint['value']} {suffix}".strip()
            params.setdefault(payload_key, []).append({operator_key: rendered_value})
        return params

    def _build_jsonld_expression(self, slug: str, intent_frame: dict[str, Any]) -> dict[str, Any]:
        params = self._expression_params(intent_frame)
        expectation_type = intent_frame["expectation_type"]
        return {
            "@type": "JsonLdExpression",
            "@baseType": "IntentExpression",
            "iri": f"https://tmf921.dataset.local/expressions/{slug}",
            "expressionValue": {
                "@context": ICM_CONTEXT,
                "@graph": [
                    {
                        "@id": f"idan:{slug}",
                        "@type": "icm:Intent",
                        "icm:intentOwner": "idan:DatasetGenerator",
                        "icm:hasExpectation": [{"@id": f"idan:{slug}:expectation"}],
                    },
                    {
                        "@id": f"idan:{slug}:expectation",
                        "@type": expectation_type,
                        "icm:target": {"@id": f"_:{slug}-target"},
                        "icm:params": params,
                    },
                ],
            },
        }

    def _build_turtle_expression(self, slug: str, intent_frame: dict[str, Any]) -> dict[str, Any]:
        params = self._expression_params(intent_frame)
        expectation_type = intent_frame["expectation_type"]
        param_lines = []
        param_ids = []
        param_counter = 1
        for metric, values in params.items():
            value_entries = values if isinstance(values, list) else [{"icm:value": str(values)}]
            for value in value_entries:
                key, metric_value = next(iter(value.items()))
                param_id = f"idan:{slug}_param_{param_counter}"
                param_lines.append(
                    f"{param_id} a icm:PropertyParam ; {metric} [ {key} \"{metric_value}\" ] ."
                )
                param_ids.append(param_id)
                param_counter += 1
        params_joined = ", ".join(param_ids) if param_ids else ""
        turtle = "\n".join(
            [
                "@prefix icm: <http://www.models.tmforum.org/tio/v1.0.0/IntentCommonModel#> .",
                "@prefix idan: <http://www.idan-tmforum-catalyst.org/IntentDrivenAutonomousNetworks#> .",
                "@prefix sli: <http://io.irc.huawei.com/Io/v1.0.0/SliceExtensionModel#> .",
                "@prefix met: <http://www.sdo2.org/TelecomMetrics/Version_1.0#> .",
                "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
                "",
                f"idan:{slug} a icm:Intent ;",
                "  icm:intentOwner idan:DatasetGenerator ;",
                f"  icm:hasExpectation idan:{slug}_expectation .",
                "",
                f"idan:{slug}_expectation a {expectation_type} ;",
                f"  icm:target _:{slug}_target ;",
                f"  icm:params {params_joined or '[]'} .",
                "",
                *param_lines,
            ]
        )
        return {
            "@type": "TurtleExpression",
            "@baseType": "Expression",
            "iri": f"https://tmf921.dataset.local/expressions/{slug}.ttl",
            "expressionValue": turtle,
        }

    def translate(
        self,
        nl_intent: str,
        taxonomy_target: dict[str, Any],
        retrieved_context: list[dict[str, Any]],
        seed_ids: list[str],
        sample_index: int,
        source_kpis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Use source_kpis when explicitly provided (even if empty); only extract
        # from NL text when no source was given. This prevents the NLP extractor
        # from injecting spurious KPIs (e.g., phantom device_count) when the
        # diversity agent already determined the correct set.
        if source_kpis is not None:
            kpis = dict(source_kpis)
        else:
            kpis = extract_kpis(nl_intent)
        translation_hints = self._llm_translation_options(nl_intent, taxonomy_target, kpis, retrieved_context, sample_index)
        intent_frame = build_intent_frame(nl_intent, taxonomy_target, kpis)
        slug_source = f"{taxonomy_target['taxonomy_category']}-{sample_index}"
        slug = self._slugify(slug_source)
        serialization = translation_hints.get("serialization") or self.choose_serialization(sample_index)
        expression = (
            self._build_jsonld_expression(slug, intent_frame)
            if serialization == "json-ld"
            else self._build_turtle_expression(slug, intent_frame)
        )
        payload = {
            "@type": "Intent" if serialization == "json-ld" else "ProbeIntent",
            "name": canonical_name(intent_frame),
            "description": nl_intent,
            "priority": str(translation_hints.get("priority") or self._priority_for(taxonomy_target)),
            "context": canonical_context(intent_frame),
            "version": "1.0",
            "lifecycleStatus": "active",
            "expression": expression,
        }
        IntentFVO.model_validate(payload)
        validation = self.validator.validate(payload)
        if not validation.valid:
            raise ValueError(f"Generated payload failed schema validation: {validation.errors}")
        return {
            "tmf921_intent": payload,
            "serialization": serialization,
            "metadata": {
                "taxonomy_category": taxonomy_target["taxonomy_category"],
                "taxonomy_target": dict(taxonomy_target),
                "kpis": kpis,
                "seed_id": seed_ids[0] if seed_ids else None,
                "generation_timestamp": datetime.now(timezone.utc),
                "retrieved_context_ids": [row["id"] for row in retrieved_context],
                "translation_backend": self.settings.inference_backend,
                "intent_frame": intent_frame,
                "constraint_set": intent_frame.get("constraints", []),
                "grounding_mode": self.settings.grounding_mode,
            },
            "intent_frame": intent_frame,
        }

    def repair_payload(self, payload: dict[str, Any], taxonomy_target: dict[str, Any], serialization: str, intent_frame: dict[str, Any] | None = None) -> dict[str, Any]:
        repaired = deepcopy(payload)
        intent_frame = intent_frame or build_intent_frame(str(payload.get("description", "")), taxonomy_target, {})
        repaired.setdefault("@type", "Intent" if serialization == "json-ld" else "ProbeIntent")
        repaired.setdefault("name", canonical_name(intent_frame))
        repaired.setdefault("description", taxonomy_target["taxonomy_category"])
        repaired.setdefault("priority", self._priority_for(taxonomy_target))
        repaired.setdefault("context", canonical_context(intent_frame))
        repaired.setdefault("version", "1.0")
        if "expression" not in repaired:
            repaired["expression"] = (
                self._build_jsonld_expression(self._slugify(repaired["name"]), intent_frame)
                if serialization == "json-ld"
                else self._build_turtle_expression(self._slugify(repaired["name"]), intent_frame)
            )
        return repaired
