from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from ..config import Settings
from ..models.generated.tmf921_models import IntentFVO
from ..validation.jsonschema_validator import TMFJsonSchemaValidator
from ..validation.semantic_score import extract_kpis


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

    def _expression_params(self, kpis: dict[str, Any], taxonomy_target: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "icm:targetDescription": f"6G {taxonomy_target['layer']} {taxonomy_target['traffic_profile']} intent",
        }
        if "latency_ms" in kpis:
            params["met:latency"] = [{"icm:atMost": f"{kpis['latency_ms']} ms"}]
        if "throughput_mbps" in kpis:
            params["met:throughput"] = [{"icm:atLeast": f"{kpis['throughput_mbps']} Mbps"}]
        if "throughput_gbps" in kpis:
            params["met:throughput"] = [{"icm:atLeast": f"{kpis['throughput_gbps']} Gbps"}]
        if "reliability_percent" in kpis:
            params["met:reliability"] = [{"icm:atLeast": f"{kpis['reliability_percent']} %"}]
        if "availability_percent" in kpis:
            params["met:availability"] = [{"icm:atLeast": f"{kpis['availability_percent']} %"}]
        if "energy_kwh" in kpis:
            params["met:energyConsumption"] = [{"icm:atMost": f"{kpis['energy_kwh']} kWh"}]
        if "device_count" in kpis:
            params["sli:deviceCount"] = [{"icm:value": str(kpis['device_count'])}]
        if "delivery_ratio_percent" in kpis:
            params["met:packetDeliveryRatio"] = [{"icm:atLeast": f"{kpis['delivery_ratio_percent']} %"}]
        if "reaction_time_ms" in kpis:
            params["met:reactionTime"] = [{"icm:atMost": f"{kpis['reaction_time_ms']} ms"}]
        if "reporting_interval_seconds" in kpis:
            params["icm:reportingInterval"] = [{"icm:value": str(kpis['reporting_interval_seconds'])}]
        return params

    def _build_jsonld_expression(self, slug: str, kpis: dict[str, Any], taxonomy_target: dict[str, Any]) -> dict[str, Any]:
        params = self._expression_params(kpis, taxonomy_target)
        expectation_type = "icm:ReportingExpectation" if taxonomy_target["scenario_family"] == "reporting" else "icm:DeliveryExpectation"
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

    def _build_turtle_expression(self, slug: str, kpis: dict[str, Any], taxonomy_target: dict[str, Any]) -> dict[str, Any]:
        params = self._expression_params(kpis, taxonomy_target)
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
                f"idan:{slug}_expectation a icm:DeliveryExpectation ;",
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
    ) -> dict[str, Any]:
        kpis = extract_kpis(nl_intent)
        slug = self._slugify(f"{taxonomy_target['taxonomy_category']}-{sample_index}")
        serialization = self.choose_serialization(sample_index)
        expression = (
            self._build_jsonld_expression(slug, kpis, taxonomy_target)
            if serialization == "json-ld"
            else self._build_turtle_expression(slug, kpis, taxonomy_target)
        )
        payload = {
            "@type": "Intent" if serialization == "json-ld" else "ProbeIntent",
            "name": slug.replace("-", " ").title().replace(" ", "-"),
            "description": nl_intent,
            "priority": self._priority_for(taxonomy_target),
            "context": f"6g-{taxonomy_target['layer']}-{taxonomy_target['scenario_family']}",
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
                "kpis": kpis,
                "seed_id": seed_ids[0] if seed_ids else None,
                "generation_timestamp": datetime.now(timezone.utc),
                "retrieved_context_ids": [row["id"] for row in retrieved_context],
            },
        }

    def repair_payload(self, payload: dict[str, Any], taxonomy_target: dict[str, Any], serialization: str) -> dict[str, Any]:
        repaired = deepcopy(payload)
        repaired.setdefault("@type", "Intent" if serialization == "json-ld" else "ProbeIntent")
        repaired.setdefault("name", self._slugify(taxonomy_target["taxonomy_category"]))
        repaired.setdefault("description", taxonomy_target["taxonomy_category"])
        repaired.setdefault("priority", self._priority_for(taxonomy_target))
        repaired.setdefault("context", f"6g-{taxonomy_target['layer']}-{taxonomy_target['scenario_family']}")
        repaired.setdefault("version", "1.0")
        if "expression" not in repaired:
            repaired["expression"] = (
                self._build_jsonld_expression(self._slugify(repaired["name"]), {}, taxonomy_target)
                if serialization == "json-ld"
                else self._build_turtle_expression(self._slugify(repaired["name"]), {}, taxonomy_target)
            )
        return repaired
