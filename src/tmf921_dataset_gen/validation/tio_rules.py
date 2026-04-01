from __future__ import annotations

import json
import re
from typing import Any

import rdflib

ICM_URI = "http://www.models.tmforum.org/tio/v1.0.0/IntentCommonModel#"
IDAN_URI = "http://www.idan-tmforum-catalyst.org/IntentDrivenAutonomousNetworks#"


TURTLE_INTENT_PATTERN = re.compile(r"icm:Intent|DeliveryExpectation|ReportingExpectation|PropertyExpectation")


def evaluate_tio_compliance(intent_payload: dict[str, Any]) -> dict[str, Any]:
    expression = intent_payload.get("expression", {})
    expression_type = expression.get("@type")
    notes: list[str] = []
    score = 0.0
    if expression_type == "JsonLdExpression":
        expression_value = expression.get("expressionValue", {})
        context = expression_value.get("@context", {}) if isinstance(expression_value, dict) else {}
        serialized = json.dumps(expression_value, sort_keys=True)
        if "icm" in context:
            score += 0.35
        else:
            notes.append("missing icm prefix in JSON-LD context")
        if "idan" in context:
            score += 0.15
        if "Intent" in serialized:
            score += 0.25
        else:
            notes.append("JSON-LD expression does not mention an Intent node")
        if any(term in serialized for term in ["DeliveryExpectation", "ReportingExpectation", "PropertyExpectation", "Negotiation"]):
            score += 0.25
        else:
            notes.append("JSON-LD expression lacks expectation semantics")
    elif expression_type == "TurtleExpression":
        ttl = expression.get("expressionValue", "")
        try:
            graph = rdflib.Graph()
            graph.parse(data=ttl, format="turtle")
            score += 0.4
        except Exception as exc:
            notes.append(f"turtle parse failure: {exc}")
            graph = None
        if TURTLE_INTENT_PATTERN.search(ttl):
            score += 0.35
        else:
            notes.append("turtle expression lacks core TIO terms")
        if ttl.count("@prefix") >= 4:
            score += 0.25
        else:
            notes.append("turtle expression is missing required prefixes")
    else:
        notes.append("unsupported expression type")
    return {
        "score": round(min(1.0, score), 4),
        "notes": notes,
    }
