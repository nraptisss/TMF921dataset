from __future__ import annotations

import re
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from ..rag.embeddings import EmbeddingBackend, HashingEmbeddingModel


UNIT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|gbps|mbps|kwh|%)", re.IGNORECASE)
COUNT_PATTERN = re.compile(r"(\d+)\s*(users|robots|industrial robots|sensors|vehicles|iot sensors)", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def extract_kpis(text: str) -> dict[str, Any]:
    lowered = text.lower()
    metrics: dict[str, Any] = {}
    for value, unit in UNIT_PATTERN.findall(lowered):
        numeric = float(value)
        key = unit.lower()
        if key == "ms":
            metrics.setdefault("latency_ms", numeric)
        elif key == "gbps":
            metrics.setdefault("throughput_gbps", numeric)
        elif key == "mbps":
            metrics.setdefault("throughput_mbps", numeric)
        elif key == "kwh":
            metrics.setdefault("energy_kwh", numeric)
        elif key == "%":
            metrics.setdefault("percentage_values", []).append(numeric)
    if "reliability" in lowered and metrics.get("percentage_values"):
        metrics["reliability_percent"] = metrics["percentage_values"][0]
    if "availability" in lowered and metrics.get("percentage_values"):
        metrics["availability_percent"] = metrics["percentage_values"][-1]
    for count, noun in COUNT_PATTERN.findall(lowered):
        metrics[f"count_{noun.replace(' ', '_')}"] = int(count)
    if "5 minutes" in lowered:
        metrics["reporting_interval_seconds"] = 300
    if "60 seconds" in lowered:
        metrics["reporting_interval_seconds"] = 60
    return metrics


def intent_payload_to_text(intent_payload: dict[str, Any]) -> str:
    expression = intent_payload.get("expression", {})
    expression_type = expression.get("@type", "")
    important_terms = [expression_type]
    if expression_type == "JsonLdExpression":
        serialized = __import__("json").dumps(expression.get("expressionValue", {}), sort_keys=True)
        for marker in ["DeliveryExpectation", "ReportingExpectation", "Negotiation", "energyConsumption", "latency", "throughput", "reliability"]:
            if marker in serialized:
                important_terms.append(marker)
    elif expression_type == "TurtleExpression":
        ttl = expression.get("expressionValue", "")
        for marker in ["DeliveryExpectation", "ReportingExpectation", "Negotiation", "energyConsumption", "latency", "throughput", "reliability"]:
            if marker in ttl:
                important_terms.append(marker)
    return " ".join(
        str(part)
        for part in [
            intent_payload.get("name", ""),
            intent_payload.get("description", ""),
            intent_payload.get("context", ""),
            " ".join(important_terms),
        ]
        if part
    )


def _token_overlap(lhs: str, rhs: str) -> float:
    left_tokens = set(TOKEN_PATTERN.findall(lhs.lower()))
    right_tokens = set(TOKEN_PATTERN.findall(rhs.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def score_semantic_faithfulness(
    nl_intent: str,
    intent_payload: dict[str, Any],
    embedder: EmbeddingBackend | None = None,
) -> dict[str, Any]:
    embedder = embedder or HashingEmbeddingModel()
    nl_kpis = extract_kpis(nl_intent)
    payload_text = intent_payload_to_text(intent_payload)
    payload_kpis = extract_kpis(payload_text)

    overlap_keys = set(nl_kpis).intersection(payload_kpis)
    overlap_score = len(overlap_keys) / max(1, len(set(nl_kpis)))
    lexical_overlap = _token_overlap(nl_intent, payload_text)
    nl_vec = embedder.embed_query(nl_intent)
    payload_vec = embedder.embed_query(payload_text)
    cosine = float(cosine_similarity([nl_vec], [payload_vec])[0][0])
    # Use actual cosine similarity without artificial boosting
    # Score combines overlap and cosine without artificial floors
    score = max(0.0, min(1.0, (0.5 * overlap_score) + (0.5 * cosine)))
    return {
        "cosine": cosine,
        "overlap": overlap_score,
        "lexical_overlap": lexical_overlap,
        "score": score,
        "nl_kpis": nl_kpis,
        "payload_kpis": payload_kpis,
    }
