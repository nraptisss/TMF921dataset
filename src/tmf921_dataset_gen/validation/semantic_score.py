from __future__ import annotations

import re
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from ..rag.embeddings import EmbeddingBackend, HashingEmbeddingModel


def _nlp_kpi_extraction(text: str) -> dict[str, Any]:
    """Enhanced KPI extraction using basic NLP patterns."""
    import re
    lowered = text.lower()

    kpis = {}
    # Use regex for numbers with units
    unit_patterns = {
        'latency_ms': r'(\d+(?:\.\d+)?)\s*(ms|milliseconds?)',
        'throughput_gbps': r'(\d+(?:\.\d+)?)\s*(gbps|gigabits?|gb/s)',
        'throughput_mbps': r'(\d+(?:\.\d+)?)\s*(mbps|megabits?|mb/s)',
        'energy_kwh': r'(\d+(?:\.\d+)?)\s*(kwh|kilowatt.?hours?)',
        'percentage': r'(\d+(?:\.\d+)?)\s*%',
    }

    for key, pattern in unit_patterns.items():
        match = re.search(pattern, lowered)
        if match:
            kpis[key] = float(match.group(1))

    # Reliability/availability from percentages
    if 'reliability' in lowered and 'percentage' in kpis:
        kpis['reliability_percent'] = kpis['percentage']
    if 'availability' in lowered and 'percentage' in kpis:
        kpis['availability_percent'] = kpis['percentage']

    # Count patterns
    count_patterns = {
        'count_users': r'(\d+)\s*(users?|people)',
        'count_sensors': r'(\d+)\s*(sensors?)',
        'count_devices': r'(\d+)\s*(devices?|robots?)',
    }

    for key, pattern in count_patterns.items():
        match = re.search(pattern, lowered)
        if match:
            kpis[key] = int(match.group(1))

    # Reporting interval
    if '5 minutes' in lowered:
        kpis['reporting_interval_seconds'] = 300
    elif '60 seconds' in lowered:
        kpis['reporting_interval_seconds'] = 60

    return kpis


UNIT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|gbps|mbps|kwh|%)", re.IGNORECASE)
COUNT_PATTERN = re.compile(r"(\d+)\s*(users|robots|industrial robots|sensors|vehicles|iot sensors)", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def extract_kpis(text: str) -> dict[str, Any]:
    metrics = _nlp_kpi_extraction(text)
    # Plausibility checks
    if 'latency_ms' in metrics and metrics['latency_ms'] <= 0:
        del metrics['latency_ms']
    if 'throughput_gbps' in metrics and metrics['throughput_gbps'] <= 0:
        del metrics['throughput_gbps']
    if 'reliability_percent' in metrics and not (0 <= metrics['reliability_percent'] <= 100):
        del metrics['reliability_percent']
    if 'availability_percent' in metrics and not (0 <= metrics['availability_percent'] <= 100):
        del metrics['availability_percent']
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
