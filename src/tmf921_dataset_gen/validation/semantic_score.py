from __future__ import annotations

import re
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from ..rag.embeddings import EmbeddingBackend, HashingEmbeddingModel


def _first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _parse_number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _parse_int(raw: str) -> int:
    return int(raw.replace(",", ""))


def _nlp_kpi_extraction(text: str) -> dict[str, Any]:
    lowered = text.lower()
    kpis: dict[str, Any] = {}

    # Pattern: (Metric Name) ... (Value) (Unit)  OR  (Value) (Unit) ... (Metric Name)
    def extract_metric(metric_pattern, value_pattern, unit_pattern):
        # Case 1: Metric then Value (e.g., "latency of 5 ms")
        match1 = re.search(rf"{metric_pattern}[^0-9]{{0,40}}?({value_pattern})\s*{unit_pattern}", lowered)
        if match1:
            return _parse_number(match1.group(1))
        # Case 2: Value then Metric (e.g., "5 ms latency")
        match2 = re.search(rf"({value_pattern})\s*{unit_pattern}[^0-9]{{0,40}}?{metric_pattern}", lowered)
        if match2:
            return _parse_number(match2.group(1))
        return None

    numeric_pattern = r"\d[\d,]*(?:\.\d+)?"

    val = extract_metric(r"latency", numeric_pattern, r"(?:ms|milliseconds?)")
    if val is not None: kpis["latency_ms"] = val

    val = extract_metric(r"throughput", numeric_pattern, r"(?:gbps|gigabits?|gb/s)")
    if val is not None: kpis["throughput_gbps"] = val

    val = extract_metric(r"throughput", numeric_pattern, r"(?:mbps|megabits?|mb/s)")
    if val is not None: kpis["throughput_mbps"] = val

    val = extract_metric(r"energy(?:consumption)?", numeric_pattern, r"(?:kwh|kilowatt.?hours?)")
    if val is not None: kpis["energy_kwh"] = val

    val = extract_metric(r"(?:reaction\s*time|corrective\s*action|failover|trigger|remediation)", numeric_pattern, r"(?:ms|milliseconds?)")
    if val is not None: kpis["reaction_time_ms"] = val

    # Device count is slightly different
    device_match = re.search(r"(\d[\d,]*)\s+(?:devices?|robots?|sensors?|vehicles|users|people)", lowered)
    if device_match:
        kpis["device_count"] = _parse_int(device_match.group(1))

    val = extract_metric(r"reliability", numeric_pattern, r"%")
    if val is not None: kpis["reliability_percent"] = val

    val = extract_metric(r"availability", numeric_pattern, r"%")
    if val is not None: kpis["availability_percent"] = val

    val = extract_metric(r"(?:packet\s*delivery\s*ratio|delivery\s*ratio)", numeric_pattern, r"%")
    if val is not None: kpis["delivery_ratio_percent"] = val

    report_match = re.search(r"(?:reports? every|reporting interval(?: of)?)\s*(\d+)\s*(seconds?|minutes?)", lowered, flags=re.IGNORECASE)
    if report_match:
        interval_value = int(report_match.group(1))
        interval_unit = report_match.group(2).lower()
        kpis["reporting_interval_seconds"] = interval_value * (60 if interval_unit.startswith("minute") else 1)
    else:
        reporting_interval_seconds = _first_int(r"reportinginterval[^0-9]{0,20}?(\d+)", lowered)
        if reporting_interval_seconds is not None:
            kpis["reporting_interval_seconds"] = reporting_interval_seconds

    return kpis


UNIT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|gbps|mbps|kwh|%)", re.IGNORECASE)
COUNT_PATTERN = re.compile(r"(\d+)\s*(users|robots|industrial robots|sensors|vehicles|iot sensors)", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def score_kpi_plausibility(kpis: dict[str, Any], priority: str = "high") -> dict[str, Any]:
    """Check if KPI values are realistic for the given priority."""
    issues = []
    score = 1.0
    
    # Reliability/Availability usually > 90%
    for m in ["reliability_percent", "availability_percent"]:
        if m in kpis:
            val = kpis[m]
            if val < 90.0:
                issues.append(f"{m} is unusually low: {val}%")
                score -= 0.2
            if priority == "critical" and val < 99.0:
                issues.append(f"{m} for critical intent should be > 99%: {val}%")
                score -= 0.2
                
    if "latency_ms" in kpis and kpis["latency_ms"] > 1000:
        issues.append(f"latency_ms is unusually high: {kpis['latency_ms']}ms")
        score -= 0.2
        
    if "device_count" in kpis and kpis["device_count"] <= 0:
        issues.append(f"device_count must be positive: {kpis['device_count']}")
        score -= 0.3
        
    return {"score": max(0.0, score), "notes": issues}

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
    """Convert a payload to text suitable for semantic comparison with NL intent.

    Uses human-readable fields (name, description, context) rather than raw
    JSON-LD/Turtle serialization, which would produce misleadingly low cosine
    similarity due to structural differences.
    """
    parts = []
    # Use description (the NL intent) and context (structured summary) as the
    # primary text representations for comparison
    desc = intent_payload.get("description", "")
    context = intent_payload.get("context", "")
    name = intent_payload.get("name", "")
    if desc:
        parts.append(desc)
    if context:
        parts.append(context)
    if name:
        parts.append(name)
    return " ".join(parts)


def _token_overlap(lhs: str, rhs: str) -> float:
    left_tokens = set(TOKEN_PATTERN.findall(lhs.lower()))
    right_tokens = set(TOKEN_PATTERN.findall(rhs.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _numeric_match(lhs: Any, rhs: Any) -> float:
    try:
        left = float(lhs)
        right = float(rhs)
    except (TypeError, ValueError):
        return 1.0 if lhs == rhs else 0.0
    if left == right:
        return 1.0
    # Use relative tolerance: 10% of the larger value, with a minimum floor of 1.0
    # but scale the floor based on the metric magnitude to avoid inconsistent tolerances
    magnitude = max(abs(left), abs(right), 1.0)
    tolerance = max(magnitude * 0.1, 1.0)
    return max(0.0, 1.0 - (abs(left - right) / tolerance))


def _kpi_match_score(nl_kpis: dict[str, Any], payload_kpis: dict[str, Any]) -> float:
    if not nl_kpis:
        return 0.0
    scores: list[float] = []
    for key, nl_value in nl_kpis.items():
        if key not in payload_kpis:
            scores.append(0.0)
            continue
        scores.append(_numeric_match(nl_value, payload_kpis[key]))
    return sum(scores) / len(scores)


def score_semantic_faithfulness(
    nl_intent: str,
    intent_payload: dict[str, Any],
    embedder: EmbeddingBackend | None = None,
) -> dict[str, Any]:
    embedder = embedder or HashingEmbeddingModel()
    nl_kpis = extract_kpis(nl_intent)
    payload_text = intent_payload_to_text(intent_payload)
    payload_kpis = extract_kpis(payload_text)

    overlap_score = _kpi_match_score(nl_kpis, payload_kpis)
    lexical_overlap = _token_overlap(nl_intent, payload_text)
    nl_vec = embedder.embed_query(nl_intent)
    payload_vec = embedder.embed_query(payload_text)
    cosine = float(cosine_similarity([nl_vec], [payload_vec])[0][0])
    if nl_kpis:
        score = max(0.0, min(1.0, (0.7 * overlap_score) + (0.2 * cosine) + (0.1 * lexical_overlap)))
    else:
        score = max(0.0, min(1.0, (0.75 * cosine) + (0.25 * lexical_overlap)))
    return {
        "cosine": cosine,
        "overlap": overlap_score,
        "lexical_overlap": lexical_overlap,
        "score": score,
        "nl_kpis": nl_kpis,
        "payload_kpis": payload_kpis,
    }
