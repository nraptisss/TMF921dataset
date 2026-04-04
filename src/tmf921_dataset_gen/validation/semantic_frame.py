from __future__ import annotations

import json
import re
from typing import Any

import rdflib


SCENARIO_KEYWORDS = {
    "energy": {"energy", "consumption", "efficiency"},
    "slicing": {"slice", "deploy", "instantiate", "provision"},
    "predictive_assurance": {"predictive", "assurance", "degradation", "corrective", "remediation"},
    "multi_domain": {"multi-domain", "transport", "core", "coordinate"},
    "conflict_negotiation": {"conflict", "negotiate", "balance", "resolve"},
    "resilience": {"resilience", "protect", "preserve", "maintain"},
    "reporting": {"report", "reporting", "monitoring", "monitor"},
    "closed_loop_autonomy": {"closed-loop", "autonomous", "failover", "trigger", "heal", "reconfigure"},
}

METRIC_SPECS = {
    "latency_ms": {"aliases": ("latency",), "unit": "ms", "payload_key": "met:latency", "default_operator": "at_most"},
    "throughput_mbps": {"aliases": ("throughput",), "unit": "Mbps", "payload_key": "met:throughput", "default_operator": "at_least"},
    "throughput_gbps": {"aliases": ("throughput",), "unit": "Gbps", "payload_key": "met:throughput", "default_operator": "at_least"},
    "reliability_percent": {"aliases": ("reliability",), "unit": "%", "payload_key": "met:reliability", "default_operator": "at_least"},
    "availability_percent": {"aliases": ("availability",), "unit": "%", "payload_key": "met:availability", "default_operator": "at_least"},
    "energy_kwh": {"aliases": ("energy", "consumption"), "unit": "kWh", "payload_key": "met:energyConsumption", "default_operator": "at_most"},
    "device_count": {"aliases": ("devices", "device", "support"), "unit": "count", "payload_key": "sli:deviceCount", "default_operator": "exactly"},
    "delivery_ratio_percent": {"aliases": ("delivery ratio", "packet delivery ratio"), "unit": "%", "payload_key": "met:packetDeliveryRatio", "default_operator": "at_least"},
    "reaction_time_ms": {"aliases": ("reaction time", "corrective action", "failover", "trigger"), "unit": "ms", "payload_key": "met:reactionTime", "default_operator": "trigger_within"},
    "reporting_interval_seconds": {"aliases": ("report", "reporting interval", "reports every"), "unit": "seconds", "payload_key": "icm:reportingInterval", "default_operator": "periodic_every"},
}

OPERATOR_PATTERN_MAP = {
    "exactly": re.compile(r"\b(exactly|strictly|equal to|must be exactly)\b", re.IGNORECASE),
    "at_least": re.compile(r"\b(at least|minimum|min |no less than|exceeds?|greater than)\b", re.IGNORECASE),
    "at_most": re.compile(r"\b(at most|under|below|less than|no more than|max(?:imum)?)\b", re.IGNORECASE),
    "within": re.compile(r"\b(within)\b", re.IGNORECASE),
    "between": re.compile(r"\b(between)\b", re.IGNORECASE),
    "periodic_every": re.compile(r"\b(every|interval)\b", re.IGNORECASE),
    "trigger_within": re.compile(r"\b(trigger|failover|corrective action|remediation|heal(?:ing)?)\b", re.IGNORECASE),
}

ICM_URI = "http://www.models.tmforum.org/tio/v1.0.0/IntentCommonModel#"
MET_URI = "http://www.sdo2.org/TelecomMetrics/Version_1.0#"
SLI_URI = "http://io.irc.huawei.com/Io/v1.0.0/SliceExtensionModel#"


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def infer_expectation_type(scenario_family: str) -> str:
    return "icm:ReportingExpectation" if scenario_family == "reporting" else "icm:DeliveryExpectation"


def _extract_metric_context(text: str, metric_key: str) -> str:
    """Extract the text segment belonging to a specific metric for operator inference.

    This prevents operator keywords from one metric (e.g., 'under' for energy)
    from being incorrectly applied to another metric (e.g., throughput).

    Strategy: find the position of this metric's mention, then extract text
    from the previous metric boundary (or start) to the next metric boundary
    (or end). This creates per-metric text segments.
    """
    spec = METRIC_SPECS.get(metric_key)
    if not spec:
        return text.lower()

    lowered = text.lower()

    # Find the position of this metric's first mention
    metric_pos = -1
    for alias in spec["aliases"]:
        idx = lowered.find(alias.lower())
        if idx != -1 and (metric_pos == -1 or idx < metric_pos):
            metric_pos = idx

    if metric_pos == -1:
        return lowered

    # Find all metric mention positions to determine segment boundaries
    all_positions = []
    for other_key, other_spec in METRIC_SPECS.items():
        for alias in other_spec["aliases"]:
            idx = lowered.find(alias.lower())
            if idx != -1:
                all_positions.append(idx)

    all_positions.sort()
    # Deduplicate positions (same position from different aliases/metrics)
    all_positions = sorted(set(all_positions))

    # This metric's segment: from THIS metric's position to the NEXT metric's position (or end).
    # This ensures each metric only sees its own text, not the previous metric's operator keywords.
    seg_start = metric_pos
    seg_end = len(text)
    for i, pos in enumerate(all_positions):
        if pos == metric_pos:
            # Next metric position defines end of this segment
            if i + 1 < len(all_positions):
                seg_end = all_positions[i + 1]
            break

    return text[seg_start:seg_end].lower()


def infer_operator(text: str, metric_key: str, scenario_family: str) -> str:
    lowered = text.lower()
    if metric_key == "reporting_interval_seconds":
        return "periodic_every"
    if metric_key == "reaction_time_ms":
        return "trigger_within"

    # Extract text local to this metric to avoid cross-metric operator contamination
    local_text = _extract_metric_context(text, metric_key)

    if OPERATOR_PATTERN_MAP["exactly"].search(local_text) and metric_key in {"device_count", "delivery_ratio_percent"}:
        return "exactly"
    if OPERATOR_PATTERN_MAP["at_most"].search(local_text):
        return "at_most"
    if OPERATOR_PATTERN_MAP["at_least"].search(local_text):
        return "at_least"
    if OPERATOR_PATTERN_MAP["within"].search(local_text) and metric_key.endswith("_ms"):
        return "within"
    return METRIC_SPECS[metric_key]["default_operator"]


def _value_to_string(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _constraint_record(metric_key: str, value: Any, operator: str | None, source: str) -> dict[str, Any]:
    spec = METRIC_SPECS[metric_key]
    return {
        "metric": metric_key,
        "value": value,
        "unit": spec["unit"],
        "payload_key": spec["payload_key"],
        "operator": operator or spec["default_operator"],
        "source": source,
    }


def build_intent_frame(
    nl_intent: str,
    taxonomy_target: dict[str, Any],
    source_kpis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .semantic_score import extract_kpis

    extracted = extract_kpis(nl_intent)
    merged = dict(source_kpis or {})
    merged.update(extracted)
    constraints: list[dict[str, Any]] = []
    for metric_key, value in merged.items():
        if metric_key not in METRIC_SPECS:
            continue
        operator = infer_operator(nl_intent, metric_key, taxonomy_target["scenario_family"])
        constraints.append(_constraint_record(metric_key, value, operator, "nl_intent" if metric_key in extracted else "sampled"))

    scenario_family = taxonomy_target["scenario_family"]
    event_required = scenario_family in {"predictive_assurance", "closed_loop_autonomy"}
    event_trigger = None
    if re.search(r"\b(if|when).*(degradation|violation|failure|breach)\b", nl_intent, flags=re.IGNORECASE):
        event_trigger = "degradation_detected"
    elif scenario_family == "closed_loop_autonomy":
        event_trigger = "autonomous_remediation"
    elif scenario_family == "predictive_assurance":
        event_trigger = "predictive_alert"

    return {
        "taxonomy_category": taxonomy_target["taxonomy_category"],
        "layer": taxonomy_target["layer"],
        "traffic_profile": taxonomy_target["traffic_profile"],
        "scenario_family": scenario_family,
        "domain_context": taxonomy_target.get("domain_context", ""),
        "expectation_type": infer_expectation_type(scenario_family),
        "constraints": constraints,
        "event_required": event_required,
        "event_trigger": event_trigger,
        "grounding_claims": [
            {"claim_type": "domain_context", "value": taxonomy_target.get("domain_context", "")},
            {"claim_type": "scenario_family", "value": scenario_family},
        ],
    }


def canonical_name(intent_frame: dict[str, Any]) -> str:
    traffic = intent_frame["traffic_profile"].upper()
    scenario = intent_frame["scenario_family"].replace("_", " ").title()
    layer = intent_frame["layer"].title()
    context = intent_frame.get("domain_context", "").replace("-", " ").title()
    return f"{scenario} {traffic} {layer} Intent for {context}".strip()


def canonical_context(intent_frame: dict[str, Any]) -> str:
    metrics = ", ".join(
        f"{constraint['metric']} {constraint['operator']} {_value_to_string(constraint['value'])} {constraint['unit']}"
        for constraint in intent_frame.get("constraints", [])
    )
    return (
        f"{intent_frame['layer']} {intent_frame['traffic_profile']} "
        f"{intent_frame['scenario_family']} intent in {intent_frame.get('domain_context', 'generic context')}"
        f"{': ' + metrics if metrics else ''}"
    )


def payload_constraints(payload: dict[str, Any]) -> list[dict[str, Any]]:
    expression = payload.get("expression", {})
    expression_type = expression.get("@type")
    if expression_type == "JsonLdExpression":
        return _jsonld_constraints(expression.get("expressionValue", {}))
    if expression_type == "TurtleExpression":
        return _turtle_constraints(expression.get("expressionValue", ""))
    return []


def _jsonld_constraints(expression_value: dict[str, Any]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    graph = expression_value.get("@graph", [])
    for node in graph:
        if not isinstance(node, dict):
            continue
        params = node.get("icm:params")
        if not isinstance(params, dict):
            continue
        for payload_key, values in params.items():
            if payload_key == "icm:targetDescription":
                continue
            metric_key = next((key for key, spec in METRIC_SPECS.items() if spec["payload_key"] == payload_key), None)
            if metric_key is None or not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict) or not item:
                    continue
                operator_key, raw_value = next(iter(item.items()))
                operator = {
                    "icm:atMost": "at_most",
                    "icm:atLeast": "at_least",
                    "icm:value": "exactly" if metric_key == "device_count" else METRIC_SPECS[metric_key]["default_operator"],
                }.get(operator_key, operator_key.replace("icm:", "").lower())
                # reaction_time_ms uses trigger_within in intent_frame but atMost in JSON-LD
                if metric_key == "reaction_time_ms" and operator == "at_most":
                    operator = "trigger_within"
                constraints.append(_constraint_record(metric_key, _parse_metric_value(metric_key, raw_value), operator, "payload"))
    return constraints


def _turtle_constraints(ttl: str) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    if not ttl.strip():
        return constraints
    graph = rdflib.Graph()
    graph.parse(data=ttl, format="turtle")
    operator_suffix_map = {
        f"{ICM_URI}atMost": "at_most",
        f"{ICM_URI}atLeast": "at_least",
        f"{ICM_URI}value": "exactly",
    }
    for subject, predicate, obj in graph:
        predicate_str = str(predicate)
        metric_key = None
        if predicate_str.startswith(MET_URI) or predicate_str.startswith(SLI_URI) or predicate_str == f"{ICM_URI}reportingInterval":
            metric_key = next((key for key, spec in METRIC_SPECS.items() if spec["payload_key"].endswith(predicate_str.rsplit("/", 1)[-1]) or spec["payload_key"] == _shrink_predicate(predicate_str)), None)
            if metric_key is None:
                metric_key = next((key for key, spec in METRIC_SPECS.items() if _matches_predicate(spec["payload_key"], predicate_str)), None)
        if metric_key is None or not isinstance(obj, rdflib.term.BNode):
            continue
        for inner_predicate, inner_obj in graph.predicate_objects(obj):
            operator = operator_suffix_map.get(str(inner_predicate), "exactly")
            constraints.append(_constraint_record(metric_key, _parse_metric_value(metric_key, str(inner_obj)), operator, "payload"))
    return constraints


def _shrink_predicate(predicate_str: str) -> str:
    if predicate_str.startswith(MET_URI):
        return f"met:{predicate_str[len(MET_URI):]}"
    if predicate_str.startswith(SLI_URI):
        return f"sli:{predicate_str[len(SLI_URI):]}"
    if predicate_str.startswith(ICM_URI):
        return f"icm:{predicate_str[len(ICM_URI):]}"
    return predicate_str


def _matches_predicate(payload_key: str, predicate_str: str) -> bool:
    return payload_key == _shrink_predicate(predicate_str)


def _parse_metric_value(metric_key: str, raw_value: Any) -> Any:
    text = str(raw_value).strip('"')
    numeric = re.search(r"(\d+(?:\.\d+)?)", text)
    if metric_key == "device_count":
        return int(numeric.group(1)) if numeric else text
    if numeric:
        number = float(numeric.group(1))
        return int(number) if number.is_integer() else number
    return text


def attribute_evidence(
    intent_frame: dict[str, Any],
    retrieved_context: list[dict[str, Any]],
    grounding_mode: str,
    similarity_threshold: float,
) -> dict[str, Any]:
    evidence_map: dict[str, list[str]] = {}
    supported_claims = 0
    unsupported_claims = 0
    allowed_sources = {"seed", "oas_example", "oas_schema", "tr290_docx", "tr290_pdf", "tr290_markdown", "idan_reference"}
    claims: list[tuple[str, str]] = []
    for constraint in intent_frame.get("constraints", []):
        claims.append((constraint["metric"], _value_to_string(constraint["value"])))
    claims.append(("domain_context", intent_frame.get("domain_context", "")))
    claims.append(("scenario_family", intent_frame.get("scenario_family", "")))

    for claim_key, claim_value in claims:
        matches: list[str] = []
        claim_tokens = {token for token in _normalize_text(claim_value).split() if token}
        for row in retrieved_context:
            source_type = row.get("metadata", {}).get("source_type")
            if source_type not in allowed_sources:
                continue
            similarity = max(0.0, 1.0 - float(row.get("distance", 1.0)))
            row_text = row.get("text", "")
            row_norm = _normalize_text(row_text)
            if similarity < similarity_threshold:
                continue
            if claim_key == "domain_context":
                if claim_tokens and claim_tokens.issubset(set(row_norm.split())):
                    matches.append(row["id"])
            elif claim_key == "scenario_family":
                scenario_tokens = SCENARIO_KEYWORDS.get(claim_value, set())
                if scenario_tokens and any(token in row_norm for token in scenario_tokens):
                    matches.append(row["id"])
            else:
                if claim_value in row_text or claim_value in row_norm:
                    matches.append(row["id"])
        evidence_map[claim_key] = matches
        if matches:
            supported_claims += 1
        else:
            unsupported_claims += 1

    total_claims = max(1, len(claims))
    supported_ratio = supported_claims / total_claims
    grounding_pass = grounding_mode != "grounded_corpus" or unsupported_claims == 0
    return {
        "evidence_map": evidence_map,
        "supported_claim_ratio": round(supported_ratio, 4),
        "unsupported_claim_count": unsupported_claims,
        "grounding_pass": grounding_pass,
    }


def verify_semantic_alignment(
    nl_intent: str,
    intent_frame: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    payload_cons = payload_constraints(payload)
    expected_by_metric = {constraint["metric"]: constraint for constraint in intent_frame.get("constraints", [])}
    payload_by_metric = {constraint["metric"]: constraint for constraint in payload_cons}
    missing_metrics: list[str] = []
    operator_mismatches: list[str] = []
    value_mismatches: list[str] = []
    contradictions: list[str] = []

    for metric_key, expected in expected_by_metric.items():
        actual = payload_by_metric.get(metric_key)
        if actual is None:
            missing_metrics.append(metric_key)
            continue
        if actual["operator"] != expected["operator"]:
            operator_mismatches.append(metric_key)
        if _parse_metric_value(metric_key, actual["value"]) != _parse_metric_value(metric_key, expected["value"]):
            value_mismatches.append(metric_key)

    payload_name = _normalize_text(str(payload.get("name", "")))
    scenario = intent_frame["scenario_family"]
    # Only check name for contradictory scenarios — the context field contains
    # structured operator names (trigger_within, at_most, etc.) that can false-positive
    # against scenario keywords.
    contradictory_scenarios = {
        other for other, keywords in SCENARIO_KEYWORDS.items()
        if other != scenario and any(keyword in payload_name for keyword in keywords)
    }
    if contradictory_scenarios:
        contradictions.extend([f"contradictory scenario token: {item}" for item in sorted(contradictory_scenarios)])
    if scenario == "reporting" and "reportingInterval" not in json.dumps(payload):
        contradictions.append("reporting scenario missing reporting interval")
    if intent_frame.get("event_required") and "reaction_time_ms" not in expected_by_metric:
        contradictions.append("event-driven scenario missing reaction_time_ms in intent frame")
    if intent_frame.get("event_required") and "reaction_time_ms" not in payload_by_metric:
        contradictions.append("event-driven scenario missing reaction_time_ms in payload")
    expectation_type = intent_frame["expectation_type"]
    if payload.get("expression", {}).get("@type") == "JsonLdExpression":
        serialized = json.dumps(payload.get("expression", {}).get("expressionValue", {}), sort_keys=True)
        if expectation_type not in serialized:
            contradictions.append("wrong expectation type for scenario")
    elif payload.get("expression", {}).get("@type") == "TurtleExpression":
        if expectation_type not in str(payload.get("expression", {}).get("expressionValue", "")):
            contradictions.append("wrong expectation type for scenario")

    expected_count = len(expected_by_metric)
    matched_count = expected_count - len(missing_metrics) - len(value_mismatches)
    coverage = matched_count / expected_count if expected_count else 1.0
    operator_pass = not operator_mismatches
    semantic_pass = not missing_metrics and not value_mismatches and not contradictions and operator_pass
    notes = []
    notes.extend(f"missing constraint: {metric}" for metric in missing_metrics)
    notes.extend(f"operator mismatch: {metric}" for metric in operator_mismatches)
    notes.extend(f"value mismatch: {metric}" for metric in value_mismatches)
    notes.extend(contradictions)
    return {
        "semantic_pass": semantic_pass,
        "operator_pass": operator_pass,
        "constraint_coverage": round(coverage, 4),
        "unsupported_claim_count": 0,
        "contradiction_count": len(contradictions),
        "payload_constraints": payload_cons,
        "notes": notes,
    }
