from __future__ import annotations

import json
import re
from typing import Any

import rdflib

ICM_URI = "http://www.models.tmforum.org/tio/v1.0.0/IntentCommonModel#"
IDAN_URI = "http://www.idan-tmforum-catalyst.org/IntentDrivenAutonomousNetworks#"
MET_URI = "http://www.sdo2.org/TelecomMetrics/Version_1.0#"
SLI_URI = "http://io.irc.huawei.com/Io/v1.0.0/SliceExtensionModel#"


TURTLE_INTENT_PATTERN = re.compile(r"icm:Intent|DeliveryExpectation|ReportingExpectation|PropertyExpectation")


def _jsonld_nodes(expression_value: dict[str, Any]) -> list[dict[str, Any]]:
    graph = expression_value.get("@graph", [])
    return [node for node in graph if isinstance(node, dict)]


def _jsonld_has_nontrivial_params(nodes: list[dict[str, Any]]) -> bool:
    for node in nodes:
        params = node.get("icm:params")
        if isinstance(params, dict):
            for key in params:
                if key != "icm:targetDescription":
                    return True
    return False


def _graph_has_expectation(graph: rdflib.Graph) -> bool:
    expectation_types = {
        rdflib.URIRef(f"{ICM_URI}DeliveryExpectation"),
        rdflib.URIRef(f"{ICM_URI}ReportingExpectation"),
        rdflib.URIRef(f"{ICM_URI}PropertyExpectation"),
    }
    for expectation_type in expectation_types:
        if any(graph.subjects(rdflib.RDF.type, expectation_type)):
            return True
    return False


def _graph_has_nontrivial_metric(graph: rdflib.Graph) -> bool:
    allowed_predicates = {rdflib.URIRef(f"{ICM_URI}reportingInterval")}
    for _, predicate, _ in graph:
        predicate_str = str(predicate)
        if predicate in allowed_predicates:
            return True
        if predicate_str.startswith(MET_URI) or predicate_str.startswith(SLI_URI):
            return True
    return False


def evaluate_tio_compliance(intent_payload: dict[str, Any]) -> dict[str, Any]:
    expression = intent_payload.get("expression", {})
    expression_type = expression.get("@type")
    notes: list[str] = []
    score = 0.0
    if expression_type == "JsonLdExpression":
        expression_value = expression.get("expressionValue", {})
        context = expression_value.get("@context", {}) if isinstance(expression_value, dict) else {}
        serialized = json.dumps(expression_value, sort_keys=True)
        nodes = _jsonld_nodes(expression_value if isinstance(expression_value, dict) else {})
        if "icm" in context:
            score += 0.2
        else:
            notes.append("missing icm prefix in JSON-LD context")
        if "idan" in context:
            score += 0.1
        if any(node.get("@type") == "icm:Intent" for node in nodes):
            score += 0.2
        else:
            notes.append("JSON-LD expression does not mention an Intent node")
        if any(
            node.get("@type") in {"icm:DeliveryExpectation", "icm:ReportingExpectation", "icm:PropertyExpectation"}
            for node in nodes
        ):
            score += 0.2
        else:
            notes.append("JSON-LD expression lacks expectation semantics")
        if _jsonld_has_nontrivial_params(nodes):
            score += 0.3
        else:
            notes.append("JSON-LD expression is missing concrete KPI constraints")
        if serialized:
            score += 0.0
    elif expression_type == "TurtleExpression":
        ttl = expression.get("expressionValue", "")
        try:
            graph = rdflib.Graph()
            graph.parse(data=ttl, format="turtle")
            score += 0.4
            intent_nodes = list(graph.subjects(rdflib.RDF.type, rdflib.URIRef(f"{ICM_URI}Intent")))
            if intent_nodes:
                score += 0.15
            else:
                notes.append("no valid Intent nodes found in RDF graph")
            if _graph_has_expectation(graph):
                score += 0.15
            else:
                notes.append("no valid expectation nodes found in RDF graph")
            if _graph_has_nontrivial_metric(graph):
                score += 0.3
            else:
                notes.append("turtle expression is missing concrete KPI constraints")
        except Exception as exc:
            notes.append(f"turtle parse failure: {exc}")
            graph = None
        if TURTLE_INTENT_PATTERN.search(ttl):
            score += 0.0
        else:
            notes.append("turtle expression lacks core TIO terms")
        if ttl.count("@prefix") >= 4:
            score += 0.0
        else:
            notes.append("turtle expression is missing required prefixes")
    else:
        notes.append("unsupported expression type")
    return {
        "score": round(min(1.0, score), 4),
        "notes": notes,
    }
