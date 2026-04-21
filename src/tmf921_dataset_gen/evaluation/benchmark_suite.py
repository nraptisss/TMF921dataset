"""Evaluation and benchmark suite for TMF921 dataset generation."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ..config import Settings
from ..validation.semantic_frame import verify_semantic_alignment


class BenchmarkSuite:
    """Benchmark suite for evaluating TMF921 dataset generation quality."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.evaluation_set_path = settings.repo.root / "benchmarks" / "evaluation_set.jsonl"
        self.benchmarks_dir = settings.repo.root / "benchmarks"

    def load_evaluation_set(self) -> list[dict[str, Any]]:
        """Load the hand-audited evaluation set."""
        if not self.evaluation_set_path.exists():
            raise FileNotFoundError(f"Evaluation set not found at {self.evaluation_set_path}")
        
        evaluation_cases = []
        with open(self.evaluation_set_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    evaluation_cases.append(json.loads(line))
        return evaluation_cases

    def run_operator_preservation_test(self, dataset_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Test operator preservation between NL intent and payload."""
        total = len(dataset_records)
        passed = 0
        failures = []
        
        for record in dataset_records:
            nl_intent = record["nl_intent"]
            intent_frame = record.get("metadata", {}).get("intent_frame")
            payload = record["tmf921_intent"]
            
            if not intent_frame:
                continue
                
            symbolic = verify_semantic_alignment(nl_intent, intent_frame, payload)
            if symbolic["operator_pass"]:
                passed += 1
            else:
                failures.append({
                    "record_id": record.get("id", "unknown"),
                    "notes": symbolic["notes"]
                })
        
        return {
            "metric": "operator_preservation_rate",
            "passed": passed,
            "total": total,
            "rate": passed / total if total > 0 else 0,
            "failures": failures[:10]  # Sample failures
        }

    def run_missing_constraint_detection_test(self, dataset_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Test detection of missing constraints.

        Extracts constraints from the intent_frame (the ground truth) and
        verifies they appear in the payload's expressionValue (JSON-LD @graph
        or Turtle text).
        """
        total = len(dataset_records)
        passed = 0
        failures = []

        # Metric key to JSON-LD predicate mapping
        METRIC_PREDICATES = {
            "latency_ms": "met:latency",
            "throughput_mbps": "met:throughput",
            "throughput_gbps": "met:throughput",
            "reliability_percent": "met:reliability",
            "availability_percent": "met:availability",
            "energy_kwh": "met:energyConsumption",
            "device_count": "sli:deviceCount",
            "delivery_ratio_percent": "met:packetDeliveryRatio",
            "reaction_time_ms": "met:reactionTime",
            "reporting_interval_seconds": "icm:reportingInterval",
        }

        for record in dataset_records:
            intent_frame = record.get("metadata", {}).get("intent_frame")
            payload = record["tmf921_intent"]

            if not intent_frame:
                continue

            expected_constraints = {c["metric"] for c in intent_frame.get("constraints", [])}

            # Extract constraint predicates from the JSON-LD expression
            expression = payload.get("expression", {})
            expression_value = expression.get("expressionValue", {})
            payload_predicates = set()

            if isinstance(expression_value, dict) and "@graph" in expression_value:
                for node in expression_value["@graph"]:
                    if isinstance(node, dict):
                        params = node.get("icm:params", {})
                        if isinstance(params, dict):
                            for key in params:
                                # Map predicate back to metric key
                                for metric, pred in METRIC_PREDICATES.items():
                                    if pred == key:
                                        payload_predicates.add(metric)

            # Also check the context field as a fallback
            context_text = payload.get("context", "")
            for metric in expected_constraints:
                if metric in context_text:
                    payload_predicates.add(metric)

            missing = expected_constraints - payload_predicates
            if not missing:
                passed += 1
            else:
                failures.append({
                    "record_id": record.get("id", "unknown"),
                    "missing_constraints": list(missing)
                })

        return {
            "metric": "missing_constraint_detection_rate",
            "passed": passed,
            "total": total,
            "rate": passed / total if total > 0 else 0,
            "failures": failures[:10]
        }

    def run_semantic_preservation_test(self, evaluation_cases: list[dict[str, Any]], dataset_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Test semantic preservation against gold evaluation set.

        Match candidates by current taxonomy fields and metric overlap first,
        then validate exact constraint/operator preservation.
        """

        total = len(evaluation_cases)
        passed = 0
        failures = []

        for gold_case in evaluation_cases:
            gold_nl = gold_case["nl_intent"]
            gold_frame = gold_case["gold_intent_frame"]
            gold_metrics = {constraint["metric"] for constraint in gold_frame.get("constraints", [])}

            best_match = None
            best_score = -1.0

            for record in dataset_records:
                actual_frame = record.get("metadata", {}).get("intent_frame")
                if not actual_frame:
                    continue
                actual_metrics = {constraint["metric"] for constraint in actual_frame.get("constraints", [])}
                score = 0.0
                if actual_frame.get("traffic_profile") == gold_frame.get("traffic_profile"):
                    score += 2.0
                if actual_frame.get("scenario_family") == gold_frame.get("scenario_family"):
                    score += 2.0
                if actual_frame.get("layer") == gold_frame.get("layer"):
                    score += 1.0
                metric_union = gold_metrics | actual_metrics
                if metric_union:
                    score += len(gold_metrics & actual_metrics) / len(metric_union)
                if actual_frame.get("domain_context") == gold_frame.get("domain_context"):
                    score += 0.5
                if score > best_score:
                    best_score = score
                    best_match = record

            if best_match is None or best_score < 3.0:
                failures.append({
                    "nl_intent": gold_nl[:100] + "...",
                    "reason": "no matching record found in dataset",
                    "gold_target": {
                        "layer": gold_frame.get("layer"),
                        "traffic_profile": gold_frame.get("traffic_profile"),
                        "scenario_family": gold_frame.get("scenario_family"),
                    },
                })
                continue

            actual_frame = best_match.get("metadata", {}).get("intent_frame")

            if not actual_frame:
                failures.append({
                    "nl_intent": gold_nl[:100] + "...",
                    "reason": "matched record missing intent_frame",
                    "match_score": round(best_score, 3),
                })
                continue

            gold_constraints = {c["metric"]: c for c in gold_frame.get("constraints", [])}
            actual_constraints = {c["metric"]: c for c in actual_frame.get("constraints", [])}

            constraint_match = True
            for metric, gold_c in gold_constraints.items():
                actual_c = actual_constraints.get(metric)
                if not actual_c:
                    constraint_match = False
                    break
                if gold_c["operator"] != actual_c["operator"]:
                    constraint_match = False
                    break

            if constraint_match:
                passed += 1
            else:
                failures.append({
                    "nl_intent": gold_nl[:100] + "...",
                    "gold_constraints": {m: c["operator"] for m, c in gold_constraints.items()},
                    "actual_constraints": {m: c["operator"] for m, c in actual_constraints.items()},
                    "match_score": round(best_score, 3),
                })

        return {
            "metric": "semantic_preservation_rate",
            "passed": passed,
            "total": total,
            "rate": passed / total if total > 0 else 0,
            "failures": failures[:10]
        }

    def run_dataset_audit(self, dataset_records: list[dict[str, Any]]) -> dict[str, Any]:
        """Run full dataset audit with all benchmark suites."""
        evaluation_cases = self.load_evaluation_set()
        
        results = {
            "operator_preservation": self.run_operator_preservation_test(dataset_records),
            "missing_constraint_detection": self.run_missing_constraint_detection_test(dataset_records),
            "semantic_preservation": self.run_semantic_preservation_test(evaluation_cases, dataset_records),
            "dataset_stats": {
                "total_records": len(dataset_records),
                "grounding_modes": {},
                "semantic_pass_rate": 0,
                "unsupported_claim_rate": 0,
                "duplicate_rate": 0,
            }
        }
        
        # Aggregate dataset stats
        semantic_passes = 0
        unsupported_claims = 0
        grounding_modes = {}
        
        for record in dataset_records:
            metadata = record.get("metadata", {})
            grounding_modes[metadata.get("grounding_mode", "unknown")] = grounding_modes.get(metadata.get("grounding_mode", "unknown"), 0) + 1
            
            if metadata.get("semantic_pass"):
                semantic_passes += 1
            unsupported_claims += metadata.get("unsupported_claim_count", 0)
        
        results["dataset_stats"].update({
            "semantic_pass_rate": semantic_passes / len(dataset_records) if dataset_records else 0,
            "unsupported_claim_rate": unsupported_claims / len(dataset_records) if dataset_records else 0,
            "grounding_modes": grounding_modes,
        })
        
        # Release gate checks
        gates = {
            "semantic_preservation_gate": results["semantic_preservation"]["rate"] >= 0.95,
            "unsupported_claim_gate": results["dataset_stats"]["unsupported_claim_rate"] <= 0.05,
            "operator_preservation_gate": results["operator_preservation"]["rate"] >= 0.90,
            "overall_pass": all([
                results["semantic_preservation"]["rate"] >= 0.95,
                results["dataset_stats"]["unsupported_claim_rate"] <= 0.05,
                results["operator_preservation"]["rate"] >= 0.90,
            ])
        }
        
        results["release_gates"] = gates
        return results
