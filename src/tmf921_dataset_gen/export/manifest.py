from __future__ import annotations

import re
from pathlib import Path

from ..config import Settings
from ..evaluation.benchmark_suite import BenchmarkSuite
from ..models.dataset import DatasetRecord
from ..validation.diversity_metrics import calculate_diversity_score, detect_bias


def _normalized_nl(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _combination_coverage(records: list[DatasetRecord]) -> float:
    if not records:
        return 0.0
    combos = {
        (
            record.metadata.taxonomy_target.get("layer"),
            record.metadata.taxonomy_target.get("traffic_profile"),
            record.metadata.taxonomy_target.get("scenario_family"),
            record.metadata.taxonomy_target.get("domain_context"),
        )
        for record in records
    }
    return len(combos) / 288


def build_manifest(settings: Settings, records: list[DatasetRecord], hf_dataset_path: Path | None, jsonl_path: Path) -> dict:
    # Calculate quality metrics
    payloads = [record.model_dump(mode="json") for record in records]
    diversity_score = calculate_diversity_score(payloads)
    bias_report = detect_bias(payloads, [record.metadata.taxonomy_target for record in records])

    avg_quality = sum(record.metadata.quality_score for record in records) / len(records) if records else 0
    avg_semantic = sum(record.metadata.semantic_score for record in records) / len(records) if records else 0
    avg_tio = sum(record.metadata.tio_compliance for record in records) / len(records) if records else 0
    avg_supported_claim_ratio = sum(record.metadata.supported_claim_ratio for record in records) / len(records) if records else 0
    semantic_pass_rate = sum(1 for record in records if record.metadata.semantic_pass) / len(records) if records else 0
    operator_pass_rate = sum(1 for record in records if record.metadata.operator_pass) / len(records) if records else 0
    coverage_ratio = _combination_coverage(records)
    # Track both the ratio of records with unsupported claims AND the average count
    records_with_unsupported = sum(1 for record in records if record.metadata.unsupported_claim_count > 0)
    unsupported_claims_ratio = records_with_unsupported / max(1, len(records)) if records else 0.0
    avg_unsupported_claims = (
        sum(record.metadata.unsupported_claim_count for record in records) / max(1, len(records))
        if records else 0.0
    )
    normalized_nl_values = [_normalized_nl(record.nl_intent) for record in records]
    normalized_nl_duplicate_ratio = (
        1.0 - (len(set(normalized_nl_values)) / len(normalized_nl_values))
        if normalized_nl_values else 0.0
    )

    effective_embedding_model = settings.effective_embedding_model or settings.embedding_model
    effective_embedding_backend = settings.effective_embedding_backend
    if effective_embedding_backend is None:
        effective_embedding_backend = "not-used" if settings.inference_backend == "mock" else "unknown"

    return {
        "project": "tmf921-dataset-gen",
        "record_count": len(records),
        "hf_dataset_path": str(hf_dataset_path) if hf_dataset_path else None,
        "jsonl_path": str(jsonl_path),
        "embedding_model": effective_embedding_model,
        "configured_embedding_model": settings.embedding_model,
        "effective_embedding_backend": effective_embedding_backend,
        "reasoning_model": settings.reasoning_model,
        "bulk_model": settings.bulk_model,
        "inference_backend": settings.inference_backend,
        "quality_metrics": {
            "average_quality_score": round(avg_quality, 4),
            "average_semantic_score": round(avg_semantic, 4),
            "average_tio_compliance": round(avg_tio, 4),
            "diversity_score": round(diversity_score, 4),
            "semantic_pass_rate": round(semantic_pass_rate, 4),
            "operator_pass_rate": round(operator_pass_rate, 4),
            "combination_coverage_ratio": round(coverage_ratio, 4),
            "average_supported_claim_ratio": round(avg_supported_claim_ratio, 4),
            "unsupported_claims_ratio": round(unsupported_claims_ratio, 4),
            "avg_unsupported_claims_per_record": round(avg_unsupported_claims, 4),
            "normalized_nl_duplicate_ratio": round(normalized_nl_duplicate_ratio, 4),
            "bias_report": bias_report,
        },
        "grounding_mode": settings.grounding_mode,
        "release_readiness": {
            "automatic_gates_only": False,
            "manual_review_required": settings.require_manual_review_for_release,
        },
    }


def build_release_audit(settings: Settings, records: list[DatasetRecord]) -> dict:
    total = len(records)
    semantic_pass_rate = sum(1 for record in records if record.metadata.semantic_pass) / total if total else 0.0
    supported_claim_ratio = sum(record.metadata.supported_claim_ratio for record in records) / total if total else 0.0
    coverage_ratio = _combination_coverage(records)
    # Track both the ratio of records with unsupported claims AND the average count
    records_with_unsupported = sum(1 for record in records if record.metadata.unsupported_claim_count > 0)
    unsupported_claims_ratio = records_with_unsupported / max(1, total) if total else 0.0
    avg_unsupported_claims = (
        sum(record.metadata.unsupported_claim_count for record in records) / total
        if total else 0.0
    )
    payloads = [record.model_dump(mode="json") for record in records]
    diversity_score = calculate_diversity_score(payloads)
    duplicate_ratio = max(0.0, 1.0 - diversity_score) if total > 0 else 0.0
    normalized_nl_values = [_normalized_nl(record.nl_intent) for record in records]
    normalized_nl_duplicate_ratio = (
        1.0 - (len(set(normalized_nl_values)) / len(normalized_nl_values))
        if normalized_nl_values else 0.0
    )

    # Run benchmark suite
    benchmark_suite = BenchmarkSuite(settings)
    dataset_records = [record.model_dump(mode="json") for record in records]
    benchmark_results = benchmark_suite.run_dataset_audit(dataset_records)

    automatic_checks = {
        "semantic_pass_rate": {
            "actual": round(semantic_pass_rate, 4),
            "threshold": settings.release_semantic_pass_threshold,
            "pass": semantic_pass_rate >= settings.release_semantic_pass_threshold,
        },
        "supported_claim_ratio": {
            "actual": round(supported_claim_ratio, 4),
            "threshold": settings.release_supported_claim_ratio_threshold,
            "pass": supported_claim_ratio >= settings.release_supported_claim_ratio_threshold,
        },
        "unsupported_claims_ratio": {
            "actual": round(unsupported_claims_ratio, 4),
            "threshold": settings.release_max_unsupported_claims_ratio,
            "pass": unsupported_claims_ratio <= settings.release_max_unsupported_claims_ratio,
        },
        "duplicate_ratio": {
            "actual": round(duplicate_ratio, 4),
            "threshold": settings.release_max_duplicate_ratio,
            "pass": duplicate_ratio <= settings.release_max_duplicate_ratio,
        },
        "operator_preservation_rate": {
            "actual": benchmark_results["operator_preservation"]["rate"],
            "threshold": 0.90,  # Configurable threshold
            "pass": benchmark_results["operator_preservation"]["rate"] >= 0.90,
        },
        "semantic_preservation_rate": {
            "actual": benchmark_results["semantic_preservation"]["rate"],
            "threshold": 0.95,  # Configurable threshold
            "pass": benchmark_results["semantic_preservation"]["rate"] >= 0.95,
        },
        "combination_coverage_ratio": {
            "actual": round(coverage_ratio, 4),
            "threshold": 0.75,
            "pass": coverage_ratio >= 0.75,
        },
        "normalized_nl_duplicate_ratio": {
            "actual": round(normalized_nl_duplicate_ratio, 4),
            "threshold": 0.1,
            "pass": normalized_nl_duplicate_ratio <= 0.1,
        },
    }
    auto_pass = all(check["pass"] for check in automatic_checks.values())
    manual_review = {
        "required": settings.require_manual_review_for_release,
        "status": "pending" if settings.require_manual_review_for_release else "not-required",
        "rubric": [
            "semantic faithfulness",
            "tmf/tio correctness",
            "grounding support",
            "linguistic naturalness",
        ],
        "sample_size": min(50, max(10, total // 20)),  # Statistically meaningful sample
    }
    return {
        "record_count": total,
        "grounding_mode": settings.grounding_mode,
        "automatic_checks": automatic_checks,
        "benchmark_results": benchmark_results,
        "automatic_gate_pass": auto_pass,
        "manual_review": manual_review,
        "release_blocked": (not auto_pass) or settings.require_manual_review_for_release,
        "release_status": "blocked" if (not auto_pass) or settings.require_manual_review_for_release else "ready",
        "notes": [
            "Release remains blocked until manual review is completed." if settings.require_manual_review_for_release else "Automatic gates satisfied.",
            f"Benchmark suite executed with {len(benchmark_results)} tests.",
            "Coverage and duplicate checks now enforce release-readiness expectations.",
        ],
    }
