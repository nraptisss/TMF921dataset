from pathlib import Path

from tmf921_dataset_gen.dashboard.streamlit_app import load_dataset_preview
from tmf921_dataset_gen.models.dataset import DatasetMetadata, DatasetRecord
from tmf921_dataset_gen.export.hf_exporter import export_dataset_records
from tmf921_dataset_gen.config import Settings


def test_exporter_writes_manifest_and_jsonl(tmp_path: Path) -> None:
    settings = Settings.from_env(Path.cwd())
    record = DatasetRecord(
        nl_intent="Ensure latency below 1 ms.",
        tmf921_intent={"@type": "Intent", "name": "test", "expression": {"@type": "JsonLdExpression", "iri": "x", "expressionValue": {"@context": {"icm": "x"}}}},
        serialization="json-ld",
        metadata=DatasetMetadata(
            taxonomy_category="service/urllc/predictive_assurance",
            taxonomy_target={"layer": "service", "traffic_profile": "urllc", "scenario_family": "predictive_assurance"},
            kpis={"latency_ms": 1},
            quality_score=0.95,
            tio_compliance=1.0,
            generation_timestamp="2026-03-25T00:00:00Z",
        ),
    )
    report = export_dataset_records(settings, [record], tmp_path)
    assert report.jsonl_path.exists()
    assert report.manifest_path.exists()
    assert report.release_audit_path.exists()
    manifest = __import__("json").loads(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["quality_metrics"]["bias_report"]["bias_score"] == 0.0


def test_dashboard_preview_reads_export(tmp_path: Path) -> None:
    settings = Settings.from_env(Path.cwd())
    record = DatasetRecord(
        nl_intent="Ensure latency below 1 ms.",
        tmf921_intent={"@type": "Intent", "name": "test", "expression": {"@type": "JsonLdExpression", "iri": "x", "expressionValue": {"@context": {"icm": "x"}}}},
        serialization="json-ld",
        metadata=DatasetMetadata(
            taxonomy_category="service/urllc/predictive_assurance",
            taxonomy_target={"layer": "service", "traffic_profile": "urllc", "scenario_family": "predictive_assurance"},
            kpis={"latency_ms": 1},
            quality_score=0.95,
            tio_compliance=1.0,
            generation_timestamp="2026-03-25T00:00:00Z",
        ),
    )
    export_dataset_records(settings, [record], tmp_path)
    manifest, rows = load_dataset_preview(tmp_path)
    assert manifest["record_count"] == 1
    assert rows[0]["nl_intent"] == "Ensure latency below 1 ms."
