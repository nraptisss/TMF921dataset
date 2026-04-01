from pathlib import Path

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.ingestion.corpus_builder import build_corpus, chunk_text
from tmf921_dataset_gen.ingestion.idan_loader import load_idan_documents
from tmf921_dataset_gen.ingestion.seed_loader import load_seed_records
from tmf921_dataset_gen.ingestion.tr290_extractor import extract_tr290_documents


def test_seed_loader_parses_all_seed_records() -> None:
    settings = Settings.from_env(Path.cwd())
    seeds = load_seed_records(settings)
    assert len(seeds) == 15
    assert seeds[0]["metadata"]["seed_id"] == "seed-001"
    assert seeds[1]["tmf921_intent"]["priority"] == "medium"
    assert (settings.repo.normalized_dir / "seeds" / "seeds.normalized.json").exists()


def test_tr290_extractor_writes_markdown_outputs() -> None:
    settings = Settings.from_env(Path.cwd())
    docs = extract_tr290_documents(settings)
    assert len(docs) >= 2
    assert any("Intent" in doc["text"] for doc in docs)
    assert (settings.repo.normalized_dir / "tr290" / "TR290_Intent_Common_Model_v3.0.0.docx.md").exists()


def test_idan_loader_handles_missing_directory() -> None:
    settings = Settings.from_env(Path.cwd())
    docs = load_idan_documents(settings)
    assert docs == []
    manifest = settings.repo.normalized_dir / "idan" / "manifest.json"
    assert manifest.exists()
    assert "missing" in manifest.read_text(encoding="utf-8")


def test_idan_loader_reads_present_directory(tmp_path: Path) -> None:
    settings = Settings.from_env(Path.cwd())
    idan_dir = tmp_path / "idan-reference"
    idan_dir.mkdir()
    (idan_dir / "example.ttl").write_text("@prefix icm: <http://example.com/icm#> .", encoding="utf-8")
    settings.repo.idan_reference_dir = idan_dir
    docs = load_idan_documents(settings)
    assert len(docs) == 1
    assert docs[0]["source_type"] == "idan_reference"


def test_corpus_builder_writes_corpus_and_chunks() -> None:
    settings = Settings.from_env(Path.cwd())
    corpus = build_corpus(settings)
    assert len(corpus) > 100
    assert (settings.repo.normalized_dir / "corpus" / "corpus.jsonl").exists()
    assert (settings.repo.normalized_dir / "corpus" / "corpus_chunks.jsonl").exists()


def test_chunk_text_splits_long_payloads() -> None:
    chunks = chunk_text("x" * 3000, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
