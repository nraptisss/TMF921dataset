from __future__ import annotations

import hashlib
import json
from typing import Any

from ..config import Settings
from ..ingestion.corpus_builder import build_corpus
from .embeddings import (
    EmbeddingBackend,
    HashingEmbeddingModel,
    SentenceTransformerEmbeddingModel,
    build_embedding_backend,
)
from .vector_store import ChromaVectorStore


GROUNDING_SOURCE_PRIORITY = [
    "seed",
    "tr290_docx",
    "tr290_pdf",
    "tr290_markdown",
    "idan_reference",
]

REFERENCE_SOURCE_PRIORITY = [
    "oas_example",
    "oas_schema",
    "postman_operation",
]


class BalancedRetriever:
    def __init__(self, settings: Settings, embedder: EmbeddingBackend | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or build_embedding_backend(
            settings.resolve_embedding_model(),
            prefer_hashing=settings.inference_backend == "mock",
            local_files_only=settings.is_local_model_backend and settings.local_files_only,
        )
        self._record_effective_embedding()
        self.store = ChromaVectorStore(settings)

    def _record_effective_embedding(self) -> None:
        if isinstance(self.embedder, HashingEmbeddingModel):
            self.settings.effective_embedding_backend = "hashing-vectorizer"
            self.settings.effective_embedding_model = "HashingVectorizer"
            return
        if isinstance(self.embedder, SentenceTransformerEmbeddingModel):
            self.settings.effective_embedding_backend = "sentence-transformers"
            self.settings.effective_embedding_model = self.embedder.model_name
            return
        self.settings.effective_embedding_backend = type(self.embedder).__name__
        self.settings.effective_embedding_model = type(self.embedder).__name__

    def build(self, force_rebuild: bool = False) -> int:
        chunks_path = self.settings.repo.normalized_dir / "corpus" / "corpus_chunks.jsonl"
        if chunks_path.exists():
            documents = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            build_corpus(self.settings)
            documents = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        index_manifest_path = self.settings.vector_index_dir / "index_manifest.json"
        expected_manifest = self._build_index_manifest(documents)
        if not force_rebuild and index_manifest_path.exists():
            current_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
            expected_count = len(documents)
            current_count = self.store.count()
            if expected_count > 0 and current_count == expected_count and current_manifest == expected_manifest:
                return current_count
        indexed_count = self.store.index_documents(documents, self.embedder)
        index_manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
        return indexed_count

    def _build_index_manifest(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        source_counts: dict[str, int] = {}
        digest = hashlib.sha256()
        for document in documents:
            source_type = document.get("source_type", "unknown")
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
            digest.update((document.get("chunk_id") or document.get("id") or "").encode("utf-8"))
            digest.update(document.get("text", "").encode("utf-8"))
        return {
            "collection_name": self.store.collection_name,
            "document_count": len(documents),
            "effective_embedding_backend": self.settings.effective_embedding_backend,
            "effective_embedding_model": self.settings.effective_embedding_model,
            "corpus_sha256": digest.hexdigest(),
            "source_counts": source_counts,
        }

    def retrieve(self, query_text: str, top_k: int = 8, per_source: int | None = None) -> list[dict[str, Any]]:
        source_caps = {
            "seed": 2,
            "tr290_docx": 2,
            "tr290_pdf": 2,
            "tr290_markdown": 2,
            "idan_reference": 2,
            "oas_example": 1,
            "oas_schema": 1,
            "postman_operation": 1,
        }
        if per_source is not None:
            for source_type in source_caps:
                source_caps[source_type] = max(1, per_source)

        candidate_rows: list[dict[str, Any]] = []
        for source_type in [*GROUNDING_SOURCE_PRIORITY, *REFERENCE_SOURCE_PRIORITY]:
            source_cap = source_caps.get(source_type, 1)
            candidate_rows.extend(
                self.store.query(
                    query_text,
                    self.embedder,
                    top_k=max(source_cap * 3, 3),
                    where={"source_type": source_type},
                )
            )

        candidate_rows.extend(self.store.query(query_text, self.embedder, top_k=max(top_k * 3, 12)))
        candidate_rows.sort(key=lambda row: float(row.get("distance", 1.0)))

        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        per_source_counts: dict[str, int] = {}
        for row in candidate_rows:
            if row["id"] in seen_ids:
                continue
            source_type = row.get("metadata", {}).get("source_type", "unknown")
            if per_source_counts.get(source_type, 0) >= source_caps.get(source_type, 2):
                continue
            results.append(row)
            seen_ids.add(row["id"])
            per_source_counts[source_type] = per_source_counts.get(source_type, 0) + 1
            if len(results) >= top_k:
                break
        return results[:top_k]
