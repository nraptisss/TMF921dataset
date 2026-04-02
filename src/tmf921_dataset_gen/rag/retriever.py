from __future__ import annotations

import json
from pathlib import Path
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


SOURCE_PRIORITY = [
    "seed",
    "oas_example",
    "oas_schema",
    "tr290_docx",
    "tr290_pdf",
    "tr290_markdown",
    "idan_reference",
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
        if not force_rebuild:
            expected_count = len(documents)
            current_count = self.store.count()
            if expected_count > 0 and current_count == expected_count:
                return current_count
        return self.store.index_documents(documents, self.embedder)

    def retrieve(self, query_text: str, top_k: int = 8, per_source: int = 2) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for source_type in SOURCE_PRIORITY:
            source_results = self.store.query(
                query_text,
                self.embedder,
                top_k=per_source,
                where={"source_type": source_type},
            )
            for row in source_results:
                if row["id"] in seen_ids:
                    continue
                results.append(row)
                seen_ids.add(row["id"])
                if len(results) >= top_k:
                    return results[:top_k]
        if len(results) < top_k:
            for row in self.store.query(query_text, self.embedder, top_k=top_k * 2):
                if row["id"] in seen_ids:
                    continue
                results.append(row)
                seen_ids.add(row["id"])
                if len(results) >= top_k:
                    break
        return results[:top_k]
