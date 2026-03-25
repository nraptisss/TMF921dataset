from __future__ import annotations

import json
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import Settings
from .embeddings import EmbeddingBackend


class ChromaVectorStore:
    def __init__(self, settings: Settings, collection_name: str = "tmf921_corpus") -> None:
        self.settings = settings
        self.collection_name = collection_name
        self.settings.vector_index_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.settings.vector_index_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_documents(self, documents: list[dict[str, Any]], embedder: EmbeddingBackend) -> int:
        if not documents:
            return 0
        self.reset()
        batch_size = 32
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            texts = [doc["text"] for doc in batch]
            embeddings = embedder.embed_documents(texts)
            ids = [doc.get("chunk_id") or doc["id"] for doc in batch]
            metadatas = [
                {
                    "source_type": doc.get("source_type", "unknown"),
                    "title": doc.get("title", ""),
                    "document_id": doc.get("id", ""),
                    "metadata_json": json.dumps(doc.get("metadata", {}), sort_keys=True),
                }
                for doc in batch
            ]
            self.collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        return len(documents)

    def query(self, query_text: str, embedder: EmbeddingBackend, top_k: int = 8, where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[embedder.embed_query(query_text)],
            n_results=top_k,
            where=where,
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        rows: list[dict[str, Any]] = []
        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            rows.append(
                {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )
        return rows
