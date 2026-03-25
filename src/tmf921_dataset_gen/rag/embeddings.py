from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from sklearn.feature_extraction.text import HashingVectorizer

LOGGER = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@dataclass(slots=True)
class HashingEmbeddingModel:
    n_features: int = 1024
    vectorizer: HashingVectorizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm="l2",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedding_backend(model_name: str, allow_fallback: bool = True, prefer_hashing: bool = False) -> EmbeddingBackend:
    if prefer_hashing:
        return HashingEmbeddingModel()
    try:
        return SentenceTransformerEmbeddingModel(model_name)
    except Exception as exc:  # pragma: no cover - depends on local model availability
        if not allow_fallback:
            raise
        LOGGER.warning("Falling back to HashingVectorizer embeddings: %s", exc)
        return HashingEmbeddingModel()
