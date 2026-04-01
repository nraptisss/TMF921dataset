from __future__ import annotations

import json
from typing import Any

from ..config import Settings
from .idan_loader import load_idan_documents
from .oas_parser import extract_oas_assets
from .postman_parser import extract_postman_assets
from .seed_loader import load_seed_records
from .tr290_extractor import extract_tr290_documents


def _fuzzy_deduplicate(corpus: list[dict[str, Any]], similarity_threshold: float = 0.8) -> list[dict[str, Any]]:
    """Remove duplicate documents based on text similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if len(corpus) <= 1:
        return corpus

    texts = [doc["text"] for doc in corpus]
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        to_remove = set()
        for i in range(len(corpus)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(corpus)):
                if similarity_matrix[i, j] > similarity_threshold:
                    # Prefer shorter or earlier source
                    if len(texts[i]) > len(texts[j]):
                        to_remove.add(i)
                        break
                    else:
                        to_remove.add(j)
        return [doc for idx, doc in enumerate(corpus) if idx not in to_remove]
    except ValueError:
        # If vectorization fails, return original
        return corpus


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 120) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_corpus(settings: Settings) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    corpus.extend(extract_oas_assets(settings))
    corpus.extend(extract_postman_assets(settings))
    corpus.extend(load_seed_records(settings))
    corpus.extend(extract_tr290_documents(settings))
    corpus.extend(load_idan_documents(settings))

    # Deduplicate corpus
    corpus = _fuzzy_deduplicate(corpus)

    corpus_dir = settings.repo.normalized_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "corpus.jsonl").write_text(
        "\n".join(json.dumps(doc) for doc in corpus),
        encoding="utf-8",
    )

    chunks: list[dict[str, Any]] = []
    for document in corpus:
        text_chunks = chunk_text(document["text"])
        total_chunks = len(text_chunks)
        for index, chunk in enumerate(text_chunks, start=1):
            chunks.append(
                {
                    **document,
                    "chunk_id": f"{document['id']}#chunk-{index}",
                    "text": chunk,
                    "metadata": {
                        **document.get("metadata", {}),
                        "chunk_index": index,
                        "chunk_count": total_chunks,
                    },
                }
            )
    (corpus_dir / "corpus_chunks.jsonl").write_text(
        "\n".join(json.dumps(chunk) for chunk in chunks),
        encoding="utf-8",
    )
    return corpus
