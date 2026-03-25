from __future__ import annotations

import json
from typing import Any

from ..config import Settings
from .idan_loader import load_idan_documents
from .oas_parser import extract_oas_assets
from .postman_parser import extract_postman_assets
from .seed_loader import load_seed_records
from .tr290_extractor import extract_tr290_documents


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
