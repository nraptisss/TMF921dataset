from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REALISM_ANCHOR_SOURCES = {"seed", "oas_example", "idan_reference"}


def score_realism(intent_payload: dict[str, Any], retrieved_context: list[dict[str, Any]]) -> dict[str, Any]:
    if not retrieved_context:
        return {"score": 0.5, "notes": ["no retrieved context provided"]}
    intent_text = __import__("json").dumps(intent_payload, sort_keys=True)
    context_texts = [row["text"] for row in retrieved_context]
    vectorizer = CountVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([intent_text, *context_texts])
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    best = float(similarities.max()) if len(similarities) else 0.0
    source_types = {row.get("metadata", {}).get("source_type") for row in retrieved_context}
    anchored = any(source_type in REALISM_ANCHOR_SOURCES for source_type in source_types)
    # Use actual similarity without artificial boosting when anchored
    score = best
    return {
        "score": max(0.0, min(1.0, score)),
        "notes": [],
    }
