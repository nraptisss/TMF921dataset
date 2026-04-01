from pathlib import Path

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.ingestion.corpus_builder import build_corpus
from tmf921_dataset_gen.rag.embeddings import HashingEmbeddingModel
from tmf921_dataset_gen.rag.retriever import BalancedRetriever
from tmf921_dataset_gen.rag.vector_store import ChromaVectorStore


def test_hashing_embedder_returns_fixed_length_vectors() -> None:
    embedder = HashingEmbeddingModel(n_features=128)
    vectors = embedder.embed_documents(["hello world", "tmf921 intent"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 128


def test_chroma_vector_store_indexes_and_queries(tmp_path: Path) -> None:
    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "chroma-index"
    store = ChromaVectorStore(settings, collection_name="test_store")
    embedder = HashingEmbeddingModel(n_features=128)
    documents = [
        {"id": "a", "text": "energy optimization for 6g core", "source_type": "seed", "title": "a", "metadata": {}},
        {"id": "b", "text": "predictive assurance for slices", "source_type": "tr290_docx", "title": "b", "metadata": {}},
    ]
    count = store.index_documents(documents, embedder)
    assert count == 2
    results = store.query("energy saving in 6g core", embedder, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "a"


def test_balanced_retriever_returns_mixed_sources(tmp_path: Path) -> None:
    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "balanced-index"
    build_corpus(settings)
    retriever = BalancedRetriever(settings, embedder=HashingEmbeddingModel(n_features=256))
    indexed = retriever.build()
    assert indexed > 100
    results = retriever.retrieve("energy efficient 6g intent for network slices", top_k=6, per_source=1)
    assert len(results) == 6
    source_types = {row["metadata"]["source_type"] for row in results}
    assert "seed" in source_types
    assert any(source.startswith("tr290") for source in source_types)
