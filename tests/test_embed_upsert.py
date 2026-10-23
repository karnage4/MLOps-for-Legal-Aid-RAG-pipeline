"""Tests for strategy-isolated embedding caches and Pinecone upserts."""

from types import SimpleNamespace
from unittest.mock import Mock

from src import embed_upsert


def _chunk(chunk_id: str = "fixed-1", text: str | None = None) -> dict:
    return {
        "id": chunk_id,
        "source": "law.txt",
        "strategy": "fixed",
        "chunk_index": 0,
        "text": text or "A complete fixed legal chunk that must remain untruncated.",
    }


def _embedding() -> list[float]:
    return [0.01] * embed_upsert.EMBEDDING_DIM


def test_fixed_profile_is_isolated_from_recursive_defaults():
    recursive = embed_upsert.STRATEGY_PROFILES["recursive"]
    fixed = embed_upsert.STRATEGY_PROFILES["fixed"]

    assert recursive["namespace"] is None
    assert recursive["cache_file"].name == "embeddings_cache.json"
    assert fixed["namespace"] == "fixed"
    assert fixed["chunks_file"].name == "chunks_fixed.json"
    assert fixed["cache_file"].name == "embeddings_fixed.json"


def test_build_vectors_preserves_full_text_and_validates_dimensions():
    long_text = "inheritance " * 400
    chunk = _chunk(text=long_text)

    vectors = embed_upsert.build_pinecone_vectors([chunk], [_embedding()])

    assert vectors[0]["metadata"]["text"] == long_text
    assert vectors[0]["metadata"]["strategy"] == "fixed"
    assert len(vectors[0]["values"]) == embed_upsert.EMBEDDING_DIM


def test_embedding_cache_uses_the_requested_strategy_path(tmp_path):
    chunks = [_chunk()]
    embeddings = [_embedding()]
    cache_path = tmp_path / "embeddings_fixed.json"

    embed_upsert.save_embedding_cache(chunks, embeddings, cache_path)

    assert embed_upsert.load_embedding_cache(chunks, cache_path) == embeddings


def test_upsert_passes_fixed_namespace_and_requires_exact_count():
    vector = embed_upsert.build_pinecone_vectors([_chunk()], [_embedding()])[0]
    index = Mock()
    index.describe_index_stats.return_value = SimpleNamespace(
        namespaces={"fixed": SimpleNamespace(vector_count=1)}
    )

    count = embed_upsert.upsert_to_pinecone(index, [vector], namespace="fixed")

    index.upsert.assert_called_once_with(vectors=[vector], namespace="fixed")
    assert count == 1


def test_namespace_sample_self_query_checks_strategy_and_text():
    vector = embed_upsert.build_pinecone_vectors([_chunk()], [_embedding()])[0]
    index = Mock()
    index.query.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(id=vector["id"], metadata=vector["metadata"]),
        ]
    )

    sample_ids = embed_upsert.verify_namespace_samples(
        index,
        [vector],
        namespace="fixed",
        expected_strategy="fixed",
    )

    index.query.assert_called_once_with(
        vector=vector["values"],
        top_k=1,
        include_metadata=True,
        namespace="fixed",
    )
    assert sample_ids == [vector["id"]]
