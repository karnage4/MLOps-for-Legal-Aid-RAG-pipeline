"""Tests for strategy-specific BM25 data preparation and validation."""

import json

import pytest

from src.prepare_bm25 import prepare_bm25_artifact, validate_chunks


def _chunk(chunk_id: str, index: int = 0, strategy: str = "fixed") -> dict:
    return {
        "id": chunk_id,
        "source": "law.txt",
        "strategy": strategy,
        "chunk_index": index,
        "text": "A sufficiently long legal inheritance passage for BM25 testing.",
    }


def test_prepare_bm25_artifact_preserves_exact_id_text_alignment(tmp_path):
    chunks = [_chunk("fixed-1"), _chunk("fixed-2", index=1)]
    chunks_path = tmp_path / "chunks_fixed.json"
    output_path = tmp_path / "bm25_corpus_fixed.json"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

    report = prepare_bm25_artifact(chunks_path, output_path, "fixed")
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == {
        "ids": [chunk["id"] for chunk in chunks],
        "texts": [chunk["text"] for chunk in chunks],
    }
    assert report["chunk_count"] == 2
    assert report["unique_id_count"] == 2
    assert report["alignment_valid"] is True
    assert len(report["chunks_sha256"]) == 64
    assert len(report["bm25_sha256"]) == 64


def test_validate_chunks_rejects_missing_values_and_invalid_ranges():
    missing_source = _chunk("fixed-1")
    missing_source["source"] = ""
    with pytest.raises(ValueError, match="source"):
        validate_chunks([missing_source], "fixed")

    invalid_index = _chunk("fixed-1")
    invalid_index["chunk_index"] = -1
    with pytest.raises(ValueError, match="chunk_index"):
        validate_chunks([invalid_index], "fixed")


def test_validate_chunks_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate IDs"):
        validate_chunks(
            [_chunk("duplicate", 0), _chunk("duplicate", 1)],
            "fixed",
        )


def test_validate_chunks_rejects_strategy_mismatch():
    with pytest.raises(ValueError, match="expected 'fixed'"):
        validate_chunks([_chunk("recursive-1", strategy="recursive")], "fixed")
