"""Prepare and validate strategy-specific BM25 corpora from chunk artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


REQUIRED_CHUNK_FIELDS = {"id", "source", "strategy", "chunk_index", "text"}
DEFAULT_ARTIFACTS = {
    "fixed": (
        Path("output/chunks_fixed.json"),
        Path("output/bm25_corpus_fixed.json"),
    ),
    "recursive": (
        Path("output/chunks_recursive.json"),
        Path("output/bm25_corpus.json"),
    ),
}


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON chunk list and reject a missing or malformed artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Chunk artifact not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError(f"Chunk artifact must contain a JSON list: {path}")
    return chunks


def validate_chunks(
    chunks: Sequence[Dict[str, Any]],
    expected_strategy: str,
) -> Dict[str, Any]:
    """Validate chunk schema, ranges, strategy, content, and ID uniqueness."""
    if expected_strategy not in DEFAULT_ARTIFACTS:
        raise ValueError(f"Unsupported chunking strategy: {expected_strategy}")
    if not chunks:
        raise ValueError("Chunk collection must not be empty")

    ids = []
    sources = set()
    text_lengths = []

    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {position} must be a JSON object")

        missing = REQUIRED_CHUNK_FIELDS.difference(chunk)
        if missing:
            raise ValueError(
                f"Chunk {position} is missing fields: {', '.join(sorted(missing))}"
            )

        chunk_id = chunk["id"]
        source = chunk["source"]
        strategy = chunk["strategy"]
        chunk_index = chunk["chunk_index"]
        text = chunk["text"]

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Chunk {position} has an empty or invalid id")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Chunk {position} has an empty or invalid source")
        if strategy != expected_strategy:
            raise ValueError(
                f"Chunk {position} has strategy {strategy!r}; "
                f"expected {expected_strategy!r}"
            )
        if (
            not isinstance(chunk_index, int)
            or isinstance(chunk_index, bool)
            or chunk_index < 0
        ):
            raise ValueError(f"Chunk {position} has an invalid chunk_index")
        if not isinstance(text, str) or len(text.strip()) < 30:
            raise ValueError(
                f"Chunk {position} text must contain at least 30 characters"
            )

        ids.append(chunk_id)
        sources.add(source)
        text_lengths.append(len(text.strip()))

    duplicate_count = len(ids) - len(set(ids))
    if duplicate_count:
        raise ValueError(f"Chunk collection contains {duplicate_count} duplicate IDs")

    return {
        "strategy": expected_strategy,
        "chunk_count": len(chunks),
        "unique_id_count": len(set(ids)),
        "source_count": len(sources),
        "min_text_chars": min(text_lengths),
        "max_text_chars": max(text_lengths),
        "average_text_chars": round(sum(text_lengths) / len(text_lengths), 2),
    }


def build_bm25_data(
    chunks: Sequence[Dict[str, Any]],
    expected_strategy: str,
) -> tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Return an aligned BM25 ID/text payload plus its validation summary."""
    report = validate_chunks(chunks, expected_strategy)
    bm25_data = {
        "ids": [chunk["id"] for chunk in chunks],
        "texts": [chunk["text"] for chunk in chunks],
    }
    validate_bm25_alignment(chunks, bm25_data)
    return bm25_data, report


def validate_bm25_alignment(
    chunks: Sequence[Dict[str, Any]],
    bm25_data: Dict[str, Any],
) -> None:
    """Require exact positional equality between chunk and BM25 artifacts."""
    if set(bm25_data) != {"ids", "texts"}:
        raise ValueError("BM25 data must contain exactly 'ids' and 'texts'")
    if bm25_data["ids"] != [chunk["id"] for chunk in chunks]:
        raise ValueError("BM25 IDs are not aligned with the source chunks")
    if bm25_data["texts"] != [chunk["text"] for chunk in chunks]:
        raise ValueError("BM25 texts are not aligned with the source chunks")


def file_sha256(path: Path) -> str:
    """Return a reproducibility checksum without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_bm25_artifact(
    chunks_path: Path,
    output_path: Path,
    expected_strategy: str,
) -> Dict[str, Any]:
    """Build, persist, reload, and validate one strategy-specific BM25 corpus."""
    chunks = load_chunks(chunks_path)
    bm25_data, report = build_bm25_data(chunks, expected_strategy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(bm25_data, file, ensure_ascii=False, indent=2)

    with output_path.open("r", encoding="utf-8") as file:
        persisted_data = json.load(file)
    validate_bm25_alignment(chunks, persisted_data)

    return {
        **report,
        "chunks_path": str(chunks_path),
        "output_path": str(output_path),
        "chunks_sha256": file_sha256(chunks_path),
        "bm25_sha256": file_sha256(output_path),
        "alignment_valid": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a validated BM25 corpus for one chunking strategy."
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(DEFAULT_ARTIFACTS),
        default="fixed",
    )
    parser.add_argument("--chunks-file", type=Path)
    parser.add_argument("--output-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_chunks, default_output = DEFAULT_ARTIFACTS[args.strategy]
    report = prepare_bm25_artifact(
        args.chunks_file or default_chunks,
        args.output_file or default_output,
        args.strategy,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
