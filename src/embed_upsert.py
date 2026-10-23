"""
Step 2 - Embedding + Pinecone Upsert for Legal Inheritance RAG
==============================================================
Reads chunks from Step 1 output, generates embeddings locally using
sentence-transformers (all-MiniLM-L6-v2), then upserts to Pinecone.

Run ONCE offline — do not re-embed on every query.

Usage:
  pip install sentence-transformers pinecone tqdm
  python -m src.embed_upsert --strategy recursive
  python -m src.embed_upsert --strategy fixed

Environment variables required (set these before running):
  PINECONE_API_KEY   — from https://app.pinecone.io → API Keys
"""

import argparse
import os
import json
import time
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from dotenv import load_dotenv

try:
    from src.prepare_bm25 import validate_chunks
except ImportError:  # supports `python src/embed_upsert.py`
    from prepare_bm25 import validate_chunks

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "legal-inheritance-rag"   # must be lowercase, no spaces
PINECONE_CLOUD      = "aws"                     # free tier uses aws
PINECONE_REGION     = "us-east-1"               # free tier region

EMBEDDING_MODEL     = "all-MiniLM-L6-v2"        # 384-dim, fast, good quality
EMBEDDING_DIM       = 384

BATCH_SIZE          = 100   # Pinecone upsert batch limit is 100 vectors
MAX_METADATA_TEXT_BYTES = 30_000

# Preserve all existing recursive defaults. Fixed chunks use an isolated cache
# and Pinecone namespace so overlapping IDs cannot replace recursive vectors.
STRATEGY_PROFILES = {
    "recursive": {
        "chunks_file": Path("./output/chunks_recursive.json"),
        "cache_file": Path("./output/embeddings_cache.json"),
        "namespace": None,
    },
    "fixed": {
        "chunks_file": Path("./output/chunks_fixed.json"),
        "cache_file": Path("./output/embeddings_fixed.json"),
        "namespace": "fixed",
    },
}
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1.  LOAD CHUNKS
# ─────────────────────────────────────────────

def load_chunks(path: Path, expected_strategy: str) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    validate_chunks(chunks, expected_strategy)
    print(f"Loaded {len(chunks):,} chunks from '{path}'")
    return chunks


# ─────────────────────────────────────────────
# 2.  GENERATE EMBEDDINGS
# ─────────────────────────────────────────────

def embed_chunks(chunks: List[Dict], model_name: str) -> List[List[float]]:
    """
    Generate embeddings for all chunks locally.
    Returns a list of float vectors, one per chunk.
    """
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts):,} chunks (this may take a few minutes)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity becomes dot product
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings.tolist()


# ─────────────────────────────────────────────
# 3.  PINECONE SETUP
# ─────────────────────────────────────────────

def init_pinecone_index(api_key: str, index_name: str) -> object:
    """
    Connect to Pinecone and create the index if it doesn't exist yet.
    Uses the free Serverless tier (no pods needed).
    """
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Set your Pinecone API key:\n"
            "  export PINECONE_API_KEY=your_key_here\n"
            "Get it at: https://app.pinecone.io → API Keys"
        )

    print(f"\nConnecting to Pinecone...")
    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing:
        print(f"Creating new index: '{index_name}'")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        # Wait for index to be ready
        print("Waiting for index to be ready...", end="")
        while not pc.describe_index(index_name).status["ready"]:
            print(".", end="", flush=True)
            time.sleep(2)
        print(" ready!")
    else:
        print(f"Index '{index_name}' already exists — connecting.")

    description = pc.describe_index(index_name)
    if description.dimension != EMBEDDING_DIM:
        raise ValueError(
            f"Index dimension is {description.dimension}; expected {EMBEDDING_DIM}"
        )
    if description.metric != "cosine":
        raise ValueError(
            f"Index metric is {description.metric!r}; expected 'cosine'"
        )

    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"Index stats: {stats.total_vector_count:,} vectors currently stored")
    return index


# ─────────────────────────────────────────────
# 4.  UPSERT TO PINECONE
# ─────────────────────────────────────────────

def build_pinecone_vectors(
    chunks: Sequence[Dict],
    embeddings: Sequence[Sequence[float]],
) -> List[Dict]:
    """
    Format chunks + embeddings into Pinecone upsert format.
    Metadata stored per vector:
      - text     : the chunk text (needed for LLM context at query time)
      - source   : original filename
      - strategy : chunking strategy used
      - chunk_index
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk/embedding count mismatch: {len(chunks)} != {len(embeddings)}"
        )

    vectors = []
    safe_ids = set()
    for chunk, embedding in zip(chunks, embeddings):
        # Pinecone requires ASCII vector IDs; strip non-ascii characters (like smart quotes)
        safe_id = chunk["id"].encode("ascii", "ignore").decode("ascii")
        if not safe_id:
            raise ValueError(f"Chunk ID {chunk['id']!r} becomes empty after ASCII cleanup")
        if safe_id in safe_ids:
            raise ValueError(f"Duplicate Pinecone vector ID after ASCII cleanup: {safe_id}")
        safe_ids.add(safe_id)

        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding for {safe_id} has {len(embedding)} dimensions; "
                f"expected {EMBEDDING_DIM}"
            )

        text_bytes = len(chunk["text"].encode("utf-8"))
        if text_bytes > MAX_METADATA_TEXT_BYTES:
            raise ValueError(
                f"Chunk {safe_id} text is {text_bytes} bytes; maximum supported "
                f"metadata text size is {MAX_METADATA_TEXT_BYTES}"
            )

        vectors.append({
            "id": safe_id,
            "values": list(embedding),
            "metadata": {
                "text":        chunk["text"],
                "source":      chunk["source"],
                "strategy":    chunk.get("strategy", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
            },
        })
    return vectors


def _namespace_vector_count(stats: Any, namespace: Optional[str]) -> int:
    """Read one namespace count from Pinecone's object or dict response."""
    namespace_key = namespace or "__default__"
    namespaces = getattr(stats, "namespaces", None)
    if namespaces is None and isinstance(stats, dict):
        namespaces = stats.get("namespaces", {})
    namespaces = namespaces or {}
    entry = namespaces.get(namespace_key)
    if entry is None:
        return 0
    if isinstance(entry, dict):
        return int(entry.get("vector_count", 0))
    return int(getattr(entry, "vector_count", 0))


def wait_for_namespace_count(
    index,
    namespace: Optional[str],
    expected_count: int,
    max_attempts: int = 10,
) -> int:
    """Wait for eventual consistency and require an exact namespace count."""
    observed_count = 0
    for attempt in range(1, max_attempts + 1):
        observed_count = _namespace_vector_count(
            index.describe_index_stats(), namespace
        )
        if observed_count == expected_count:
            return observed_count
        if attempt < max_attempts:
            time.sleep(2)
    raise RuntimeError(
        f"Namespace {namespace or '__default__'!r} contains {observed_count} "
        f"vectors; expected exactly {expected_count}"
    )


def upsert_to_pinecone(
    index,
    vectors: List[Dict],
    namespace: Optional[str] = None,
) -> int:
    """
    Upsert vectors in batches of BATCH_SIZE.
    Pinecone's upsert is idempotent — safe to re-run if interrupted.
    """
    total_batches = math.ceil(len(vectors) / BATCH_SIZE)
    namespace_label = namespace or "__default__"
    print(
        f"\nUpserting {len(vectors):,} vectors into namespace "
        f"'{namespace_label}' in {total_batches} batches..."
    )

    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Upserting", unit="batch"):
        batch = vectors[i : i + BATCH_SIZE]
        if namespace:
            index.upsert(vectors=batch, namespace=namespace)
        else:
            index.upsert(vectors=batch)

    print("Upsert complete!")

    observed_count = wait_for_namespace_count(index, namespace, len(vectors))
    print(f"Namespace '{namespace_label}' contains {observed_count:,} vectors.")
    return observed_count


# ─────────────────────────────────────────────
# 5.  SAVE EMBEDDING CACHE (optional but useful)
# ─────────────────────────────────────────────

def save_embedding_cache(
    chunks: List[Dict],
    embeddings: List[List[float]],
    cache_path: Path = Path("./output/embeddings_cache.json"),
):
    """
    Save embeddings locally alongside the chunks.
    Lets you re-upsert without re-embedding (saves time if you tweak metadata).
    """
    cache = [
        {"id": c["id"], "embedding": e}
        for c, e in zip(chunks, embeddings)
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f)
    print(f"Embedding cache saved → {cache_path}")


def load_embedding_cache(
    chunks: List[Dict],
    cache_path: Path = Path("./output/embeddings_cache.json"),
):
    """
    Load cached embeddings if available, skipping re-computation.
    Returns None if cache doesn't exist or IDs don't match.
    """
    if not cache_path.exists():
        return None

    print("Found embedding cache — loading...")
    with cache_path.open("r", encoding="utf-8") as f:
        cache = json.load(f)

    cache_ids = {item["id"]: item["embedding"] for item in cache}
    chunk_ids = {c["id"] for c in chunks}

    if cache_ids.keys() != chunk_ids:
        print("Cache IDs don't match current chunks — will re-embed.")
        return None

    if any(len(embedding) != EMBEDDING_DIM for embedding in cache_ids.values()):
        print("Cache embedding dimensions are invalid — will re-embed.")
        return None

    # Return embeddings in same order as chunks
    embeddings = [cache_ids[c["id"]] for c in chunks]
    print(f"Loaded {len(embeddings):,} cached embeddings.")
    return embeddings


def verify_namespace_samples(
    index,
    vectors: Sequence[Dict],
    namespace: Optional[str],
    expected_strategy: str,
) -> List[str]:
    """Self-query representative vectors and verify metadata and namespace.

    Pinecone's namespace statistics can become available before ``fetch``
    readback does. A nearest-neighbour query using the exact stored embedding
    exercises the same retrieval path as the RAG application and provides a
    deterministic post-upsert check: the vector must retrieve itself first.
    """
    positions = sorted({0, len(vectors) // 2, len(vectors) - 1})
    sample_vectors = [vectors[position] for position in positions]
    sample_ids = [vector["id"] for vector in sample_vectors]
    for vector in sample_vectors:
        query_args = {
            "vector": vector["values"],
            "top_k": 1,
            "include_metadata": True,
        }
        if namespace:
            query_args["namespace"] = namespace
        response = index.query(**query_args)
        matches = getattr(response, "matches", None)
        if matches is None and isinstance(response, dict):
            matches = response.get("matches", [])
        matches = matches or []
        if not matches:
            raise RuntimeError(
                f"Pinecone sample query returned no match for {vector['id']}"
            )

        match = matches[0]
        vector_id = getattr(match, "id", None)
        if vector_id is None and isinstance(match, dict):
            vector_id = match.get("id")
        if vector_id != vector["id"]:
            raise RuntimeError(
                f"Pinecone sample query returned {vector_id!r}; "
                f"expected self-match {vector['id']!r}"
            )

        metadata = getattr(match, "metadata", None)
        if metadata is None and isinstance(match, dict):
            metadata = match.get("metadata", {})
        metadata = metadata or {}
        if metadata.get("strategy") != expected_strategy:
            raise RuntimeError(
                f"Vector {vector_id} has strategy {metadata.get('strategy')!r}; "
                f"expected {expected_strategy!r}"
            )
        if metadata.get("text") != vector["metadata"]["text"]:
            raise RuntimeError(f"Vector {vector_id} text metadata failed readback")

    return sample_ids


# ─────────────────────────────────────────────
# QUICK SANITY CHECK (run after upsert)
# ─────────────────────────────────────────────

def sanity_check(
    index,
    model_name: str,
    namespace: Optional[str] = None,
    expected_strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run a test query against the index to confirm everything works.
    """
    print("\n── Sanity check ──")
    model = SentenceTransformer(model_name)

    test_query = "Who inherits property when there is no will?"
    query_vec = model.encode(test_query, normalize_embeddings=True).tolist()

    query_args = {
        "vector": query_vec,
        "top_k": 3,
        "include_metadata": True,
    }
    if namespace:
        query_args["namespace"] = namespace
    results = index.query(**query_args)

    print(f"Query: '{test_query}'")
    print(f"Top {len(results['matches'])} results:")
    matches = results["matches"]
    if not matches:
        raise RuntimeError("Pinecone sanity query returned no matches")
    if expected_strategy and any(
        match["metadata"].get("strategy") != expected_strategy
        for match in matches
    ):
        raise RuntimeError(
            f"Sanity query returned a vector outside strategy {expected_strategy!r}"
        )

    summaries = []
    for i, match in enumerate(matches, 1):
        score  = match["score"]
        source = match["metadata"].get("source", "?")
        snippet = match["metadata"].get("text", "")[:120].replace("\n", " ")
        print(f"  {i}. [{score:.3f}] {source}")
        print(f"     {snippet}...")
        summaries.append(
            {
                "id": match["id"],
                "score": round(float(score), 4),
                "source": source,
                "strategy": match["metadata"].get("strategy"),
            }
        )
    print()
    return summaries


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed and upload one isolated chunking strategy."
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGY_PROFILES),
        default="recursive",
        help="Defaults to recursive to preserve the original application workflow.",
    )
    return parser.parse_args()


def run_strategy_upload(strategy: str) -> Dict[str, Any]:
    if strategy not in STRATEGY_PROFILES:
        raise ValueError(f"Unsupported strategy: {strategy}")

    profile = STRATEGY_PROFILES[strategy]
    chunks_path = profile["chunks_file"]
    cache_path = profile["cache_file"]
    namespace = profile["namespace"]
    if strategy == "fixed" and namespace != "fixed":
        raise RuntimeError("Fixed chunks must use the isolated 'fixed' namespace")

    chunks = load_chunks(chunks_path, strategy)
    embeddings = load_embedding_cache(chunks, cache_path)
    if embeddings is None:
        embeddings = embed_chunks(chunks, EMBEDDING_MODEL)
        save_embedding_cache(chunks, embeddings, cache_path)

    vectors = build_pinecone_vectors(chunks, embeddings)
    index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX_NAME)
    before_stats = index.describe_index_stats()
    default_before = _namespace_vector_count(before_stats, None)
    namespace_before = _namespace_vector_count(before_stats, namespace)

    namespace_after = upsert_to_pinecone(index, vectors, namespace)
    sample_ids = verify_namespace_samples(index, vectors, namespace, strategy)
    query_matches = sanity_check(
        index,
        EMBEDDING_MODEL,
        namespace=namespace,
        expected_strategy=strategy,
    )

    after_stats = index.describe_index_stats()
    default_after = _namespace_vector_count(after_stats, None)
    if strategy == "fixed" and default_after != default_before:
        raise RuntimeError(
            "Recursive default namespace count changed during fixed upload: "
            f"{default_before} -> {default_after}"
        )

    return {
        "strategy": strategy,
        "index_name": PINECONE_INDEX_NAME,
        "namespace": namespace or "__default__",
        "chunk_count": len(chunks),
        "embedding_dimension": EMBEDDING_DIM,
        "embedding_model": EMBEDDING_MODEL,
        "cache_path": str(cache_path),
        "namespace_count_before": namespace_before,
        "namespace_count_after": namespace_after,
        "default_namespace_count_before": default_before,
        "default_namespace_count_after": default_after,
        "total_vector_count_after": int(after_stats.total_vector_count),
        "verified_sample_ids": sample_ids,
        "sanity_query_matches": query_matches,
    }


def main():
    args = parse_args()
    print("=" * 55)
    print("  Legal Inheritance RAG — Step 2: Embed + Upsert")
    print("=" * 55)

    report = run_strategy_upload(args.strategy)
    print(json.dumps(report, indent=2))
    print("=" * 55)


if __name__ == "__main__":
    main()
