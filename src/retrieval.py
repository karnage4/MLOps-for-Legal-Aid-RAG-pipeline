"""
Step 3 - Hybrid Retrieval: BM25 + Semantic + RRF + Cross-Encoder Re-ranking
=============================================================================
This module is imported by your Streamlit app (Step 6).
It can also be run standalone to test retrieval.

Pipeline per query:
  1. BM25 keyword search       → configurable candidate pool
  2. Semantic vector search    → configurable candidate pool (Pinecone)
  3. RRF fusion                → merged ranked list
  4. Cross-encoder re-ranking  → bounded rerank pool → final top-k chunks

Usage (standalone test):
  pip install rank-bm25 sentence-transformers pinecone \
              crossencoder nltk tqdm
  python retrieval.py

Import in your app:
  from retrieval import build_retriever, retrieve
"""

import os
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import nltk
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone, PineconeConnectionError, PineconeTimeoutError
import mlflow
from mlflow.entities import SpanType

from dotenv import load_dotenv

try:
    from src.mlflow_config import configure_mlflow
except ImportError:  # supports `python src/retrieval.py`
    from mlflow_config import configure_mlflow

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "legal-inheritance-rag"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"

# Cross-encoder for re-ranking
# ms-marco is trained on QA pairs — well suited for legal retrieval
RERANKER_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BM25_TOP_K          = max(10, int(os.getenv("BM25_CANDIDATE_K", "10")))
SEMANTIC_TOP_K      = max(10, int(os.getenv("SEMANTIC_CANDIDATE_K", "10")))
RERANK_CANDIDATE_K = max(10, int(os.getenv("RERANK_CANDIDATE_K", "12")))
RRF_K               = 60    # RRF constant (standard value — do not change)
FINAL_TOP_K         = 5     # chunks sent to LLM after re-ranking
PINECONE_QUERY_MAX_ATTEMPTS = max(
    1, int(os.getenv("PINECONE_QUERY_MAX_ATTEMPTS", "3"))
)
PINECONE_RETRY_BASE_SECONDS = max(
    0.0, float(os.getenv("PINECONE_RETRY_BASE_SECONDS", "1.5"))
)

CHUNKING_STRATEGIES = {
    "recursive": {
        "bm25_corpus_file": "./output/bm25_corpus.json",
        "pinecone_namespace": None,
    },
    "fixed": {
        "bm25_corpus_file": "./output/bm25_corpus_fixed.json",
        "pinecone_namespace": "fixed",
    },
}
RETRIEVAL_MODES = {"semantic", "hybrid"}


@dataclass(frozen=True)
class RetrievalProfile:
    """One reproducible retrieval configuration for evaluation or serving."""

    chunking_strategy: str = "recursive"
    retrieval_mode: str = "hybrid"
    top_k: int = FINAL_TOP_K

    def __post_init__(self) -> None:
        if self.chunking_strategy not in CHUNKING_STRATEGIES:
            supported = ", ".join(sorted(CHUNKING_STRATEGIES))
            raise ValueError(
                f"Unsupported chunking strategy {self.chunking_strategy!r}; "
                f"choose one of: {supported}"
            )
        if self.retrieval_mode not in RETRIEVAL_MODES:
            supported = ", ".join(sorted(RETRIEVAL_MODES))
            raise ValueError(
                f"Unsupported retrieval mode {self.retrieval_mode!r}; "
                f"choose one of: {supported}"
            )
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")

    @property
    def bm25_corpus_file(self) -> str:
        return CHUNKING_STRATEGIES[self.chunking_strategy]["bm25_corpus_file"]

    @property
    def pinecone_namespace(self) -> Optional[str]:
        return CHUNKING_STRATEGIES[self.chunking_strategy]["pinecone_namespace"]

    @property
    def semantic_only(self) -> bool:
        return self.retrieval_mode == "semantic"

    @property
    def label(self) -> str:
        return f"{self.chunking_strategy}-{self.retrieval_mode}-top{self.top_k}"


DEFAULT_RETRIEVAL_PROFILE = RetrievalProfile()
# Backward-compatible alias for imports that expect the original constant.
BM25_CORPUS_FILE = DEFAULT_RETRIEVAL_PROFILE.bm25_corpus_file
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 1.  BM25 INDEX
# ─────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer: lowercase, remove punctuation, split on whitespace.
    Legal text benefits from keeping compound terms intact, so we
    avoid aggressive stemming here.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # strip punctuation
    tokens = text.split()
    return tokens


def build_bm25_index(corpus_file: str) -> Tuple[BM25Okapi, List[str], List[str]]:
    """
    Load the BM25 corpus saved in Step 1 and build an in-memory index.
    Returns (bm25, ids, texts) — ids are needed to map scores back to chunks.
    """
    print(f"Loading BM25 corpus from '{corpus_file}'...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids   = data["ids"]
    texts = data["texts"]

    print(f"Tokenizing {len(texts):,} documents for BM25...")
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    print(f"BM25 index built — {len(texts):,} documents.")
    return bm25, ids, texts


# ─────────────────────────────────────────────
# 2.  PINECONE + EMBEDDING MODEL
# ─────────────────────────────────────────────

def init_pinecone(api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index


def init_models(embedding_model: str, reranker_model: str):
    print(f"Loading embedding model: {embedding_model}")
    embedder = SentenceTransformer(embedding_model)

    print(f"Loading cross-encoder: {reranker_model}")
    reranker = CrossEncoder(reranker_model, max_length=512)

    return embedder, reranker


# ─────────────────────────────────────────────
# 3.  INDIVIDUAL SEARCH FUNCTIONS
# ─────────────────────────────────────────────

def bm25_search(
    query: str,
    bm25: BM25Okapi,
    bm25_ids: List[str],
    bm25_texts: List[str],
    top_k: int = BM25_TOP_K,
) -> List[Dict]:
    """
    BM25 keyword search.
    Returns list of { id, text, bm25_score } dicts, sorted descending.
    """
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Get top_k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:   # skip zero-score results
            results.append({
                "id":         bm25_ids[idx],
                "text":       bm25_texts[idx],
                "bm25_score": float(scores[idx]),
            })

    return results


def semantic_search(
    query: str,
    embedder: SentenceTransformer,
    pinecone_index,
    top_k: int = SEMANTIC_TOP_K,
    namespace: Optional[str] = None,
    expected_strategy: Optional[str] = None,
    timings_ms: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Semantic vector search via Pinecone.
    Returns list of { id, text, source, semantic_score } dicts.
    """
    embedding_started = time.perf_counter()
    query_vec = embedder.encode(query, normalize_embeddings=True).tolist()
    if timings_ms is not None:
        timings_ms["embedding_ms"] = round(
            (time.perf_counter() - embedding_started) * 1000, 3
        )

    query_args = {
        "vector": query_vec,
        "top_k": top_k,
        "include_metadata": True,
    }
    if namespace:
        query_args["namespace"] = namespace
    pinecone_started = time.perf_counter()
    response = _query_pinecone_with_retry(pinecone_index, query_args)
    if timings_ms is not None:
        timings_ms["pinecone_ms"] = round(
            (time.perf_counter() - pinecone_started) * 1000, 3
        )

    results = []
    for match in response["matches"]:
        metadata = match["metadata"]
        strategy = metadata.get("strategy")
        if expected_strategy and strategy != expected_strategy:
            raise RuntimeError(
                f"Pinecone returned strategy {strategy!r} while querying "
                f"the {expected_strategy!r} retrieval profile"
            )
        results.append({
            "id":             match["id"],
            "text":           metadata.get("text", ""),
            "source":         metadata.get("source", "unknown"),
            "chunk_index":    metadata.get("chunk_index", 0),
            "strategy":       strategy,
            "semantic_score": float(match["score"]),
        })

    return results


def _query_pinecone_with_retry(pinecone_index, query_args: Dict):
    """Retry only transient Pinecone transport failures with bounded backoff."""
    for attempt in range(1, PINECONE_QUERY_MAX_ATTEMPTS + 1):
        try:
            return pinecone_index.query(**query_args)
        except (PineconeConnectionError, PineconeTimeoutError):
            if attempt == PINECONE_QUERY_MAX_ATTEMPTS:
                raise
            delay = PINECONE_RETRY_BASE_SECONDS * attempt
            print(
                "Transient Pinecone query failure; retrying "
                f"in {delay:.1f}s ({attempt}/{PINECONE_QUERY_MAX_ATTEMPTS})...",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("unreachable Pinecone retry state")


# ─────────────────────────────────────────────
# 4.  RECIPROCAL RANK FUSION (RRF)
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: List[Dict],
    semantic_results: List[Dict],
    k: int = RRF_K,
) -> List[Dict]:
    """
    Merge BM25 and semantic ranked lists using Reciprocal Rank Fusion.

    Formula:  RRF(d) = Σ  1 / (k + rank(d))
                      lists

    k=60 is the standard constant from the original RRF paper
    (Cormack et al., 2009). It dampens the impact of top ranks,
    making the fusion robust to outliers in either list.

    Returns merged list sorted by RRF score (descending),
    with full metadata from whichever list had the richer entry.
    """
    rrf_scores: Dict[str, float] = {}
    metadata:   Dict[str, Dict]  = {}

    # Score from BM25 ranking
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in metadata:
            metadata[doc_id] = result

    # Score from semantic ranking
    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        # Prefer semantic metadata (has source/chunk_index from Pinecone)
        if doc_id not in metadata or "source" not in metadata[doc_id]:
            metadata[doc_id] = result

    # Sort by fused score
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, rrf_score in fused:
        entry = dict(metadata[doc_id])
        entry["rrf_score"] = round(rrf_score, 6)
        results.append(entry)

    return results


# ─────────────────────────────────────────────
# 5.  CROSS-ENCODER RE-RANKING
# ─────────────────────────────────────────────

def rerank(
    query: str,
    candidates: List[Dict],
    reranker: CrossEncoder,
    top_k: int = FINAL_TOP_K,
) -> List[Dict]:
    """
    Re-rank the RRF-fused candidates using a cross-encoder.

    Unlike bi-encoders (which encode query and doc independently),
    a cross-encoder sees the query and document together — giving it
    much richer interaction signal. Slower, but only runs on the
    fused top candidates (not the full corpus).

    Returns top_k chunks with a 'rerank_score' field added.
    """
    if not candidates:
        return []

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ─────────────────────────────────────────────
# 6.  FULL HYBRID PIPELINE
# ─────────────────────────────────────────────

def _retrieve_impl(
    query: str,
    bm25: BM25Okapi,
    bm25_ids: List[str],
    bm25_texts: List[str],
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    pinecone_index,
    top_k: int = FINAL_TOP_K,
    semantic_only: bool = False,   # for ablation study
    pinecone_namespace: Optional[str] = None,
    chunking_strategy: str = "recursive",
) -> Dict:
    """
    Full hybrid retrieval pipeline.

    Args:
        query          : user's question
        semantic_only  : if True, uses dense retrieval without reranking
        top_k          : number of final chunks to return

    Returns dict with:
        chunks         : list of final top_k chunks with all scores
        debug          : intermediate results for the UI / report
    """
    retrieval_started = time.perf_counter()
    stage_timings: Dict[str, float] = {
        "bm25_ms": 0.0,
        "embedding_ms": 0.0,
        "pinecone_ms": 0.0,
        "semantic_total_ms": 0.0,
        "fusion_ms": 0.0,
        "rerank_ms": 0.0,
    }

    # Step A: BM25 search
    if not semantic_only:
        bm25_started = time.perf_counter()
        bm25_results = bm25_search(query, bm25, bm25_ids, bm25_texts)
        stage_timings["bm25_ms"] = round(
            (time.perf_counter() - bm25_started) * 1000, 3
        )
    else:
        bm25_results = []

    # Step B: Semantic search
    semantic_started = time.perf_counter()
    sem_results = semantic_search(
        query,
        embedder,
        pinecone_index,
        namespace=pinecone_namespace,
        expected_strategy=chunking_strategy,
        timings_ms=stage_timings,
    )
    stage_timings["semantic_total_ms"] = round(
        (time.perf_counter() - semantic_started) * 1000, 3
    )

    # Pure semantic retrieval keeps Pinecone's dense ranking. Hybrid retrieval
    # fuses sparse+dense candidates and then applies cross-encoder reranking.
    if semantic_only:
        fused = sem_results
        rerank_candidates = []
        final_chunks = sem_results[:top_k]
    elif bm25_results:
        fusion_started = time.perf_counter()
        fused = reciprocal_rank_fusion(bm25_results, sem_results)
        stage_timings["fusion_ms"] = round(
            (time.perf_counter() - fusion_started) * 1000, 3
        )
        rerank_candidates = fused[:RERANK_CANDIDATE_K]
        rerank_started = time.perf_counter()
        final_chunks = rerank(query, rerank_candidates, reranker, top_k=top_k)
        stage_timings["rerank_ms"] = round(
            (time.perf_counter() - rerank_started) * 1000, 3
        )
    else:
        fused = sem_results
        rerank_candidates = fused[:RERANK_CANDIDATE_K]
        rerank_started = time.perf_counter()
        final_chunks = rerank(query, rerank_candidates, reranker, top_k=top_k)
        stage_timings["rerank_ms"] = round(
            (time.perf_counter() - rerank_started) * 1000, 3
        )

    # BM25-only hybrid candidates do not carry Pinecone metadata. They still
    # come from the strategy-specific BM25 corpus, so attach that resolved
    # strategy while rejecting any contradictory Pinecone metadata.
    for chunk in final_chunks:
        observed_strategy = chunk.get("strategy")
        if observed_strategy and observed_strategy != chunking_strategy:
            raise RuntimeError(
                f"Retrieved chunk {chunk.get('id')!r} has strategy "
                f"{observed_strategy!r}; expected {chunking_strategy!r}"
            )
        chunk["strategy"] = chunking_strategy

    stage_timings["retrieval_total_ms"] = round(
        (time.perf_counter() - retrieval_started) * 1000, 3
    )

    return {
        "chunks": final_chunks,
        "debug": {
            "bm25_count":     len(bm25_results),
            "semantic_count": len(sem_results),
            "fused_count":    len(fused),
            "rerank_candidate_count": len(rerank_candidates),
            "final_count":    len(final_chunks),
            "mode":           "semantic" if semantic_only else "hybrid",
            "chunking_strategy": chunking_strategy,
            "pinecone_namespace": pinecone_namespace or "__default__",
            "top_k": top_k,
            "candidate_limits": {
                "bm25": BM25_TOP_K,
                "semantic": SEMANTIC_TOP_K,
                "rerank": RERANK_CANDIDATE_K,
            },
            "timings_ms": stage_timings,
        }
    }


def retrieve(
    query: str,
    bm25: BM25Okapi,
    bm25_ids: List[str],
    bm25_texts: List[str],
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    pinecone_index,
    top_k: int = FINAL_TOP_K,
    semantic_only: bool = False,
    pinecone_namespace: Optional[str] = None,
    chunking_strategy: str = "recursive",
) -> Dict:
    """Run retrieval while recording inputs and full retrieved content in MLflow."""
    configure_mlflow()
    with mlflow.start_span(name="hybrid_legal_retrieval", span_type=SpanType.RETRIEVER) as span:
        span.set_inputs(
            {
                "query": query,
                "top_k": top_k,
                "semantic_only": semantic_only,
                "retrieval_mode": "semantic" if semantic_only else "hybrid",
                "chunking_strategy": chunking_strategy,
                "pinecone_namespace": pinecone_namespace or "__default__",
                "bm25_top_k": BM25_TOP_K,
                "semantic_top_k": SEMANTIC_TOP_K,
                "rerank_candidate_k": RERANK_CANDIDATE_K,
                "rrf_k": RRF_K,
                "embedding_model": EMBEDDING_MODEL,
                "reranker_model": RERANKER_MODEL,
            }
        )
        result = _retrieve_impl(
            query,
            bm25,
            bm25_ids,
            bm25_texts,
            embedder,
            reranker,
            pinecone_index,
            top_k=top_k,
            semantic_only=semantic_only,
            pinecone_namespace=pinecone_namespace,
            chunking_strategy=chunking_strategy,
        )
        for timing_name, timing_value in result["debug"]["timings_ms"].items():
            span.set_attribute(f"retrieval.{timing_name}", timing_value)
        span.set_attribute(
            "retrieval.rerank_candidate_count",
            result["debug"]["rerank_candidate_count"],
        )
        # MLflow's built-in RAG judges expect a RETRIEVER span to output the
        # document list directly. Each existing chunk already contains a
        # recognized ``text`` field, so this changes trace serialization only;
        # the retrieval result returned to the application remains unchanged.
        span.set_outputs(result["chunks"])
        return result


# ─────────────────────────────────────────────
# 7.  BUILDER FUNCTION (for Streamlit app)
# ─────────────────────────────────────────────

def build_retrievers(
    chunking_strategies: Iterable[str] = ("recursive", "fixed"),
) -> Dict[str, Dict]:
    """Build strategy-specific BM25 indexes with shared dense resources."""
    strategies = list(dict.fromkeys(chunking_strategies))
    if not strategies:
        raise ValueError("At least one chunking strategy is required")

    profiles = {
        strategy: RetrievalProfile(chunking_strategy=strategy)
        for strategy in strategies
    }
    sparse_components = {
        strategy: build_bm25_index(profile.bm25_corpus_file)
        for strategy, profile in profiles.items()
    }
    pinecone_index = init_pinecone(PINECONE_API_KEY, PINECONE_INDEX_NAME)
    embedder, reranker = init_models(EMBEDDING_MODEL, RERANKER_MODEL)

    retrievers = {}
    for strategy, profile in profiles.items():
        bm25, bm25_ids, bm25_texts = sparse_components[strategy]
        retrievers[strategy] = {
            "bm25": bm25,
            "bm25_ids": bm25_ids,
            "bm25_texts": bm25_texts,
            "embedder": embedder,
            "reranker": reranker,
            "pinecone_index": pinecone_index,
            "pinecone_namespace": profile.pinecone_namespace,
            "chunking_strategy": profile.chunking_strategy,
        }
    return retrievers


def build_retriever(chunking_strategy: str = "recursive"):
    """
    Initialize all retrieval components.
    Call this ONCE at app startup (use st.cache_resource in Streamlit).

    Returns a dict of components to pass into retrieve().
    """
    return build_retrievers((chunking_strategy,))[chunking_strategy]


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────

def print_results(result: Dict, query: str):
    debug = result["debug"]
    chunks = result["chunks"]

    print(f"\n{'='*55}")
    print(f"Query : {query}")
    print(f"Mode  : {debug['mode']}")
    print(f"BM25 candidates     : {debug['bm25_count']}")
    print(f"Semantic candidates : {debug['semantic_count']}")
    print(f"After RRF fusion    : {debug['fused_count']}")
    print(f"After re-ranking    : {debug['final_count']}")
    print(f"{'='*55}")

    for i, chunk in enumerate(chunks, 1):
        source      = chunk.get("source", "?")
        rerank_score = chunk.get("rerank_score", 0)
        rrf_score   = chunk.get("rrf_score", 0)
        snippet     = chunk["text"][:200].replace("\n", " ")
        print(f"\n[{i}] source: {source}")
        print(f"    rerank_score: {rerank_score:.4f}  |  rrf_score: {rrf_score:.6f}")
        print(f"    {snippet}...")


def main():
    print("=" * 55)
    print("  Legal Inheritance RAG — Step 3: Retrieval Test")
    print("=" * 55)

    # Build all components
    components = build_retriever()

    test_queries = [
        "Who inherits property when there is no will?",
        "What is the share of a daughter in inheritance under Islamic law?",
        "Can a non-Muslim inherit from a Muslim estate?",
    ]

    for query in test_queries:
        # Hybrid retrieval
        result_hybrid = retrieve(query, **components, semantic_only=False)
        print_results(result_hybrid, f"[HYBRID] {query}")

        # Semantic-only (for ablation comparison)
        result_semantic = retrieve(query, **components, semantic_only=True)
        print_results(result_semantic, f"[SEMANTIC-ONLY] {query}")


if __name__ == "__main__":
    main()
