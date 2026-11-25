"""Local reproduction evidence; run from the repository root with .venv Python."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "output" / "reproducibility-2026-11-25"
DEST.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + (DEST / "mlflow.db").as_posix()
os.environ["MLFLOW_EXPERIMENT_NAME"] = "laptop-reproducibility-2026-11-25"
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
os.environ.setdefault("PINECONE_INIT_MAX_ATTEMPTS", "1")


def save(name, value):
    (DEST / (name + ".json")).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str), flush=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    from dotenv import dotenv_values
    from huggingface_hub.constants import HF_HUB_CACHE

    config = dotenv_values(ROOT / ".env")
    packages = sorted(
        (f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
        key=str.lower,
    )
    (DEST / "requirements-resolved.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    corpus_manifest = [
        {"path": str(p.relative_to(ROOT / "corpus")), "sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted((ROOT / "corpus").rglob("*")) if p.is_file()
    ]
    (DEST / "corpus-manifest.json").write_text(json.dumps(corpus_manifest, indent=2), encoding="utf-8")
    save("inventory", {
        "python": sys.version,
        "platform": platform.platform(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "credentials": {k: "configured" if config.get(k) else "missing" for k in ("PINECONE_API_KEY", "GROQ_API_KEY")},
        "generation_model": config.get("GROQ_MODEL") or "qwen/qwen3.8-27b",
        "model_cache": HF_HUB_CACHE,
        "cached_models": [p.name for p in Path(HF_HUB_CACHE).glob("models--*")],
        "cached_model_main_revisions": {p.name: (p / "refs" / "main").read_text().strip() for p in Path(HF_HUB_CACHE).glob("models--*") if (p / "refs" / "main").exists()},
        "project_cached_model_configs": [str(p.relative_to(ROOT)) for p in (ROOT / ".cache").glob("**/config.json")],
        "corpus_files": len(list((ROOT / "corpus").rglob("*.pdf"))) + len(list((ROOT / "corpus").rglob("*.txt"))),
        "runtime_artifacts": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in (ROOT / "output").glob("*.json")},
    })


def preprocess():
    from src import preprocess as prep
    from src.prepare_bm25 import validate_bm25_alignment

    generated = DEST / "prepared"
    generated.mkdir(exist_ok=True)
    prep.OUTPUT_DIR = str(generated)
    prep.main()
    checks = {}
    for name in ("chunks_fixed.json", "chunks_recursive.json", "bm25_corpus_fixed.json", "bm25_corpus.json"):
        current = ROOT / "output" / name
        rebuilt = generated / name
        original = json.loads(current.read_text(encoding="utf-8"))
        reproduced = json.loads(rebuilt.read_text(encoding="utf-8"))
        checks[name] = {
            "entries": len(reproduced) if isinstance(reproduced, list) else len(reproduced["ids"]),
            "json_equal": original == reproduced,
            "bytes_equal": digest(current) == digest(rebuilt),
            "original_sha256": digest(current),
            "reproduced_sha256": digest(rebuilt),
        }
    for chunks_name, bm25_name in (("chunks_fixed.json", "bm25_corpus_fixed.json"), ("chunks_recursive.json", "bm25_corpus.json")):
        validate_bm25_alignment(
            json.loads((generated / chunks_name).read_text(encoding="utf-8")),
            json.loads((generated / bm25_name).read_text(encoding="utf-8")),
        )
    save("preprocess", {"status": "passed" if all(v["json_equal"] for v in checks.values()) else "mismatch", "checks": checks})
    assert all(v["json_equal"] for v in checks.values()), "Preprocessing output mismatch"


def services():
    import requests
    from dotenv import load_dotenv
    from pinecone import Pinecone

    load_dotenv(ROOT / ".env")
    result = {}
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        description = pc.describe_index("legal-inheritance-rag")
        stats = pc.Index("legal-inheritance-rag").describe_index_stats()
        result["pinecone"] = {"status": "passed", "dimension": description.dimension, "metric": description.metric, "stats": stats.to_dict()}
    except Exception as exc:
        result["pinecone"] = {"status": "failed", "error_type": type(exc).__name__}
    try:
        response = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]}, timeout=20)
        ids = sorted(row["id"] for row in response.json().get("data", []))
        selected = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        result["groq"] = {"status": "passed" if response.ok and selected in ids else "failed", "http_status": response.status_code, "configured_model": selected, "configured_model_available": selected in ids, "available_models": ids}
    except Exception as exc:
        result["groq"] = {"status": "failed", "error_type": type(exc).__name__}
    save("services", result)
    assert all(v["status"] == "passed" for v in result.values()), "Provider validation failed"


def trace():
    from src.mlflow_smoke_test import run_smoke_test
    save("trace", run_smoke_test())


def index_audit():
    import numpy as np
    import requests
    from dotenv import load_dotenv
    from pinecone import Pinecone
    from urllib.parse import quote, urlencode
    from src.embed_upsert import STRATEGY_PROFILES, build_pinecone_vectors

    load_dotenv(ROOT / ".env")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    host = pc.describe_index("legal-inheritance-rag").host
    base = host if host.startswith("https://") else "https://" + host
    session = requests.Session()
    session.headers.update({"Api-Key": os.environ["PINECONE_API_KEY"]})
    report = {}
    for strategy, settings in STRATEGY_PROFILES.items():
        chunks = json.loads(settings["chunks_file"].read_text(encoding="utf-8"))
        cached = json.loads(settings["cache_file"].read_text(encoding="utf-8"))
        expected = build_pinecone_vectors(chunks, [row["embedding"] for row in cached])
        missing, metadata_mismatches, vector_mismatches, max_diff = 0, 0, 0, 0.0
        batches, batch, url_bytes = [], [], 0
        for vector in expected:
            size = len(quote(vector["id"], safe="")) + 5
            if batch and url_bytes + size > 4000:
                batches.append(batch)
                batch, url_bytes = [], 0
            batch.append(vector)
            url_bytes += size
        if batch:
            batches.append(batch)
        for batch in batches:
            # Explicit %20 encoding avoids SDK 10.3 fetch misses for IDs with spaces.
            params = [("ids", v["id"]) for v in batch]
            if settings["namespace"]:
                params.append(("namespace", settings["namespace"]))
            response = session.get(base.rstrip("/") + "/vectors/fetch?" + urlencode(params, quote_via=quote), timeout=30)
            response.raise_for_status()
            fetched = response.json().get("vectors", {})
            for vector in batch:
                actual = fetched.get(vector["id"])
                if actual is None:
                    missing += 1
                    continue
                metadata_mismatches += actual.get("metadata") != vector["metadata"]
                values = np.asarray(actual["values"])
                baseline = np.asarray(vector["values"])
                vector_mismatches += not np.allclose(values, baseline, atol=1e-6, rtol=1e-5)
                max_diff = max(max_diff, float(np.max(np.abs(values - baseline))))
        report[strategy] = {"expected_vectors": len(expected), "missing": missing, "metadata_mismatches": int(metadata_mismatches), "vector_mismatches_atol_1e-6_rtol_1e-5": int(vector_mismatches), "max_absolute_difference": max_diff, "remote_writes": 0}
    save("index-audit", report)
    assert all(v["missing"] == v["metadata_mismatches"] == v["vector_mismatches_atol_1e-6_rtol_1e-5"] == 0 for v in report.values()), "Index artifact mismatch"


def web():
    import mlflow
    from fastapi.testclient import TestClient
    from src.web import app

    started = time.perf_counter()
    result = {"requests": []}
    with TestClient(app) as client:
        result["startup_seconds"] = round(time.perf_counter() - started, 2)
        for route in ("/health", "/docs", "/ui/", "/metrics", "/stats"):
            response = client.get(route)
            entry = {"route": route, "http_status": response.status_code}
            if "application/json" in response.headers.get("content-type", ""):
                entry["body"] = response.json()
            result["requests"].append(entry)
        for strategy, mode, k in (("fixed", "semantic", 3), ("fixed", "hybrid", 5), ("recursive", "hybrid", 5)):
            response = client.post("/predict", json={"query": "What share does a daughter receive in Islamic inheritance?", "chunking_strategy": strategy, "retrieval_mode": mode, "top_k": k, "temperature": 0.2})
            entry = {"profile": f"{strategy}-{mode}-top{k}", "http_status": response.status_code, "body": response.json()}
            trace_id = mlflow.get_last_active_trace_id()
            mlflow.flush_trace_async_logging()
            recorded = mlflow.get_trace(trace_id) if trace_id else None
            entry["trace_id"] = trace_id
            entry["spans"] = [s.name for s in recorded.data.spans] if recorded else []
            result["requests"].append(entry)
        result["final_metrics"] = client.get("/metrics").json()
    save("web", result)
    assert all(row["http_status"] == 200 for row in result["requests"]), "Application smoke check failed"
    expected_spans = {"legal_aid_rag_query", "hybrid_legal_retrieval", "legal_generation"}
    assert all(expected_spans.issubset(row["spans"]) for row in result["requests"] if "profile" in row), "Missing RAG trace spans"


def embeddings():
    import numpy as np
    from src.embed_upsert import EMBEDDING_MODEL, STRATEGY_PROFILES, embed_chunks, load_chunks

    checks = {}
    for strategy, settings in STRATEGY_PROFILES.items():
        chunks = load_chunks(settings["chunks_file"], strategy)
        cached = json.loads(settings["cache_file"].read_text(encoding="utf-8"))
        assert [row["id"] for row in cached] == [row["id"] for row in chunks]
        rebuilt = np.asarray(embed_chunks(chunks, EMBEDDING_MODEL), dtype=np.float32)
        original = np.asarray([row["embedding"] for row in cached], dtype=np.float32)
        np.savez_compressed(DEST / f"embeddings-{strategy}.npz", embeddings=rebuilt)
        checks[strategy] = {
            "shape": list(rebuilt.shape),
            "allclose_atol_1e-5_rtol_1e-4": bool(np.allclose(rebuilt, original, atol=1e-5, rtol=1e-4)),
            "max_absolute_difference": float(np.max(np.abs(rebuilt - original))),
            "mean_cosine_similarity": float(np.mean(np.sum(rebuilt * original, axis=1))),
            "ids_aligned": True,
        }
    save("embeddings", checks)
    assert all(v["allclose_atol_1e-5_rtol_1e-4"] for v in checks.values()), "Embedding cache mismatch"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inventory", "preprocess", "services", "trace", "web", "embeddings", "index_audit"))
    args = parser.parse_args()
    globals()[args.stage]()
