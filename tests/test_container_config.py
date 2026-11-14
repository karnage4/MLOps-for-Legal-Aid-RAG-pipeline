import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_serves_fastapi_and_checks_health():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "uvicorn src.app:app" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile
    assert "--workers 1" in dockerfile
    assert "chown app:app /app" in dockerfile
    assert "USER app" in dockerfile
    assert "streamlit" not in dockerfile.lower()


def test_docker_context_is_allow_listed_and_excludes_secrets():
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    lines = {line.strip() for line in dockerignore.splitlines() if line.strip()}

    assert "*" in lines
    assert "!src/**" in lines
    assert "!output/bm25_corpus.json" in lines
    assert "!output/bm25_corpus_fixed.json" in lines
    assert "!.env" not in lines
    assert "!.env.txt" not in lines


def test_container_runtime_artifacts_are_complete():
    expected_counts = {
        "bm25_corpus.json": 3593,
        "bm25_corpus_fixed.json": 883,
    }
    for filename, expected_count in expected_counts.items():
        corpus = PROJECT_ROOT / "output" / filename
        assert corpus.is_file()
        payload = json.loads(corpus.read_text(encoding="utf-8"))
        assert len(payload["ids"]) == expected_count
        assert len(payload["texts"]) == expected_count
        assert all(payload["ids"])
        assert all(payload["texts"])
