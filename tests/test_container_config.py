import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _module_string_constants(path: Path) -> dict[str, str]:
    """Read top-level string constants without importing heavy NLP modules."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    return constants


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


def test_dockerfile_installs_cpu_only_torch():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "download.pytorch.org/whl/cpu" in dockerfile
    assert '"torch==$TORCH_VERSION"' in dockerfile
    assert "torch.version.cuda" in dockerfile


def test_dockerfile_preloads_the_models_the_retriever_loads():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    retrieval = _module_string_constants(PROJECT_ROOT / "src" / "retrieval.py")

    assert f"SentenceTransformer('{retrieval['EMBEDDING_MODEL']}')" in dockerfile
    assert f"CrossEncoder('{retrieval['RERANKER_MODEL']}'" in dockerfile
    assert "HF_HUB_OFFLINE" in dockerfile


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
