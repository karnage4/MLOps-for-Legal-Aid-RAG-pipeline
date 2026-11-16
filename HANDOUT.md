# Branch 10 Handout: Milestone 2 Containerization

## Purpose

This is the restart and rebuild guide for branch `10-m2-containerization`. It inherits the complete post-M1 RAG/evaluation work from branch 9 and packages the FastAPI service as a reproducible Docker image. It also makes both fixed and recursive retrieval artifacts available to the API.

Use this branch to understand or reproduce the container contract. CI automation begins on branch 11 and cloud deployment begins on branch 12.

## Repository and Branch

- Repository: `https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline.git`
- Working directory: `Legal-Aid-RAG-MLflow`
- Branch: `10-m2-containerization`
- Branch tip when this handout was written: `68bcbc3`
- Parent: `9-post-m1-fixes`
- Milestone roadmap: `MILESTONE2_PLAN.md`
- Keep `Legal-Aid-RAG-main/` read-only.
- Never commit `.env`, `.venv/`, `mlflow.db`, `mlruns/`, or raw corpus material.

## Branch Deliverables

### Production Docker image

The old Streamlit/port-7860 setup was replaced with a FastAPI image that:

- uses Python 3.12 slim;
- installs the pinned API dependency set from `requirements-api.txt`;
- copies only the required source and serving artifacts;
- runs as an unprivileged `app` user;
- keeps `/app` writable so local MLflow initialization does not fail;
- exposes port `8000`;
- starts one Uvicorn worker on `${PORT:-8000}`; and
- includes a healthcheck against `/health` with a 180-second startup allowance.

`.dockerignore` uses a deny-by-default allowlist. It includes the Dockerfile, production dependency file, source, and the two BM25 artifacts, while excluding secrets, virtual environments, tests, raw corpus files, MLflow databases/runs, caches, and reports.

### Versioned retrieval artifacts

Both prepared sparse-retrieval corpora are tracked so a clean Docker build does not depend on an unversioned local file.

| Chunking | File | Entries | SHA-256 |
|---|---|---:|---|
| Recursive | `output/bm25_corpus.json` | 3,593 | `51C8B6CD081F954F3E5D94D4676E98FFCCB46556C6BC5FD88A736207447DCDC2` |
| Fixed | `output/bm25_corpus_fixed.json` | 883 | `E878377705440DC9E599ECD3E3C2B55CD69299A072D23B7F1D806FD9736A7F15` |

Do not regenerate or replace these silently. If the corpus or chunking rules change, create new clearly versioned artifacts, record counts and hashes, rebuild Pinecone namespaces, and rerun evaluation.

### API support for both chunking methods

`/predict` accepts:

```json
{
  "question": "What remedies are available?",
  "chunking_strategy": "recursive"
}
```

- `chunking_strategy` is validated as `recursive` or `fixed`.
- `recursive` remains the default for backward compatibility.
- Startup calls the shared retriever builder for both strategies.
- The embedder, reranker, and Pinecone client are shared to avoid duplicate heavyweight objects.
- Each strategy has its own BM25 corpus and Pinecone namespace; fixed requests use namespace `fixed`.
- `/health` reports the availability of both strategies.
- Invalid strategy values return HTTP 422 through Pydantic validation.
- The Gradio app retains its single-strategy loading path for local manual use.

## Important Files

- `Dockerfile`: production API image definition and healthcheck.
- `.dockerignore`: build-context security and size allowlist.
- `requirements-api.txt`: pinned runtime dependencies only.
- `src/app.py`: FastAPI lifespan, health, validation, and prediction routing.
- `src/retrieval.py`: shared retriever construction and strategy-specific artifacts/namespaces.
- `output/bm25_corpus.json`: recursive BM25 serving artifact.
- `output/bm25_corpus_fixed.json`: fixed BM25 serving artifact.
- `tests/`: API, loader, health, validation, and artifact checks.
- `MILESTONE2_PLAN.md`: planned branch sequence through final submission.
- `MLOPS_LOG.md`: chronological proof and known limitations.
- `README.md`: normal setup and Docker usage.

## Environment Contract

Create a local `.env` file and keep it out of Git:

```dotenv
PINECONE_API_KEY=...
GROQ_API_KEY=...
```

The Pinecone index must contain embeddings for both the recursive/default namespace and the `fixed` namespace. The local JSON files provide BM25 data; they do not replace Pinecone semantic retrieval.

The serving path uses Groq directly. Hugging Face is no longer attempted because the former model was removed.

## Local Python Verification

```powershell
git fetch origin
git switch 10-m2-containerization
git pull --ff-only origin 10-m2-containerization
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m pytest -q
```

At the completed branch state, the full suite passed `117` tests with one pre-existing SQLAlchemy `noload` warning.

Run the API without Docker when diagnosing code independently from the container runtime:

```powershell
.venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs` and test both chunking strategies.

## Docker Build and Smoke Test

From the repository root:

```powershell
docker build --tag legal-aid-rag-api:m2-phase10 .
docker run --detach --name legal-aid-rag-api --env-file .env --publish 8000:8000 legal-aid-rag-api:m2-phase10
docker logs --follow legal-aid-rag-api
```

In another terminal:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/docs
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"question":"What is legal aid?","chunking_strategy":"recursive"}'
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"question":"What is legal aid?","chunking_strategy":"fixed"}'
```

Inspect the image contract if needed:

```powershell
docker image inspect legal-aid-rag-api:m2-phase10
docker inspect legal-aid-rag-api
```

Stop and remove only this named test container when finished:

```powershell
docker stop legal-aid-rag-api
docker rm legal-aid-rag-api
```

## Local Docker Limitation Encountered

Docker Desktop on the original Windows machine failed before the Dockerfile could execute:

```text
Wsl/Service/RegisterDistro/CreateVm/HCS/ERROR_FILE_NOT_FOUND
```

This is a local Docker Desktop/WSL virtualization problem, not proof of a Dockerfile failure. Do not reset or delete Docker Desktop/WSL data without explicit approval because that may remove unrelated images and volumes. Branch 11's GitHub-hosted Linux Docker build later verified that the image builds cleanly.

## Test Coverage Added

- model/retriever startup with both artifacts;
- recursive and fixed prediction routing;
- invalid `chunking_strategy` returns 422;
- health response lists both strategies;
- artifact count/integrity expectations; and
- existing prediction, MLflow, evaluation, and preparation tests remain intact.

## Known Constraints and Diagnostics

- Application startup loads sentence-transformer, reranker, BM25 data, and Pinecone components before Uvicorn completes startup. This is memory- and time-intensive.
- A healthcheck cannot pass until the lifespan startup finishes.
- One worker is intentional: multiple workers would duplicate the heavyweight model state.
- Live predictions require external Pinecone and Groq connectivity and valid quota.
- Container-local MLflow state is ephemeral unless an external tracking server/database is configured.
- Never bake API keys into the image; pass them at runtime.

If startup hangs, check in this order:

1. container logs and memory limit;
2. existence and readability of both JSON artifacts;
3. required environment variables;
4. Pinecone index/namespaces and network access;
5. model download/cache availability; and
6. whether the hosting platform allows enough startup time and RAM.

## Commit Map

- `d582441` — plan Milestone 2 delivery phases.
- `f518517` — record the main-branch CI prerequisite.
- `4fd2bf7` — containerize the FastAPI RAG service.
- `68bcbc3` — serve and validate both fixed and recursive artifacts.

## Handoff to Branch 11

Branch 10 intentionally stops after containerization. Continue on `11-m2-ci` for Ruff, pytest, and Docker-build automation. Do not retrofit CI-only changes here unless repairing the historical container phase itself.
