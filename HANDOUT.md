# Branch 11 Handout: Milestone 2 CI

## Purpose

This file is the restart and rebuild guide for branch `11-m2-ci`. It inherits the verified post-M1 pipeline and Docker work, then adds automated Ruff, pytest, and Docker-build checks in GitHub Actions.

The `handoffs/` directory also contains restart documents for every remaining planned M2 branch. Read them in numeric order before continuing deployment work.

## Repository and Branch

- Repository: `https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline.git`
- Working directory: `Legal-Aid-RAG-MLflow`
- Branch: `11-m2-ci`
- Functional branch tip before this documentation: `7b27401`
- Parent: `10-m2-containerization`
- Milestone plan: `MILESTONE2_PLAN.md`
- Never commit `.env`, `.venv/`, `mlflow.db`, `mlruns/`, caches, or raw corpus files.
- Treat `Legal-Aid-RAG-main/` as a read-only legacy reference.

## Cumulative System at This Point

- Groq-only grounded generation; no obsolete Hugging Face first attempt.
- Pinecone semantic retrieval plus optional BM25 hybrid retrieval/reranking.
- Recursive and fixed chunking supported by `/predict`.
- Versioned recursive artifact: 3,593 entries.
- Versioned fixed artifact: 883 entries.
- FastAPI `/health`, `/predict`, and `/docs`.
- Docker image on Python 3.12 slim, one Uvicorn worker, port 8000, unprivileged user, and `/health` healthcheck.
- Post-M1 eight-configuration evaluation and inference-ablation evidence retained under `artifacts/`.

## CI Implementation

Workflow: `.github/workflows/ci.yml`

Triggers:

- every push;
- every pull request; and
- manual `workflow_dispatch`.

Permissions are read-only (`contents: read`). Concurrency cancels an obsolete run for the same workflow/ref.

| Visible job | Command/result | Timeout |
|---|---|---:|
| `Ruff lint` | Installs `requirements-dev.txt`; runs `ruff check src tests gradio_app.py --output-format=github` | 10 min |
| `Pytest` | Installs `requirements.txt`; runs the full quiet pytest suite | 30 min |
| `Docker build` | Builds an image tagged with the commit SHA and inspects it | 45 min |

The workflow uses Ubuntu runners, Python 3.12, `actions/checkout@v6`, `actions/setup-python@v5`, and pip caching.

## Ruff Configuration and Cleanup

- Ruff is pinned as `ruff==0.17.5` in `requirements-dev.txt`.
- Configuration is in `pyproject.toml` with Python 3.12 target and rules `E4`, `E7`, `E9`, and `F`.
- `gradio_app.py` ignores `E402` because `MPLCONFIGDIR` must be set before importing libraries that initialize Matplotlib.
- Genuine findings were fixed: unused imports/assignments, an unnecessary f-string, and obsolete `nltk` dependency usage.
- The cleanup did not broaden into unrelated formatting or behavior changes.

## Verification Evidence

Local verification at this branch:

- Ruff passed.
- Focused CI-related tests passed (`32` tests).
- Full suite passed (`117` tests).
- One pre-existing SQLAlchemy `noload` warning remained and did not fail CI.

GitHub Actions evidence:

- Branch run `43652143632`: all three jobs passed.
- Branch run URL: `https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline/actions/runs/43652143632`
- Main run `43652828766` at commit `3a3c422`: all required jobs passed.
- Main run URL: `https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline/actions/runs/43652828766`

The GitHub Linux build proves the Dockerfile builds successfully even though Docker Desktop/WSL was broken on the original Windows machine.

## Reproduce Locally

```powershell
git fetch origin
git switch 11-m2-ci
git pull --ff-only origin 11-m2-ci
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
.venv\Scripts\ruff.exe check src tests gradio_app.py
.venv\Scripts\python.exe -m pytest -q
```

Docker, if the local engine is healthy:

```powershell
docker build --tag legal-aid-rag-api:branch11 .
docker image inspect legal-aid-rag-api:branch11
```

GitHub automatically reruns CI after a push. The workflow does not require Pinecone or Groq secrets because unit tests mock external behavior and the build does not start the app.

## Important Files

- `.github/workflows/ci.yml`: CI definition.
- `pyproject.toml`: Ruff rules.
- `requirements-dev.txt`: pinned lint dependency.
- `Dockerfile`, `.dockerignore`, `requirements-api.txt`: container contract inherited from branch 10.
- `output/bm25_corpus.json`, `output/bm25_corpus_fixed.json`: tracked serving artifacts.
- `MILESTONE2_PLAN.md`: branch 10–14 roadmap.
- `MLOPS_LOG.md`: chronological proof.
- `handoffs/12-m2-render-deployment.md`: deployment attempt, failure evidence, and hosting decision.
- `handoffs/13-m2-monitoring-runbook.md`: monitoring/runbook implementation checklist.
- `handoffs/14-m2-submission.md`: final rubric and submission checklist.

## Known Constraints

- CI builds but does not boot/smoke-test the heavyweight application image.
- Live runtime still needs `PINECONE_API_KEY`, `GROQ_API_KEY`, Pinecone data, network access, and sufficient memory.
- One Uvicorn worker is deliberate because each worker would duplicate embedding/reranking memory.
- In-container local MLflow state is ephemeral unless a durable external backend is configured.
- Never add secrets to workflow YAML or GitHub logs.

## Commit Map

- `a683f28` — add lint, test, and Docker CI jobs.
- `3a3c422` — record successful CI verification.
- `7b27401` — document successful main-branch CI result.

## Continuation Rule

The remaining planned chain is:

```text
11-m2-ci
  -> 12-m2-render-deployment
  -> 13-m2-monitoring-runbook
  -> 14-m2-submission
```

Create a later branch only from the verified tip of its predecessor. The Render attempt has already begun and is documented in the branch-12 handoff; do not repeat it without reading its memory measurements and failure analysis.
