# Branch 12 Handoff: Deployment

## Goal and Current State

Branch `12-m2-render-deployment` was created from branch 11 to deploy the Dockerized API through a Render Blueprint. Source configuration, tests, and CI passed, but the live Render Free instance failed during application startup because the RAG process exceeds the 512 MiB memory limit.

This is an infrastructure-capacity blocker, not a proven port/CMD defect. The process starts, enters FastAPI lifespan startup, loads heavyweight dependencies, exceeds memory, and never reaches the point where Uvicorn binds the public socket.

## Git and CI Evidence

- Branch: `12-m2-render-deployment`
- Commit: `d052881` — `Configure Render Docker deployment`
- The same verified commit was fast-forwarded to `main`.
- Branch CI run: `43654589310`
- Ruff, pytest, and Docker build all passed.
- Local pre-push checks: YAML parsed, Ruff passed, `117` tests passed in about `403.26 s`, secret/diff checks passed.

## Render Configuration

`render.yaml` defines a Docker web service with:

- name `legal-aid-rag-api`;
- GitHub repository and `main` branch;
- Docker context/Dockerfile at repository root;
- health path `/health`;
- one worker through the existing Docker command;
- `autoDeployTrigger: checksPass`;
- nonsecret production variables in configuration; and
- `PINECONE_API_KEY` and `GROQ_API_KEY` as `sync: false` protected values.

Render resources created during the manual setup:

- Blueprint name: `legal-aid-rag-m2`
- Blueprint ID: `exs-dbsu96gae00c73941ueg`
- Service ID: `srv-dbsv7n67bikc73cj9tv0`
- Deploy ID: `dep-dbsv7nm7bikc73cj9vcg`
- Intended URL: `https://legal-aid-rag-api.onrender.com`

Secrets were entered manually in Render. They must remain protected and must never be copied into source, screenshots, reports, or this handoff.

## Observed Failure

Key Render log sequence:

```text
INFO: Started server process [1]
INFO: Waiting for application startup.
No open ports detected...
Port scan timeout reached...
Out of memory (used over 512Mi)
```

The startup lifespan constructs retrieval dependencies before startup completes. Therefore, the repeated “no open ports” message is a consequence of startup/OOM, not evidence that `${PORT}` or Uvicorn host binding is wrong.

## Memory Evidence

Approximate local process measurements during staged imports/initialization:

| Stage | Resident memory |
|---|---:|
| Plain Python | 14.4 MiB |
| FastAPI/Uvicorn | 43.9 MiB |
| Plus MLflow | 139.4 MiB |
| Plus `sentence-transformers` import | 446.1 MiB |
| Plus embedding model | 498.9 MiB |

These figures are already at the Render Free ceiling before the reranker, BM25 corpora, Pinecone client, and normal runtime overhead are fully resident. Trimming a few small files or dependencies cannot reliably solve this limit.

## Hosting Decision Required

Do not keep redeploying the same build on Render Free; it will reproduce the OOM and consume time.

Recommended options:

1. **Google Cloud Run** — preferred technical fit. Deploy the existing Docker image with at least 2 GiB RAM, one instance initially, and scale-to-zero. Billing setup is required even when usage stays within free allowances.
2. **Paid Render** — simplest continuation of the current Blueprint. A plan near 2 GiB was observed around USD 25/month. Changing to a paid service requires explicit user approval.
3. **Oracle Always Free ARM VM** — more memory, but more manual server/container setup and possible regional capacity limitations.
4. **Modal or another compute-credit platform** — potentially workable, but usually requires a platform wrapper rather than deploying the Docker service unchanged.
5. **Architectural lite mode** — move embedding/reranking outside the web process or replace local models with hosted inference. This changes runtime architecture and must be evaluated/retested rather than treated as simple deployment configuration.

Hugging Face free/static/Gradio options do not meet the current Docker API requirement or memory need. Koyeb/Render-class 512 MiB free instances are also insufficient for the current process.

## Resume Procedure

```powershell
git fetch origin
git switch 12-m2-render-deployment
git pull --ff-only origin 12-m2-render-deployment
git status
.venv\Scripts\ruff.exe check src tests gradio_app.py
.venv\Scripts\python.exe -m pytest -q
```

Then choose exactly one path:

- **Cloud Run:** configure a project, Artifact Registry, billing/budget alerts, secret variables, 2 GiB+ memory, port 8000/`PORT`, and one initial instance. Deploy the verified SHA, then record service/revision IDs and public smoke tests.
- **Paid Render:** obtain cost approval, change only the plan/resource setting, redeploy the same verified SHA, and capture health/docs/predict evidence.
- **Lite refactor:** create a separate branch, measure memory after each change, rerun all 117+ tests and core response-quality checks, and update the architecture documentation.

## Deployment Acceptance Gate

Do not call branch 12 complete until all are true:

- hosting reports a healthy deployment;
- public `/health` and `/docs` respond;
- a public `/predict` call succeeds for at least one strategy;
- ideally both `recursive` and `fixed` are smoke-tested;
- no secrets appear in logs/responses;
- cold-start time and memory are documented;
- deployment/revision ID, Git SHA, timestamp, and endpoint are recorded in `MLOPS_LOG.md` and README.

Only then create branch 13.
