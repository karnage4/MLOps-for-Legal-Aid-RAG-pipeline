# Milestone 2 Video Script and Checklist

Target duration: **5 minutes 30 seconds**. The required range is 4–6 minutes;
the absolute marking limit is 7 minutes. Record a real run rather than slides.

## Before recording

- Wake the scale-to-zero Azure service and verify `/health`.
- Confirm `.env` exists locally but is never opened on screen.
- Confirm Docker Desktop has at least 2 GiB available.
- Start from a clean terminal with the repository root visible.
- Prepare a harmless documentation-only change that can be pushed to trigger
  CI, or record immediately after such a push.
- Open GitHub Actions, MLflow, and the live Azure URL in separate tabs.
- Do not show Azure secret values, environment-variable values, subscription
  identifiers, tokens, or private dashboards containing credentials.
- Do not begin final recording until the required live `/stats` endpoint has
  been implemented, tested, deployed, and documented.

## 0:00–0:35 — Introduction

> We built a legal-inheritance retrieval-augmented generation system. The
> pipeline prepares and validates a legal PDF corpus, supports fixed and
> recursive chunking, retrieves evidence through Pinecone and BM25, generates
> grounded answers through Groq, and tracks evaluation in MLflow. The service
> is exposed through FastAPI and the Miras Gradio interface, packaged with
> Docker, tested in GitHub Actions, and deployed on Azure Container Apps.

State the Project Focus: extensive MLflow experimentation and LLM-as-a-Judge
evaluation for RAG.

## 0:35–1:45 — Clean local start

Show the GitHub repository and README. In the clean local checkout run:

```bash
docker compose up --build --detach
docker compose ps
```

Open `http://127.0.0.1:8000/docs`, call `GET /health`, then call `POST /predict`
with a short inheritance-law question. Point out request validation, the
selected chunking/retrieval profile, sources, model/provider, and latency.

## 1:45–2:15 — Docker

Briefly show the Dockerfile:

- Python 3.12 and CPU-only PyTorch;
- preloaded embedding and reranking models;
- non-root user;
- both versioned BM25 artifacts;
- port 8000 and one Uvicorn worker; and
- Docker `HEALTHCHECK` calling `/health`.

Run `docker ps` and point to `healthy` in the STATUS column.

## 2:15–2:55 — Monitoring

Open local `/stats` and note the current request count and average latency.
Send two or three `/predict` requests, refresh `/stats`, and show that the
request count and latency changed. Explain that logs record route, status,
request ID, and duration but not questions, answers, headers, or API keys.

This segment is pending until `/stats` is added; `/metrics` alone does not match
the revised instructor checklist.

## 2:55–3:35 — CI

Open the GitHub Actions page and show the separate green jobs:

- Ruff lint;
- Pytest; and
- Docker build.

Show that a small push automatically starts a new workflow. Do not wait for a
long Docker build on camera; show the new run triggered, then open the latest
completed green run. A deliberately failing run is optional.

## 3:35–4:30 — Live deployment

Open the public Azure URL. Call the deployed `/health`, `/predict`, and `/stats`
using the README curl examples or `/docs`. Mention Azure Container Apps, one
vCPU, 2 GiB, zero-to-one replicas, platform secrets, and scale-to-zero cold
starts. Briefly state that Render free was tested first but its 512 MiB limit
was below the measured process memory.

## 4:30–5:20 — Project Focus

Show MLflow evidence for:

- the managed held-out dataset and expected facts;
- versioned LLM judges;
- nested query, retrieval, and generation traces;
- all eight retrieval configurations; and
- the prompt, model, and temperature comparison.

Conclude with the quality champion (`fixed-hybrid-top5`) and latency challenger
(`fixed-semantic-top3`). State that the next step is to use a larger evaluation
set or production feedback to gate future retrieval and prompt changes.

## 5:20–5:30 — Close

> This demonstrates the complete baseline pipeline, automated CI, healthy
> container, live deployment, monitoring, and our MLflow/LLM-as-a-Judge focus.

## Final recording checklist

- Duration is between 4 and 6 minutes and never above 7 minutes.
- The video shows a real local container start, not only slides.
- `docker ps` visibly shows `healthy`.
- Local `/health`, `/predict`, and `/stats` are demonstrated.
- CI lint, test, and build jobs are green and an automatic trigger is shown.
- Live `/health`, `/predict`, and `/stats` are demonstrated.
- Project Focus and next steps are covered in the last 30–60 seconds.
- No credential, secret, private key, or `.env` value appears.
