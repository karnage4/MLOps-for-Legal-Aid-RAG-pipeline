# Milestone 2 Live Demo Presentation

## Deck brief

- **Format:** 16:9 widescreen
- **Core deck:** 7 slides
- **Target video length:** 5 minutes 30 seconds
- **Slide time:** about 90 seconds total
- **Live demonstration time:** about 4 minutes
- **Audience:** instructor and teaching assistant
- **Visual style:** dark navy background, white text, cyan accents, and one
  evidence visual per slide
- **Typeface:** Aptos, Inter, or another clean sans serif
- **Rule:** keep the live terminal and browser demonstrations central. The
  video must show a real run rather than only these slides.

Use each horizontal divider below as a slide boundary. Text under **On slide**
belongs on the slide. Text under **Speaker notes** guides the presenter and
should not appear on the slide.

---

## Slide 1: Legal-Aid RAG MLOps Pipeline

### On slide

**Legal-Aid RAG MLOps Pipeline**

Milestone 2: Containerization, CI/CD, deployment, and monitoring

Karam Hussain · 27857<br>
Aadesh Panjwani · 29303<br>
Muhammad Rayyan Khan · 28410

### Visual

Use `../report/m2-evidence/00-live-interface.png` as a right-side crop. Keep the
title and team information on the left.

### Speaker notes, 15 seconds

> We extended our legal-inheritance RAG project into a reproducible MLOps
> service. Milestone 2 packages the existing evaluated pipeline in Docker,
> validates it through GitHub Actions, deploys it on Azure Container Apps, and
> exposes live request monitoring.

Transition directly to the architecture slide.

---

## Slide 2: Deployed system architecture

### On slide

```text
Legal PDF corpus
      ↓
Validated fixed and recursive chunks
      ↓
BM25 artifacts + Pinecone vectors
      ↓
Semantic or hybrid retrieval
      ↓
Groq grounded generation + MLflow tracing
      ↓
FastAPI + Miras interface
      ↓
Docker image on Azure Container Apps
```

**Serving routes**

`/health` · `/predict` · `/docs` · `/stats` · `/ui`

### Visual

Turn the pipeline into a simple left-to-right diagram. Use one icon for the
corpus, one for retrieval, one for generation, and one cloud/container icon.
Do not use dashboard cards.

### Speaker notes, 20 seconds

> The deployment preserves the evaluated RAG architecture. The container holds
> both BM25 artifacts and the pretrained embedding and reranking models.
> Pinecone supplies dense retrieval, Groq generates grounded answers, and
> MLflow records the nested query, retrieval, and generation trace. FastAPI and
> the Miras interface share the same loaded retrievers.

The `/stats` route is included here as the required target. Do not record the
final presentation until that route works locally and on Azure.

---

## Slide 3: Project Focus

### On slide

**Extensive MLflow experimentation and LLM-as-a-Judge evaluation**

- 8 retrieval configurations
- 2 chunking methods: fixed and recursive
- Semantic retrieval versus hybrid retrieval with reranking
- Top-K values of 3 and 5
- Controlled prompt, model, and temperature comparisons
- Versioned dataset, expectations, judges, and nested traces

**Why this focus fits**

RAG quality depends on retrieved evidence, grounding, and generation behavior.
A training-loss curve does not measure those outcomes.

### Visual

Use a clean experiment matrix: two chunking rows crossed with semantic and
hybrid columns, with small `K=3` and `K=5` labels. Avoid a dense screenshot on
this slide.

### Speaker notes, 25 seconds

> This training-exempt RAG project needs inference evaluation rather than an
> unrelated supervised training cycle. We used MLflow to compare every
> chunking, retrieval, and top-K combination. We then held retrieval constant
> and changed the prompt, model, and temperature one at a time. Versioned judges
> measured fact coverage, relevance, retrieval quality, and groundedness.

---

## Slide 4: Quality and latency trade-off

### On slide

| Configuration | Role | Fact coverage | Warm retrieval mean |
| --- | --- | ---: | ---: |
| Fixed + hybrid + top-5 | Quality champion | 0.7222 | 6,975 ms |
| Fixed + semantic + top-3 | Speed challenger | 0.5833 | 528 ms |

**Speed challenger:** 92.4% lower mean retrieval latency<br>
**Quality cost:** 0.1389 lower fact coverage

### Visual

Use one native scatter plot:

- X-axis: mean retrieval latency in milliseconds
- Y-axis: expected-fact coverage
- Highlight the quality champion and speed challenger
- Add direct labels instead of a separate legend

### Speaker notes, 20 seconds

> The experiments produced a clear operating choice. Fixed hybrid top-5 gives
> the strongest fact coverage. Fixed semantic top-3 reduces mean retrieval
> latency by 92.4 percent, with a 0.1389 reduction in coverage. We retained the
> quality champion while documenting the speed configuration for latency
> sensitive use.

---

## Slide 5: Milestone 2 delivery

### On slide

| Area | Implemented result |
| --- | --- |
| Docker | CPU-only image, non-root user, baked models, two BM25 artifacts |
| CI | Separate Ruff, pytest, and Docker-build jobs |
| Deployment | Azure Container Apps, 1 vCPU, 2 GiB, zero to one replicas |
| Monitoring | Request count, failures, average latency, safe JSON logs |
| Operations | Redeploy, rollback, key rotation, shutdown, and restore runbook |

**Measured deployment facts**

- 15.1 s recorded cold start
- 756.1 MiB measured peak during local container verification
- Render free failed at 512 MiB, so deployment moved to Azure

### Visual

Use `../report/m2-evidence/04-github-actions-run.png` on the right. On the left,
show only the five short implementation lines. Put the three measured facts in
a small footer line.

### Speaker notes, 20 seconds

> The production image uses CPU-only PyTorch and runs one Uvicorn worker. CI
> checks lint, the full test suite, and the Docker build on every push. Render's
> free plan could not hold the measured process, so we deployed the same image
> to Azure with 2 GiB and scale to zero. The runbook covers normal operation and
> rollback.

---

## Slide 6: Live demonstration

### On slide

**Local container**

```bash
docker compose up --build --detach
docker compose ps
```

1. Open `/docs`
2. Call `/health`
3. Call `/predict`
4. Send more predictions and refresh `/stats`

**Then show**

- Green Ruff, pytest, and Docker-build jobs in GitHub Actions
- A small push automatically starting a new workflow
- Live Azure `/health`, `/predict`, and `/stats`

### Visual

Keep this slide visually sparse. Use it only as the transition into the real
terminal and browser demonstration.

### Speaker notes, about 3 minutes 40 seconds

1. Run the documented Docker Compose command from a clean repository view.
2. Run `docker compose ps` and point to `healthy` in the STATUS column.
3. Open local `/docs`, execute `/health`, and send one `/predict` request.
4. Open local `/stats`, note the count, send two more predictions, then refresh
   `/stats` and show the count and average latency changing.
5. Open GitHub Actions and show separate green lint, test, and build jobs.
6. Show that a harmless push started a new workflow automatically.
7. Open the Azure service and demonstrate live `/health`, `/predict`, and
   `/stats`. Wake the service before recording to avoid spending the video on
   the scale-to-zero cold start.

Never display `.env`, Azure secret values, tokens, or subscription identifiers.

---

## Slide 7: Current result and next step

### On slide

**Milestone 2 result**

- Reproducible local container with a working healthcheck
- Automated lint, test, and image-build pipeline
- Public RAG service with observable request volume and latency
- Detailed MLflow evaluation as the project focus

**Next step**

Expand the held-out dataset and use evaluation thresholds to check retrieval or
prompt changes before deployment.

### Visual

Use `../report/m2-evidence/01-live-interface-answer.jpg` as a right-side crop. Put
the result and next step on the left.

### Speaker notes, 10 seconds

> Milestone 2 gives us a reproducible container, automated checks, a public
> service, and live monitoring. Our next step is to expand the evaluation set
> and turn the existing MLflow judges into deployment gates for retrieval and
> prompt changes.

End the recording here. Confirm the final duration is between 4 and 6 minutes
and never above 7 minutes.

---

# Backup slides

Do not show these unless the instructor asks a question. They do not count
toward the planned 5:30 flow.

## Backup A: Technology stack

| Layer | Technology |
| --- | --- |
| Data preparation | Python, PDF extraction, validated fixed and recursive chunks |
| Retrieval | Pinecone, BM25, reciprocal-rank fusion, cross-encoder reranking |
| Generation | Groq API, Qwen and GPT-OSS comparison |
| Evaluation | MLflow tracking, datasets, judges, traces, JSON artifacts |
| Service | FastAPI, Pydantic, Uvicorn, Gradio |
| Delivery | Docker, GitHub Actions, Docker Hub, Azure Container Apps |
| Quality | Ruff, pytest, local and public smoke tests |

## Backup B: Monitoring response

Show the live `/stats` JSON with:

- uptime;
- total request count;
- success and failure counts;
- average latency; and
- per-route values.

Explain that in-memory counters reset after a restart or scale-to-zero event.
The structured logs persist in the hosting platform's log stream.

## Backup C: Deployment decision

| Platform | Outcome |
| --- | --- |
| Render free | Process exceeded the 512 MiB memory limit during startup |
| Azure Container Apps | Healthy at 2 GiB, public HTTPS endpoint, scale to zero |

The architecture stayed unchanged. Only the hosting platform and resource limit
changed.
