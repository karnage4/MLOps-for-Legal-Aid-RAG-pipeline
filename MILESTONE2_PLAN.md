# Milestone 2 Implementation Plan

## Objective

Package the Milestone-1 FastAPI RAG service as a reproducible Docker image,
validate it automatically with GitHub Actions, deploy it to Render, expose
basic operational monitoring, and document redeployment and rollback.

Milestone 2 starts from the completed `9-post-m1-fixes` branch. Work proceeds
through numbered, cumulative branches. A new branch is created only after the
current phase passes its acceptance checks and is pushed.

## Initial audit

| Requirement | Starting state | Required work |
| --- | --- | --- |
| FastAPI `/health` | Implemented and unit-tested | Exercise it inside the built container |
| Dockerfile | Obsolete Streamlit command on port 7860 | Replace with production FastAPI/uvicorn image |
| Runtime RAG artifact | BM25 JSON exists locally but is ignored by Git | Package the generated recursive BM25 artifact without the raw corpus |
| Docker healthcheck | Missing | Add an image healthcheck that calls `/health` |
| Docker build context | No `.dockerignore` | Exclude secrets, virtual environments, MLflow state, reports, caches, and raw data |
| CI | Missing | Add Ruff, pytest, and Docker build jobs |
| `main` branch | No local or remote `main` branch is currently visible | Establish `main` from a verified branch before the CI acceptance test |
| Deployment | Missing | Add Render configuration and deploy with protected environment variables |
| Monitoring | Missing | Add request logging plus request-count and average-latency metrics |
| Operations documentation | Missing | Add redeploy and image-tag rollback procedures |
| M2 contributions | M1 document only | Replace with an M2 member-to-task table on the submission branch |

## Phase 10 — containerization

Branch: `10-m2-containerization`

1. Replace the obsolete Dockerfile with a small Python 3.12 FastAPI image.
2. Add a production-only dependency file so report, notebook, UI, and test
   packages are not installed in the deployed API image unless required.
3. Add `.dockerignore`; explicitly exclude `.env`, `.env.txt`, `.git`, `.venv`,
   caches, local MLflow databases/runs, reports, raw corpus files, and tests.
4. Package the generated recursive BM25 runtime JSON used by the API. Do not
   package the raw legal corpus or any credentials.
5. Run `uvicorn src.app:app` on `${PORT:-8000}`, expose port 8000, and add a
   Docker `HEALTHCHECK` that calls `/health`.
6. Build the image locally, start it with environment variables supplied at
   runtime, verify `/health`, `/docs`, and one `/predict` request, then stop it.
7. Scan image history and contents to confirm no secret file was copied.

Acceptance gate:

- clean Docker build;
- container reports healthy;
- `/health` and `/docs` respond;
- one real `/predict` request succeeds;
- existing pytest suite remains green;
- no credentials are present in the image.

## Phase 11 — CI

Branch: `11-m2-ci`

1. Add Ruff configuration and resolve only genuine source/test lint findings.
2. Add `.github/workflows/ci.yml` with separate visible jobs:
   - `lint`: install Ruff and run `ruff check`;
   - `test`: install test dependencies and run `pytest`;
   - `build`: build the Docker image on pushes to `main`.
3. Also run lint and tests on M2 branch pushes and pull requests so failures
   are caught before merging.
4. Establish or confirm the GitHub `main` branch from a verified project tip;
   do not overwrite unrelated history. Confirm that a push/merge to `main`
   triggers the required Docker build job.
5. Use dependency caching where safe and never expose repository secrets to
   pull-request jobs.
6. Push and verify all GitHub Actions checks in the GitHub UI.

Acceptance gate:

- lint job passes;
- all tests pass with a visible test job;
- Docker build job passes on `main` according to the milestone requirement;
- an intentionally failing check can be demonstrated locally or in a temporary
  commit without retaining the failure in branch history.

## Phase 12 — Render deployment

Branch: `12-m2-render-deployment`

1. Add `render.yaml` (or document equivalent dashboard configuration) for a
   Docker web service with `/health` as the health-check path.
2. Configure only secret names in source control. Store `PINECONE_API_KEY` and
   `GROQ_API_KEY` in Render's protected environment settings.
3. Configure the production model/retry variables explicitly and run one
   uvicorn worker to avoid duplicating the memory-heavy embedding/reranker
   models.
4. Deploy from GitHub and verify the public `/health`, `/docs`, and `/predict`
   endpoints.
5. Record the Render deploy identifier, Git commit, image/build identifier,
   deployment time, cold-start behavior, and endpoint evidence.

Acceptance gate:

- Render reports a successful healthy deployment;
- the public endpoint is reachable;
- one public prediction succeeds;
- no secrets appear in logs or API responses;
- deployment resource usage is compatible with the selected Render plan.

## Phase 13 — monitoring and runbook

Branch: `13-m2-monitoring-runbook`

1. Add FastAPI middleware that records structured request logs, response
   status, and elapsed time without logging request bodies or credentials.
2. Add a simple operations endpoint returning at least:
   - process uptime;
   - total request count;
   - successful and failed request counts;
   - average request latency.
3. Add unit tests for successful requests, failed requests, metric increments,
   and latency calculation. Document that in-memory counters reset after a
   Render restart.
4. Redeploy and capture evidence from Render logs and the monitoring endpoint.
5. Add `RUNBOOK.md` covering health diagnosis, log inspection, redeployment,
   environment variables, rollback to a known Git commit/image tag, and
   post-rollback verification.
6. Update README with the live URL, deployment steps, monitoring example, and
   `curl` examples for `/health`, `/predict`, and the monitoring endpoint.

Acceptance gate:

- monitoring values change after requests;
- average latency is demonstrable;
- Render logs contain one safe structured line per request;
- redeploy and rollback procedures are executable and unambiguous.

## Phase 14 — M2 submission

Branch: `14-m2-submission`

1. Tally every project-document requirement against repository and live-service
   evidence.
2. Update `CONTRIBUTION.md` with the same student names/ERP IDs and an M2-only
   member-to-task table based on actual completed work.
3. Add final screenshots/evidence for Docker health, GitHub Actions, Render,
   public API calls, logs, and monitoring.
4. Run the complete pytest suite, Ruff, Docker build, container smoke test, and
   public endpoint smoke test one final time.
5. Verify ignored secrets, repository cleanliness, README links, and rollback
   instructions; then tag the submission commit if desired.

Acceptance gate:

- every Milestone-2 rubric item has code, documentation, and test evidence;
- final CI is green;
- deployed service is healthy or README explicitly records that it was shut
  down after grading;
- submission branch is pushed and reproducible from a fresh clone.

## Planned branch chain

```text
9-post-m1-fixes
  -> 10-m2-containerization
  -> 11-m2-ci
  -> 12-m2-render-deployment
  -> 13-m2-monitoring-runbook
  -> 14-m2-submission
```

Branches are cumulative: each new phase begins from the verified tip of the
previous branch. The M1 branches remain unchanged as historical evidence.
