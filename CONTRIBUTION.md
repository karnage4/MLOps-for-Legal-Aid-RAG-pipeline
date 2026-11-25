# Milestone 2 Contribution Document

## Team Members

| Member | Student name | ERP ID |
| --- | --- | --- |
| Person 1 | Karam Hussain | 27857 |
| Person 2 | Aadesh Panjwani | 29303 |
| Person 3 | Muhammad Rayyan Khan | 28410 |

## Milestone 2 Task Division

| Member | Responsibility area | Milestone 2 deliverables |
| --- | --- | --- |
| Karam Hussain | Pipeline/API validation and reproducibility | Verified the data and runtime-artifact contract used by the container; reviewed fixed and recursive retrieval routing through the FastAPI service; manually checked `/health`, `/docs`, and `/predict` behavior and request validation; reviewed local Docker/startup instructions and deployment curl examples; checked architecture, API, and reproducibility documentation for consistency with the existing RAG pipeline. |
| Aadesh Panjwani | Containerization, CI, and technical integration | Planned the cumulative Milestone 2 branch workflow; implemented the initial FastAPI Docker image and secret-safe build context; versioned and integrated the fixed and recursive BM25 serving artifacts; configured Ruff, pytest, and Docker-build GitHub Actions jobs; established and verified CI on `main`; prepared the initial Render deployment configuration; reviewed the container, CI, and submission integration. |
| Muhammad Rayyan Khan | Deployment, monitoring, operations, and final QA | Optimized the image for CPU-only PyTorch and preloaded embedding/reranking models; added Docker Compose and completed local healthy-container and memory verification; deployed and troubleshot the service on Azure Container Apps; hardened secret entry and Pinecone startup retries; implemented request metrics, privacy-safe structured logs, and request IDs; wrote the runbook and rollback procedure; mounted and verified the public Miras interface; captured deployment/CI evidence and completed fresh-clone, public endpoint, and final CI verification. |

## Detailed Deliverables

### Karam Hussain — Pipeline/API validation and reproducibility

- Reviewed the prepared-data and runtime-artifact assumptions required by a
  clean container build.
- Checked that recursive and fixed chunking select the correct BM25 artifact
  and Pinecone namespace without changing the evaluated RAG architecture.
- Manually validated the FastAPI health, prediction, documentation, and invalid
  request behavior used in the deployment smoke tests.
- Reviewed local Docker and curl instructions so another user can reproduce the
  API from the documented environment variables and serving artifacts.
- Cross-checked the deployment architecture and README descriptions against the
  pipeline, API schema, and existing Milestone 1 evaluation results.
- Supported final documentation review and submission-readiness checks.

### Aadesh Panjwani — Containerization, CI, and technical integration

- Converted the service packaging from the obsolete interface command to a
  FastAPI/Uvicorn Docker deployment on port 8000 with a `/health` healthcheck.
- Added the production dependency/build-context separation and ensured secrets,
  local MLflow state, raw data, virtual environments, and reports were excluded
  from the image context.
- Integrated the versioned 3,593-entry recursive and 883-entry fixed BM25
  artifacts and preserved strategy-specific Pinecone routing.
- Added Ruff configuration and separate visible GitHub Actions jobs for lint,
  the complete pytest suite, and the Docker build.
- Verified successful CI runs on the feature branch and `main` and coordinated
  the numbered Milestone 2 branch handoffs.
- Prepared the initial Render Blueprint and helped document why its 512 MiB
  free plan could not host the measured RAG process.

### Muhammad Rayyan Khan — Deployment, monitoring, operations, and final QA

- Reworked the container to install CPU-only PyTorch, bake both retrieval
  models into the image, and run without Hugging Face network access at runtime.
- Added Docker Compose and measured local startup, predictions, image size,
  memory, health status, and secret safety under a 2 GiB limit.
- Moved hosting from the failed Render free-plan experiment to Azure Container
  Apps and verified public recursive/fixed predictions through Groq.
- Diagnosed and corrected the initial Azure secret-entry failure, then added
  validated, hidden key input and automated key-rotation support.
- Implemented thread-safe request counts, success/failure totals, average
  latency, per-route metrics, structured JSON access logs, and correlation IDs
  without logging questions, answers, headers, or credentials.
- Wrote `RUNBOOK.md` covering checks, diagnosis, redeployment, rollback, key
  replacement, shutdown, and restoration.
- Deployed the Miras interface with the API, captured public-service and CI
  evidence, and completed fresh-clone lint, test, Docker, local-container,
  public-endpoint, and final GitHub Actions verification.

## Shared Review

The team reviewed the final architecture, live-service behavior, monitoring
evidence, documentation, and submission checklist. API credentials remain in
local or managed-platform secret stores and are not part of the repository.
