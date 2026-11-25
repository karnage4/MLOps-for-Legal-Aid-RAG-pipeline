# MLOps Implementation Log

This file is the running record for introducing MLOps features one at a time. Update it after every implementation, manual test, and branch transition.

## Working rules

- `Legal-Aid-RAG-main/` is the read-only baseline and must not be edited.
- Preserve the original RAG stack and variable names unless a later phase explicitly changes them.
- Add one MLOps capability per branch and manually verify it before starting the next capability.
- Never commit `.env`, API tokens, `mlflow.db`, `mlruns/`, downloaded models,
  raw corpus files, or general generated output. The reviewed recursive and
  fixed BM25 JSON files are the sole exceptions because they are the required
  versioned serving artifacts for the Milestone-2 container.
- Record real run IDs, trace IDs, results, and failures; do not substitute estimated metrics.

## Phase overview

| Phase | Feature | Branch | Implementation | Manual verification | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | MLflow integration | `1-mlflow-integration` | Complete | Smoke, UI, and full RAG trace passed | Accepted |
| 2 | RAG evaluation experiments | `2-rag-evaluation-experiments` | Complete | All six steps and final eight-run suite passed | Accepted |
| 3 | GenAI datasets and judges | `3-mlflow-genai-datasets-judges` | Complete | Dataset, judges, and 8/8 profile evaluation passed | Accepted |
| 4 | FastAPI service | `4-fastapi-service` | Complete | Automated and manual endpoint tests passed | Accepted |
| 5 | Milestone documentation | `5-milestone-documentation` | Complete | README, rubric audit, and contribution allocation completed | Accepted |
| 6 | Local Gradio interface | `6-gradio-interface` | Complete | Unit, HTTP, live RAG, fallback, and MLflow trace passed | Awaiting visual acceptance |
| 7 | Prompt/model/temperature inference ablation | `7-inference-ablation-evaluation` | Complete | 4/4 valid; structured prompt selected | Active |
| 8 | Post-M1 correctness and latency fixes | `9-post-m1-fixes` | Complete | 8/8 core-v2 evaluations and 150-request latency comparison passed | Active |
| 9 | M2 containerization | `10-m2-containerization` | Complete | Clean GitHub Linux image build passed; live container checks were later completed locally (9a) and on Azure (11) | Build accepted |
| 9a | M2 CPU-only Docker image | `10a-m2-docker-cpu-image` | Complete | Local Docker build, healthy container, `/health`, `/docs`, and both `/predict` strategies passed; 756.1 MiB peak memory | Verified; included in `10-m2-containerization` |
| 10 | M2 continuous integration | `11-m2-ci` | Complete | Ruff, 117 tests, and Docker build passed in GitHub Actions | Accepted |
| 11 | M2 deployment (Render, then Azure Container Apps) | `12-m2-render-deployment` | Complete | Render free plan ran out of memory at startup; Azure Container Apps in `swedencentral` passed public `/health`, `/docs`, and `/predict` for both strategies | Live; awaiting team acceptance |
| 13 | M2 submission | `14-m2-submission` | Revised documentation complete; `/stats` implemented (live redeploy pending); video pending | Rubric table, Project Focus, M2 contribution allocation, AI disclosure, video plan, Render cleanup, screenshots, fresh-clone verification, public smoke test, and CI evidence completed | Redeploy with `/stats`, then record the 4-6 minute video |
| 12 | M2 monitoring, runbook, and public interface | `13-m2-monitoring-runbook` | Complete | Live `/metrics` counts and JSON access logs verified in Azure; `RUNBOOK.md` added; the Miras Gradio interface serves the public URL and answered a question in a browser; `/stats` added and verified in the local container (live redeploy pending) | Live; awaiting team acceptance |

## Phase 9 — M2 containerization

### Objective

Replace the obsolete Streamlit container with a reproducible, secret-safe
FastAPI image suitable for Render, including a real `/health` Docker
healthcheck and both versioned RAG serving artifacts.

### Implemented

- Replaced the port-7860 Streamlit command with one unprivileged uvicorn worker
  serving `src.app:app` on `${PORT:-8000}`.
- Added a 180-second cold-start grace period and a Docker healthcheck calling
  `/health`.
- Added pinned API-only dependencies and an allow-listed Docker context that
  cannot copy `.env`, local MLflow data, virtual environments, tests, reports,
  caches, or raw corpus files.
- Added the validated 3,593-entry recursive and 883-entry fixed BM25 JSON files
  as versioned runtime corpus artifacts. `/predict` validates a
  `chunking_strategy` choice and routes to the matching BM25 artifact and
  Pinecone namespace. Both retrievers share one embedding model, reranker, and
  Pinecone client. Secrets remain runtime environment variables.
- Added static regression tests for the command, port, worker count,
  healthcheck, build-context allow list, and runtime artifact presence.

### Verification status

- Focused API/container suite: 10 passed; the broader API/container/retrieval
  profile suite passed all 26 tests.
- Complete repository regression: 117 passed with one pre-existing MLflow
  SQLAlchemy deprecation warning.
- Local environment dependency check: no broken requirements.
- Recursive runtime artifact: 3,593 IDs, 3,593 non-empty texts, SHA-256
  `51C8B6CD081F954F3E5D94D4676E98FFCCB46556C6BC5FD88A736207447DCDC2`.
- Fixed runtime artifact: 883 IDs, 883 non-empty texts, SHA-256
  `E878377705440DC9E599ECD3E3C2B55CD69299A072D23B7F1D806FD9736A7F15`.
- Docker build attempt did not reach Dockerfile execution. Docker Desktop 4.91
  failed while registering its WSL distribution with
  `Wsl/Service/RegisterDistro/CreateVm/HCS/ERROR_FILE_NOT_FOUND`. A safe WSL
  shutdown/restart reproduced the host failure. No Docker data was reset or
  deleted.
- GitHub Actions run `43652143632` built and inspected the same image
  successfully on a clean Linux runner. This closes the clean-build gate; the
  live `/health` and `/predict` container checks will be completed on Render,
  while local verification can be repeated after Docker Desktop is repaired.

## Phase 9a — M2 CPU-only Docker image

### Objective

Fix the branch-10 image inside the containerization phase: remove the unusable
CUDA PyTorch build, stop runtime model downloads, and close the Phase 9 live
container smoke-test gate with a real local run, including the memory figure
that decides which hosts can run the API.

### Implemented

- `Dockerfile` installs `torch==2.15.0` from the PyTorch CPU wheel index before
  `requirements-api.txt` and fails the build if a CUDA torch is present.
- Both Hugging Face models (`all-MiniLM-L6-v2` and
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) are downloaded at build time into
  `HF_HOME=/opt/hf-cache` in a layer before `COPY src`, and `HF_HUB_OFFLINE=1`
  stops the running container from calling huggingface.co.
  `--build-arg PRELOAD_MODELS=0` restores download-on-start.
- Added `compose.yaml` (service `api`, container `legal-aid-rag-api`, image
  `legal-aid-rag-api:local`, port 8000, `env_file: .env`, `mem_limit: 2g`).
- Added two tests to `tests/test_container_config.py` for the CPU wheel index,
  CUDA guard, and preloaded model names matching `src/retrieval.py`; added
  README notes for Windows and macOS and `handoffs/10a-m2-docker-cpu-image.md`.
- Fixed `HANDOUT.md`: requests use `query` (the API rejects `question` with
  HTTP 422) and `/predict` smoke tests use `Invoke-RestMethod`, because Windows
  PowerShell 5.1 strips the inner quotes of a `curl.exe -d '{...}'` body.
- No API, retrieval, or generation code changed. The same Dockerfile change
  was first tested as `12a-m2-docker-cpu-image` on top of branch 12; this
  branch moves it onto branch 10's own code, which still installs `nltk`.

### Verification status

- Tested commit `8a531de` on a Windows 11 laptop (Docker Desktop 4.60.1,
  Engine 29.2.0, containerd image store, 8 CPUs, 3.65 GiB VM memory) with real
  Pinecone and Groq keys from the git-ignored `.env`.
- `tests/test_container_config.py`: 5 passed (pytest 9.2.0, Python 3.12.10).
  `docker build --check`: no warnings. Branch 10 has no CI workflow, so Ruff
  and the full suite were not run on this branch; a local Ruff install failed
  because the laptop disk was full. The identical Dockerfile and test file
  passed GitHub Actions on `12a-m2-docker-cpu-image` (run `44912728419` at
  `2cb2c26` and run `44921194963` at `bc34379`: Ruff, 119 tests, Docker build).
- Image `legal-aid-rag-api:local` (`1e9c7787797f`): 784 MB content size
  (compressed) and 3.18 GB disk usage (unpacked), versus 10.3 GB for the
  original branch-10 CUDA image.
- `docker compose up --build --detach`: 512.6 s, exit code 0, with the base
  image and two early layers cached from an earlier build (pip install
  291.5 s, model preload 97.8 s, image export and unpack 115.2 s).
- First start failed after 24 s: the startup Pinecone index lookup hit a
  transient DNS error (`[Errno -5] No address associated with hostname;
  request to https://api.pinecone.io`), so the container exited with code 3
  (`OOMKilled` false). The laptop and a fresh container resolved the name
  immediately afterwards. Startup has no retry around Pinecone
  initialization, so one DNS failure stops the container; hosted restarts
  can hit the same failure.
- Recreated container: `StartedAt` 10:59:49.488Z to `Application startup
  complete` 11:00:05.891Z = 16.4 s with both retrievers loaded from the baked
  models. `docker compose ps` first showed `(healthy)` at 11:00:11.933Z
  (22.4 s after start, 5-second polling).
- `GET /health`: `status: ok`, `retriever_loaded: true`, strategies
  `recursive` and `fixed`. `GET /docs`: HTTP 200.
- `POST /predict` with the `query` field ("Who inherits property when a person
  dies without a will?"), hybrid retrieval, top-5, Groq `qwen/qwen3.8-27b`,
  no fallback:
  - `recursive`: HTTP 200, 5 chunks, 2 sources; retrieval 3,039 ms,
    generation 579 ms, total 3,618 ms (first request; client round trip
    11,656 ms). The 903-character answer said the context did not list the
    intestate succession rules explicitly.
  - `fixed`: HTTP 200, 5 chunks, 3 sources; retrieval 2,255 ms, generation
    1,429 ms, total 3,684 ms; 2,077-character answer.
- Memory after both requests: `docker stats` 600.2 MiB of the 2 GiB limit
  (29.31%); cgroup `memory.peak` since start 792,850,432 bytes = 756.1 MiB;
  `memory.events` recorded 0 OOM events. The `12a` run on branch-12 code
  peaked at 931.7 MiB with 289 MB of block reads versus 61.6 MB here, which
  is consistent with the model files already being in the Docker VM page
  cache, which `memory.peak` partly counts. Use 931.7 MiB as the conservative
  sizing figure.
- Hosting: the 756.1 MiB peak is 1.48 times a 512 MiB Render free instance,
  and even the 600.2 MiB steady state exceeds it, so the free plan cannot run
  this image. On a 2 GiB host headroom is 2,048 - 756.1 = 1,291.9 MiB
  (1,116.3 MiB using the conservative 931.7 MiB peak).
- Secrets: `find / -xdev -name '.env*'` inside the running container found
  nothing; `/app` holds only `src`, `output`, `requirements-api.txt`, and the
  runtime-created `mlflow.db` and `.config`.
- Container logs on the second start had no errors and every request returned
  200; `docker compose down` removed the container and network. This closes
  the Phase 9 image-build and live container smoke-test gate locally.

## Phase 10 — M2 continuous integration

### Objective

Provide visible, automated lint, test, and Docker-build checks for every change
and guarantee that every push to `main` builds the production image.

### Implemented

- Added `.github/workflows/ci.yml` with independent `Ruff lint`, `Pytest`, and
  `Docker build` jobs on pushes, pull requests, and manual dispatches.
- Restricted the workflow token to read-only repository contents and enabled
  branch-aware concurrency cancellation.
- Pinned Ruff 0.17.5 in `requirements-dev.txt` and configured a stable
  correctness baseline (`E4`, `E7`, `E9`, and `F`) for source, tests, and the
  Gradio entry point.
- Resolved seven real baseline findings: unused imports/variables, one
  unnecessary f-string, and the obsolete NLTK production dependency. The two
  Gradio post-environment imports are explicitly documented and ignored.
- Used Python 3.12 and pip caches for CI parity with the Docker image. The
  Docker job performs a real `docker build` and image inspection without
  requiring repository secrets or pushing an image.

### Verification status

- Local Ruff: passed with zero findings.
- Focused API, container, Gradio, and retrieval regression: 32 passed.
- Complete local regression: 117 passed with one pre-existing MLflow
  SQLAlchemy deprecation warning.
- GitHub Actions run `43652143632` (`a683f28`) completed successfully:
  `Ruff lint`, `Pytest`, and `Docker build` all passed. The Docker job built and
  inspected the production image on Ubuntu without credentials.
- Created the previously missing remote `main` branch from verified commit
  `3a3c422` without overwriting any history. Its automatically triggered run
  `43652828766` also passed `Ruff lint`, `Pytest`, and `Docker build`, directly
  demonstrating the rubric-required build on a push to `main`.

## Phase 11 — M2 deployment

### Objective

Serve the containerized API at a public URL on a host whose memory fits the
measured container, with secrets held by the platform and a repeatable
redeploy and rollback path.

### Implemented

- `render.yaml` describes a Render Docker web service on the free plan that
  deploys `main` after CI passes. The team reported that this deployment ran
  out of memory at startup on the 512 MB free plan; no deploy identifier was
  recorded in this log.
- Phase 9a measured the container at 756.1-931.7 MiB peak, so the API moved to
  Azure Container Apps on the team's Azure for Students subscription:
  1 vCPU, 2 GiB, minimum 0 and maximum 1 replica.
- `deploy/azure/deploy-containerapp.sh` (run in Azure Cloud Shell) creates
  `legal-aid-rag-rg`, `legal-aid-rag-env`, and `legal-aid-rag-api`, prompts
  for the two API keys with hidden input and stores them as Container Apps
  secrets, optionally takes a read-only Docker Hub token for authenticated
  pulls, applies the same non-secret settings as `render.yaml`, waits for
  `/health`, and prints the URL. Re-running it with another image tag rolls
  forward or back.
- Student subscriptions cannot run Azure Container Registry build tasks
  (`TasksOperationsNotAllowed`), so the image is built locally and pushed to
  Docker Hub with a Git-SHA tag.
- `init_pinecone` now retries transient Pinecone connection and timeout errors
  (`PINECONE_INIT_MAX_ATTEMPTS`, default 5, linear backoff from
  `PINECONE_RETRY_BASE_SECONDS`). Phase 9a saw one local start fail on a
  single DNS error, and a scale-to-zero host cold-starts often. Two unit tests
  cover a retried success and the final failure.

### Verification status

- `bash -n deploy/azure/deploy-containerapp.sh`: no syntax errors; running it
  without `IMAGE` exits with a usage message.
- Image `docker.io/rayyankhan007/legal-aid-rag-api:75571d7` (manifest list
  `sha256:40d92abae90670d2c624b6a87a11bd5c6bbc9a49082e0ec3a2ce7ba2ba5367bc`,
  0.73 GB compressed) was built locally from `75571d7` and pushed to a public
  Docker Hub repository. Before the push it passed a local smoke test with the
  `.env` keys: `/health` ok about 15 s after start, a grounded Groq answer from
  `/predict`, and 702 MiB memory after the request. The build context
  allow-list kept `.env` out, and a scan of `src/` and both BM25 corpora found
  no key-shaped strings.
- Live deployment on 2026-11-22 in `swedencentral`, the closest of the five
  regions the Azure for Students policy allows (`austriaeast`, `eastasia`,
  `koreacentral`, `polandcentral`, `swedencentral`) to the AWS `us-east-1`
  Pinecone index:
  <https://legal-aid-rag-api.whitesand-d7432c32.swedencentral.azurecontainerapps.io>
- Public checks at 18:42 PKT: `/health` HTTP 200 with both retrievers loaded;
  `/docs` HTTP 200; `/predict` HTTP 200 for `recursive` (server total 1,573 ms
  = 982 ms retrieval + 591 ms generation, 2 sources) and `fixed` (3,402 ms =
  1,679 ms + 1,723 ms, 3 sources), both hybrid top-5 through Groq
  `qwen/qwen3.8-27b` with `fallback_used = false`. A too-short query and an
  unknown request field both returned HTTP 422.

### First-deployment incident

- The first revision never became ready. Every start ended with Pinecone
  `401 UNAUTHENTICATED Invalid API key` and `Application startup failed`, so
  the public URL returned HTTP 503 for more than eight minutes. The image was
  not at fault: the same image answered locally with the `.env` keys. The key
  stored as the Azure secret differed from the local one; the team's original
  key file wraps each value in quotes, the most likely source.
- Fix: both keys were re-entered in Cloud Shell with hidden input and a
  length/prefix check (75 characters starting `pcsk_`, 56 starting `gsk_`),
  then stored with `az containerapp secret set`.
- The script created an Express Container Apps environment, which rejects
  `az containerapp revision restart`, and the Cloud Shell CLI has no
  `az containerapp stop`/`start`. Setting a timestamp variable
  (`KEYS_ROTATED_AT`) with `az containerapp update` started a new revision that
  read the new secrets, and `/health` passed on the first check afterwards.
- Prevention: the deploy script now reads every key through
  `deploy/azure/read-key.sh`. It strips surrounding quotes and spaces, a
  `NAME=` or `export NAME=` prefix, and Windows line endings, requires the
  `pcsk_`/`gsk_` prefixes, asks again up to three times, and never echoes the
  key. `UPDATE_KEYS=1` reuses it to replace both keys on an existing app and
  rolls a new revision with `KEYS_ROTATED_AT`. Twelve tests in
  `tests/test_deploy_read_key.py` cover the clean-up and rejection cases. A dry
  run against stub `az`/`curl` commands confirmed the exact Azure calls for a
  key update, an image-only update, a first deployment with one bad paste, and
  three bad pastes (which stop before any secret or app change).

## Phase 13 — M2 submission

### Objective

Map every Milestone 2 requirement to its implementation and evidence, remove
stale claims, capture evidence of the live service and CI, and confirm that a
fresh clone reproduces the verified state.

### Implemented

- README: a "Milestone 2 at a glance" section with the live links, the branch
  chain, and a requirement / implementation / evidence table; the
  architecture diagram and project-structure list now include the Azure
  deployment, the Miras interface at `/ui`, `/metrics`, `RUNBOOK.md`, and the
  evidence folder.
- Render is labelled throughout as the failed free-plan capacity experiment:
  the README's Render section is condensed to that record,
  `MILESTONE2_PLAN.md` opens with an outcome note mapping its Render steps to
  Azure, `handoffs/12-m2-render-deployment.md` is marked superseded, and the
  Dockerfile and `compose.yaml` comments no longer name Render or Cloud Run as
  hosts. The phase-9 table row no longer says the health check is deferred to
  Render.
- `report/m2-evidence/`: six screenshots (live interface, a live answer,
  `/metrics`, `/docs`, the CI run for the deployed commit, and the CI run on
  `main`) with an index listing the Azure Portal captures that need the
  account holder.
- `RUNBOOK.md` points at branch 14 and records the measured cold start.
- Added the required `Project Focus` section describing the MLflow and
  LLM-as-a-Judge specialization, why it suits RAG, the completed comparisons,
  how it is demonstrated, and the final-project next step.
- Replaced the Milestone 1 contribution file with an M2-specific member-to-task
  allocation for Karam Hussain (27857), Aadesh Panjwani (29303), and Muhammad
  Rayyan Khan (28410), aligned with the implementation history and manual
  validation work.
- Added `AI_USAGE.md` to disclose development assistants, their uses, human
  verification, credential handling, and limitations.
- Added `video-demo/M2_VIDEO_SCRIPT.md`, a timed 5:30 real-run demonstration plan covering
  the clean Docker start, health, prediction, monitoring, CI trigger, live
  deployment, Project Focus, and secret-safe recording checks.
- Added `video-demo/M2_VIDEO_PRESENTATION.md`, a seven-slide presentation source with
  concise on-slide copy, speaker notes, live-demo transitions, evidence-image
  directions, and backup slides for the technology stack, monitoring response,
  and hosting decision.
- Added a concise README `Post-M1 problems and improvements` table that links
  the instructor's latency and correctness feedback to the identified causes,
  branch-9 fixes, and verified before-versus-after measurements.

### Verification status

- Cold start on Azure: after the app had scaled to zero, the first `/health`
  request returned 200 after 15.1 s (`started_at` 18:05:40 UTC). An interface
  question then took 4,078.5 ms (first prediction on the new replica).
- Fresh clone of `14-m2-submission` at `a48712c` from GitHub into an empty
  folder: clean status, no `.env` (only `.env.example`), both BM25 artifacts
  present. Ruff passed on `src`, `tests`, and `gradio_app.py`; the 58 API,
  interface, monitoring, container-config, and deploy-key tests passed with
  the modules loaded from the fresh clone (the laptop has no full ML
  environment; the complete suite runs in CI). `git diff --check` against
  branch 13 was clean.
- Docker image built from the fresh clone (`legal-aid-rag-api:m2-final`, 11 s
  with this laptop's layer cache) and run with the `.env` keys: healthy in
  about 18 s and Docker health status `healthy`; `GET /` 307 to `/ui/`; `/ui/`
  and `/docs` 200; `/predict` 200 for `recursive` (server 2,659 ms, 2
  sources) and `fixed` (3,675 ms, 3 sources) via Groq without fallback; a
  too-short query 422; an interface question completed; `/metrics` counted 9
  requests (8 successes, 1 failure); 9 JSON log lines, no question text, no
  errors; 671.8 MiB memory.
- Public smoke test at 23:09:55 PKT on the live replica: `/health` 200 in
  0.5 s; `GET /` 307 to the `https://` `/ui/` address; `/ui/` and `/docs` 200;
  `/predict` 200 for `recursive` (2,126 ms) and `fixed` (4,316 ms) via Groq
  without fallback; a too-short query 422; `/metrics` 17 requests (11
  successes, 6 failures, including 4 `unmatched`), `POST /predict` mean
  2,321.9 ms.
- GitHub Actions on `a48712c`
  ([run 45053312903](https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline/actions/runs/45053312903)):
  Ruff lint, Pytest (the complete suite on a clean Ubuntu runner), and Docker
  build all passed.
- Revised-requirement audit on 2026-11-25: the existing `/metrics` monitoring
  implementation satisfies the underlying request-count and latency behavior,
  but the instructor's video checklist explicitly names `/stats`; the live
  `/stats` route returned 404 while `/metrics` returned 200. The compatibility
  route, its tests/documentation, redeployment, and final video remain pending.
  Update, same day: `GET /stats` was implemented and verified on branch 13
  (`a5d818a`, see Phase 12) and merged into this branch, and the two local
  check scripts now expect `/stats` to return 200; the live redeploy and the
  video remain.
- Recommended evidence still outstanding: Azure Portal screenshots from the
  account holder, a live rollback rehearsal, and the after-grading decision to
  keep the service scaling to zero or stop it.

## Phase 12 — M2 monitoring and runbook

### Objective

Make request volume, failures, and latency observable on the live service
without logging private content, and document how to operate, redeploy, and
roll back the Azure deployment.

### Implemented

- `src/monitoring.py` adds pure ASGI middleware. For every request except
  `GET /metrics` it records the route, status, and elapsed time, writes one
  JSON line to stdout (`event`, `timestamp`, `request_id`, `method`, `route`,
  `status`, `duration_ms`), and returns the request ID as `X-Request-ID`. It
  never reads bodies, headers, or query strings. Unknown paths are grouped as
  `unmatched`; status 400 and above, and unhandled exceptions (recorded as
  500 and re-raised), count as failures.
- `GET /metrics` returns `started_at`, `uptime_seconds`, `request_count`,
  `success_count`, `failure_count`, and `average_latency_ms`, in total and per
  route, from lock-protected in-memory counters. They reset on every restart,
  redeploy, or scale to zero.
- The Docker `CMD` adds `--no-access-log`, so each request produces exactly
  one structured line instead of an extra uvicorn line.
- `RUNBOOK.md` covers the system summary and identifiers, configuration names,
  routine checks, metrics and logs, a diagnosis table, redeployment,
  key replacement, rollback to a known-good image tag or digest, post-change
  verification, shutdown and restoration, and ownership.
- Public interface: the team asked for the branch-8/9 Miras Gradio interface
  instead of the FastAPI docs page at the public address. `src/web.py` (the
  image's new entry point, `uvicorn src.web:app`) mounts `gradio_app.py` at
  `/ui` with its theme and CSS and redirects `/` there; `/docs`, `/health`,
  `/predict`, and `/metrics` are unchanged. The interface now accepts the
  retrievers the API loaded at startup (`get_components`) instead of loading
  a second copy, and reports each question to the monitor as
  `POST /ui/predict` (200, 422 invalid input, 502 retrieval or generation
  failure, 500 otherwise). Requests served by the mounted interface (page,
  assets, event stream) are not counted. `python gradio_app.py` still runs it
  standalone. `gradio==6.32.0` is pinned in `requirements-api.txt` (it
  resolves with FastAPI 0.161.0, Starlette 1.10.0, and Pydantic 2.14.1). The
  image sets `GRADIO_ANALYTICS_ENABLED=False` and `FORWARDED_ALLOW_IPS=*` so
  the interface builds `https://` URLs behind the Azure proxy.

### Verification status

- Local (FastAPI 0.161.0, Starlette 1.10.0): `tests/test_monitoring.py` (18
  tests: empty state, outcome counting, average latency, the 400 failure
  boundary, uptime, 16,000 concurrent records, endpoint counting that
  excludes `/metrics`, one safe log line per counted request with matching
  `X-Request-ID`, unhandled-error counting with the original exception
  re-raised, and the OpenAPI entry) plus the existing API, container-config,
  and deploy-key tests: 42 passed. Ruff passed on `src`, `tests`, and
  `gradio_app.py`.
- Commit `996bf7f` (monitoring without the interface): GitHub Actions passed
  Ruff, the full Pytest suite, and the Docker build. Its image
  (`sha256:471c12f8bf1b5b64e9b25ad6816a55ea60c080cd951c28593ccc6b3095f54421`)
  ran locally with the `.env` keys: healthy in about 15 s; `/metrics` moved
  from 2 to 6 requests (4 successes, 2 failures, `POST /predict` mean
  2,864.9 ms); exactly one JSON line per request, each `request_id` equal to
  the `X-Request-ID` header; no question text and no uvicorn access lines in
  the container log; 637 MiB memory.
- Live on Azure from 2026-11-22 19:11 PKT. A first evidence run at 19:17
  arrived after more than five idle minutes: two `/predict` calls returned 200,
  then the next requests got HTTP 502 without `X-Request-ID` (from the
  ingress, not the app) until a new replica reported `started_at`
  14:17:46 UTC about 20 s later. This matches the replica being scaled to
  zero as the requests arrived; the runbook lists it. The repeated run at
  19:18 PKT on the warm replica: `/metrics` went from 0 to 4 requests
  (3 successes, 1 failure; `POST /predict` 3 requests, mean 3,083.6 ms;
  `GET /health` 1.7 ms); `/predict` returned 200 for `recursive` (server
  2,343 ms) and `fixed` (3,216 ms) and 422 for a too-short query; every
  response carried an `X-Request-ID`.
- Interface (local, Gradio 6.32.0): `tests/test_web.py` (5 tests: `/`
  redirects to `/ui/`; the page and its config are served with the Miras
  title and are not counted; API routes and the OpenAPI schema are
  unchanged; a question sent through Gradio's call API uses the API's
  already-loaded `fixed` retriever, is counted once as `POST /ui/predict`, and
  logs neither question nor answer; a too-short question is counted as a 422
  failure) and 5 new `tests/test_gradio_app.py` cases (supplied retrievers and
  the 200/422/502/500 outcome mapping). With the monitoring, API,
  container-config, and deploy-key tests: 58 passed; Ruff passed.
- Commit `3062327` (interface): GitHub Actions passed Ruff, the full Pytest
  suite, and the Docker build. Local image build 574 s, 3.29 GB (Gradio adds
  about 130 MB). Pushed as
  `sha256:e0969395e040a38a92369afa4990f72fc42e2ee34d2b8674cbbbdc238c12c125`
  (one push retry after a transient broken-pipe error). Local smoke test with
  the `.env` keys: healthy in about 30 s; `GET /` returned 307 to `/ui/`;
  `/ui/` 200 HTML; the interface config reported the Miras title, custom CSS,
  and theme, and an `https://` root when sent `X-Forwarded-Proto: https`; a
  question through Gradio's call API returned a grounded Groq answer and was
  counted as `POST /ui/predict`; a question clicked through the page in a
  browser was answered; API `/predict` and `/docs` 200; one JSON log line per
  counted request, no question text, no uvicorn access lines; 789 MiB memory.
- Live on Azure from 2026-11-22 20:05 PKT (`started_at` 15:05:58 UTC). At
  20:06-20:08 PKT: `GET /` returned 307 to the `https://` `/ui/` address;
  `/ui/` 200; the interface config reported the Miras title and an
  `https://.../ui` root; `/docs` 200; API `/predict` (`fixed`) 200 with an
  `X-Request-ID`. In the browser, "Can a non-Muslim inherit from a Muslim
  estate?" received a grounded answer. A first automated attempt clicked
  "Ask" before the example text reached the box; the interface rejected the
  empty question ("at least 3 characters") and counted it as a failure, as
  designed. `/metrics` at 20:08:24 PKT: 13 requests (8 successes, 5
  failures); `POST /ui/predict` 3 requests (2 successes, 1 failure, mean
  1,056.1 ms); `POST /predict` 1 request (6,508.7 ms, the first prediction on
  the new replica); `GET unmatched` 4 failures (requests to paths outside the
  app's routes, which the log does not name by design). Azure log stream (`az containerapp logs show ... | grep
  http_request`) showed JSON lines such as
  `{"event": "http_request", ..., "method": "POST", "route": "/ui/predict", "status": 200, "duration_ms": 1648.7}`
  and `{"event": "http_request", ..., "method": "GET", "route": "/", "status": 307, "duration_ms": 3.4}`,
  with no question or answer text.

### `/stats` (added 2026-11-25)

The revised instructor checklist names a `/stats` route. Implemented in
commit `a5d818a`:

- `GET /stats` returns the `/metrics` fields plus `queries`, aggregated over
  `POST /predict` and `POST /ui/predict` by a lock-protected `QueryStats`
  collector fed by the existing middleware (the RAG pipeline is unchanged):
  `total`, `succeeded`, `failed`, `error_rate` (null with no questions),
  `per_minute`, `by_status`, `by_endpoint`, and `latency_ms` (`mean`, `min`,
  `max`, `p50`, `p90`, `p95`, `p99`, linear interpolation) over a 1,000-answer
  window of successful answers. Reads of `/stats` and `/metrics` are not
  counted.
- Privacy: only the method, route name, status, and duration are recorded;
  bodies, query strings, headers, questions, retrieved text, and answers are
  never read.
- Fail-safe: each monitoring step during a request (route lookup, response
  header, timer, metrics, stats, log line) catches its own exceptions and
  logs a warning without request data, so a collector bug cannot break a
  question.
- The Docker healthcheck reads `/health` and `/stats`; the deploy script
  checks `/stats` after `/health`; `scripts/stats_demo.py` (standard library
  only) prints `/stats`, sends three questions and one too-short question,
  prints `/stats` again, and checks the change.

Verification:

- `tests/test_stats.py` (21 tests: fresh-app schema with zeros and nulls;
  counts after 200, 422, and 502 questions; `/stats` and `/metrics` reads not
  counted; no question text, retrieved text, answer, source, or name, email,
  phone, or CNIC in `/stats`, `/metrics`, or the log; OpenAPI entry; a broken
  collector, access log, or route lookup never breaks a question; exact
  mean/min/max/p50/p90/p95/p99 for known latencies; failed questions excluded
  from latency; rate per minute with a fake clock and null at zero uptime;
  the bounded window; 8,000 concurrent records; percentile edge cases) and an
  interface check in `tests/test_web.py`. With the existing monitoring,
  interface, API, container, and deploy-key tests: 79 passed locally; Ruff
  passed on `src`, `tests`, `gradio_app.py`, and `scripts`.
- The deploy script's stubbed dry run passed all four scenarios and printed
  the new `Stats:` line.
- Image `a5d818a` (rebuilt in about 13 minutes because the local layer cache
  had been cleared) run with the `.env` keys: `scripts/stats_demo.py` passed.
  `/stats` moved from 0 to 4 questions (3 succeeded, 1 rejected with 422;
  `error_rate` 0.25; p50 9,872.7 ms, p95 15,825.4 ms; the first answer
  included model warm-up). Docker health status `healthy` with the `/stats`
  healthcheck; 7 JSON log lines, no question text, no monitoring warnings, no
  errors; 788.6 MiB memory.

## Phase 1 — MLflow integration

### Objective

Add experiment tracking and GenAI tracing to the original Streamlit/Pinecone/Hugging Face RAG project without changing its provider keys, retrieval strategy, generation models, evaluation formulas, or deployment port.

### Implemented

- Added `mlflow>=3.8,<4` to `requirements.txt`.
- Added a local SQLite tracking configuration in `src/mlflow_config.py`.
- Added a credential-free conventional-run and three-span trace check in `src/mlflow_smoke_test.py`.
- Added a root RAG chain in `src/pipeline.py`.
- Added a `RETRIEVER` span around BM25 + Pinecone + RRF + reranking.
- Added an `LLM` span around Hugging Face generation while excluding `HF_API_TOKEN`.
- Added a provider-neutral `legal_generation` span and Groq fallback after both Hugging Face attempts fail; neither provider token is traced.
- Added `real-rag-smoke-test` run logging for provider/model, fallback usage, latency, retrieved-chunk count, pass/fail status, and trace ID.
- Added offline evaluation-run logging for configuration parameters, faithfulness, relevancy, and latency.
- Updated the Streamlit application to use the traced pipeline.
- Ignored local MLflow state (`mlflow.db`, its SQLite journal files, and `mlruns/`) in Git.
- Added the missing `tiktoken` preprocessing dependency and Windows UTF-8 console handling.

### Preserved from the NLP project

```dotenv
PINECONE_API_KEY=
HF_API_TOKEN=
GROQ_API_KEY=
```

The Pinecone index name remains `legal-inheritance-rag`. Embedding, reranking, generation, and judge model settings remain unchanged.

### Verification completed so far

| Date | Check | Result | Evidence/notes |
| --- | --- | --- | --- |
| 2026-10-22 | Python bytecode compilation for `app.py` and `src/` | Passed | All changed modules produced compiled bytecode. |
| 2026-10-22 | Credential-shaped secret scan | Passed | No token-shaped values found in the working copy. |
| 2026-10-22 | Original environment variable names | Passed | The original three RAG variables remain unchanged; optional `GROQ_MODEL` and `GROQ_MAX_ATTEMPTS` settings make the provider fallback configurable. |
| 2026-10-22 | Credential-free MLflow smoke test | Passed | MLflow 3.17.1 recorded run `3b2ce216871d42278f1c2542ff5c71d4` and trace `tr-733a80cdd7ad53af15152a1874324bea`; readback confirmed the CHAIN, RETRIEVER, and LLM spans. |
| 2026-10-22 | MLflow UI and tracking API | Passed | Local UI returned HTTP 200; its API returned experiment `legal-aid-rag` and the finished smoke run with `integration_ok = 1.0`. Team browser inspection confirmed the session and evaluation run are visible. |
| 2026-10-23 | Corpus regeneration and validation | Passed | Loaded 46 of 50 supplied files and generated 3,593 unique recursive chunks. `bm25_corpus.json` matches `chunks_recursive.json`, with non-empty texts of 30-600 characters. Four PDFs were skipped because they contained fewer than 50 extractable characters. |
| 2026-10-23 | Python 3.12 environment | Passed | Created an ignored project-local Python 3.12.13 environment; PyTorch 2.14.1 CPU and Sentence Transformers 6.2.0 import successfully. |
| 2026-10-23 | Pinecone embedding and upsert | Passed | Created the approved 384-dimensional cosine index `legal-inheritance-rag`, uploaded all 3,593 vectors, and passed a three-result semantic sanity query. The pre-existing `rag-fixed-chunks` index was not changed. |
| 2026-10-23 | Hugging Face to Groq fallback unit tests | Passed | Five tests verify provider order, no unnecessary Groq call, bounded transient-error retries, secret exclusion, MLflow provider/model/latency attributes, and compact root-chain output. |
| 2026-10-23 | Full traced query with Groq fallback | Passed | Hybrid retrieval returned five chunks. Hugging Face was unavailable, Groq `qwen/qwen3.8-27b` generated the answer, and MLflow read back the complete three-span trace with `test_passed = 1.0`. |
| 2026-10-23 | Groq evaluation runs in MLflow | Moved to Phase 2 | The team accepted the MLflow foundation and separated comparative RAG evaluation into its own feature branch. |

## Phase 1 manual test plan

Use Python 3.12, matching the original Docker image. Run every command from the root of this branch.

### Test A — install the unchanged environment plus MLflow

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pass condition: installation exits successfully and `python -c "import mlflow; print(mlflow.__version__)"` prints an MLflow 3.x version.

### Test B — credential-free MLflow smoke test

```bash
python -m src.mlflow_smoke_test
```

No Pinecone, Hugging Face, or Groq key is needed. The first run may take longer while SQLite tables are created.

Pass conditions:

- output contains `"status": "ok"`;
- a run ID and trace ID are printed;
- the returned span list contains `smoke_rag_chain`, `smoke_retriever`, and `smoke_llm`;
- `mlflow.db` is created locally and remains untracked by Git.

Record the result here immediately after testing:

| Date/time | Tester | Result | Run ID | Trace ID | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-22 21:59 PKT | Codex (automated) | Passed | `3b2ce216871d42278f1c2542ff5c71d4` | `tr-733a80cdd7ad53af15152a1874324bea` | MLflow 3.17.1 on Python 3.14.3. The first database initialization was interrupted by the verification timeout; the generated test-only database was recreated cleanly, then run and trace readback passed in 13 seconds. |

### Test C — inspect the MLflow UI

Keep the virtual environment active and run:

```powershell
$trackingUri = & ".\.venv\Scripts\python.exe" -c "from src.mlflow_config import default_tracking_uri; print(default_tracking_uri())"
& ".\.venv\Scripts\python.exe" -m mlflow ui --backend-store-uri $trackingUri --port 5000
```

Open `http://127.0.0.1:5000` and select the `legal-aid-rag` experiment.

Pass conditions:

- the Runs tab contains `mlflow-smoke-test` with `integration_ok = 1.0`;
- the Traces tab contains the recorded trace;
- opening the trace shows one CHAIN span with nested RETRIEVER and LLM spans;
- the retriever span contains the example document and no secret values appear anywhere.

Current result:

| Date/time | Tester | Automated checks | Browser inspection | Notes |
| --- | --- | --- | --- | --- |
| 2026-10-22 22:00 PKT | Codex + team | Passed | Passed | UI returned HTTP 200. Server API returned experiment ID `1`, finished run `3b2ce216871d42278f1c2542ff5c71d4`, and `integration_ok = 1.0`; team confirmed visibility in Sessions and evaluation runs. |

### Test D — run a real RAG query

Required before this test:

- a `.env` file containing valid `PINECONE_API_KEY`, `HF_API_TOKEN`, and `GROQ_API_KEY` values (Groq is used only if Hugging Face is unavailable);
- the populated Pinecone index named `legal-inheritance-rag`;
- the same embedding model used for that index (`all-MiniLM-L6-v2`);
- `output/bm25_corpus.json` produced by `python src/preprocess.py`;
- network access to Pinecone, Hugging Face, and Groq.

Start the app:

```bash
streamlit run app.py
```

For a repeatable command-line trace check, run:

```bash
python -m src.mlflow_rag_smoke_test
```

Submit one legal question, then inspect the newest MLflow trace.

Pass conditions:

- the Streamlit response still works as it did in the original project;
- the trace root is `legal_aid_rag_query`;
- `hybrid_legal_retrieval` contains the retrieved chunks and scores;
- `legal_generation` contains the prompt inputs, generated result, provider/model, latency, fallback flag, and sanitized attempt outcomes;
- the `real-rag-smoke-test` run has `test_passed = 1.0` and records the trace ID;
- no API token appears in span inputs, outputs, attributes, or tags.

Record the result:

| Date/time | Tester | Query | Result | Trace ID | Retrieval mode | Model used | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-23 | Codex (automated) | What share does a daughter receive in Islamic inheritance? | Blocked at Groq authentication | `tr-9a7818f72a862414a9e090103e837a2e` was incomplete and not accepted as verification evidence | Hybrid + reranking | None | HF primary returned 402, HF fallback was unsupported, then Groq returned 401. The root output was subsequently reduced to a compact summary to prevent incomplete SQLite traces. Rerun after installing a valid `gsk_...` Groq key. |
| 2026-10-23 07:51 PKT | Codex (automated) | What share does a daughter receive in Islamic inheritance? | Passed | `tr-cf21b2fd316af6ceccdf8484c8d34c13` | Hybrid + reranking | Groq `qwen/qwen3.8-27b` | Run `ac89ff74ca9c47ff926234efa2194d45`; 5 chunks; retrieval 3,205 ms; generation 2,014 ms; Groq 961 ms; all three spans read back; no secrets recorded. |

### Test E — log offline evaluation runs

With the original data artifacts and all keys configured:

```bash
python src/evaluation.py
```

Pass conditions:

- evaluation completes without changing its original scoring formulas;
- each evaluated retrieval/chunking configuration creates an MLflow run;
- each run records configuration parameters;
- successful runs contain faithfulness, relevancy, retrieval latency, and generation latency metrics.

Record the result:

| Date/time | Tester | Result | MLflow run IDs | Best configuration | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-23 | Codex (automated) | Passed | `fad0d83034144ec6b225b75d5317ebaa`, `15fd491ef5d54a16af4794a1067bc100`, `f9787d3e95a646baa707770c2754a17a` | `C-hybrid-top5` | All configurations completed 3/3 queries; direct MLflow readback confirmed distinct parameters, metrics, passed tags, and the best-run tag. |

## Phase 1 definition of done

Phase 1 was accepted after the local tracking store, UI, GenAI traces, provider fallback, real Pinecone retrieval, and complete end-to-end trace were manually verified. Comparative experiment runs were deliberately moved to Phase 2 so each MLOps capability remains isolated by branch.

## Training exemption

The team confirmed that traditional supervised model training is exempt for this RAG project. The system uses pretrained embedding, reranking, and generation models; adding an unrelated classifier or presenting index construction as model training would misrepresent the architecture. The milestone's experiment requirement is satisfied through reproducible RAG configuration evaluation instead.

## Phase 2 — RAG evaluation experiments

### Objective

Run at least three genuine RAG configurations over the same evaluation query set, log their parameters and aggregate metrics to MLflow, select the best configuration using a documented rule, and record reproducible run IDs.

### Evaluated configurations

The final matrix crosses recursive/fixed chunking, semantic/hybrid retrieval,
and top-K values 3/5. Every configuration uses the same three held-out queries;
strategy-specific BM25 artifacts and isolated Pinecone namespaces prevent
fixed chunks from replacing recursive chunks.

### Acceptance criteria

- the retired Groq judge model is replaced by a configurable active model;
- all configurations evaluate the same non-empty query set;
- every configuration logs parameters, faithfulness, relevancy, retrieval latency, and generation latency to MLflow;
- failed queries are counted and exposed instead of silently producing a successful run;
- at least three successful MLflow runs have distinct configuration parameters;
- the best configuration is selected by mean faithfulness, then mean relevancy, then lower total latency as tie-breakers;
- results and run IDs are saved to a clearly named JSON artifact and summarized in `README.md` and this log;
- focused tests pass and no secret value is committed or logged.

### Implemented

- Replaced the retired fixed judge with configurable Groq judge settings: `GROQ_JUDGE_MODEL=openai/gpt-oss-20b` and `GROQ_JUDGE_REASONING_EFFORT=low`.
- Batched claim verification into one judge request per answer while preserving the original supported-claims/total-claims faithfulness formula.
- Added bounded retry handling for rate limits and transient provider errors.
- Ran the complete eight-profile chunking/retrieval/top-K matrix against the same three held-out questions.
- Logged configuration parameters, judge settings, faithfulness, relevancy, retrieval/generation/total latency, success counts, and status tags to MLflow.
- Selected and tagged the best successful run using mean faithfulness, mean relevancy, then mean total latency.
- Routed evaluation queries through the application-level `run_rag_query` boundary so each query, retrieval operation, and generated answer form one nested MLflow trace.
- Added the tracked summary artifact `artifacts/rag_evaluation_summary.json` and kept per-query responses in ignored `output/` data.
- Added focused unit coverage for result aggregation, batched verdict parsing, incomplete judge output, deterministic best-run selection, and the eight-configuration suite.

### Verified final results

| Configuration | Faithfulness | Relevancy | Avg. total latency | Successful | MLflow run ID |
| --- | ---: | ---: | ---: | ---: | --- |
| `recursive-semantic-top3` | 0.8889 | 0.7717 | 17,208.0 ms | 3/3 | `db8bf52010044b47a262af5e8add6dda` |
| `recursive-semantic-top5` | 0.7556 | 0.4809 | 6,851.3 ms | 3/3 | `de06c99b5b9d41d89333459657065f59` |
| `recursive-hybrid-top3` | 0.9667 | **0.7860** | 11,799.0 ms | 3/3 | `07be884a244d41c99f5f8540611a07f1` |
| `recursive-hybrid-top5` | 0.7778 | 0.7723 | 9,374.7 ms | 3/3 | `6d8ff2c40dce4ba48aa2bc77f98d95b3` |
| `fixed-semantic-top3` | **1.0000** | 0.7228 | 10,589.7 ms | 3/3 | `73841e75f54b4a968339b8b63943d5d4` |
| `fixed-semantic-top5` | 0.7963 | 0.7501 | 16,014.0 ms | 3/3 | `cc8313f367b7424eba49bf45328803f9` |
| **`fixed-hybrid-top3`** | **1.0000** | 0.7700 | 17,530.0 ms | 3/3 | `ba6464e78f014a6db1d21af5c39d508f` |
| `fixed-hybrid-top5` | 0.6667 | 0.7402 | 15,932.7 ms | 3/3 | `84ec54970978451fb33ace36af25c730` |

`fixed-hybrid-top3` is the selected best configuration. It ties
`fixed-semantic-top3` at the highest mean faithfulness (1.0000), then wins the
documented second criterion with higher mean relevancy (0.7700 versus 0.7228).
Direct MLflow readback confirmed all eight runs are `FINISHED`, each has
`success_rate=1.0` and `evaluation.status=passed`, and only the selected run in
this suite has `evaluation.best_run=true`.

The 24 newest evaluation traces were also read back from MLflow. Every trace
has state `OK` and contains exactly one `legal_aid_rag_query` root plus nested
`hybrid_legal_retrieval` and `legal_generation` spans. Hugging Face remained
first in the generation order; account credit/provider errors caused the
configured Groq fallback `qwen/qwen3.8-27b` to serve the successful answers.
Occasional judge 429 responses recovered within the bounded retry policy.

Two earlier diagnostic attempts were not accepted as milestone evidence: one exposed rate limits from using the generation model for judging, and another exposed GPT-OSS output-budget and Windows console-encoding issues. The final suite separated the judge model, used low reasoning effort, batched verdicts, and made console output Unicode-safe.

Nested-trace correction was verified live on 2026-10-23 with evaluation run `428483309e0148e08bc4eccb19975a9b` and trace `tr-83738135bb07f35ce078e9be0c71ae88`. MLflow readback showed root span `legal_aid_rag_query` (`b2b62a19aa2bd799`) and both `hybrid_legal_retrieval` and `legal_generation` with that root span ID as their `parent_id`. The query completed successfully using the normal Hugging Face-to-Groq fallback and produced faithfulness `1.0`, relevancy `0.7768`, and `success_rate=1.0`.

Final local verification passed on 2026-10-23 after Step 1: `16 passed in 187.61s` across `test_prepare_bm25.py`, `test_evaluation_experiments.py`, `test_generation_fallback.py`, and `test_pipeline_tracing.py`. The pre-push audit also passed `git diff --check`, confirmed `.env`, `mlflow.db`, and generated `output/` artifacts are ignored, and found no credential-shaped values outside excluded local files.

Step 2 regression verification passed on 2026-10-23: `21 passed in
203.88s` across `test_prepare_bm25.py`, `test_embed_upsert.py`,
`test_evaluation_experiments.py`, `test_generation_fallback.py`, and
`test_pipeline_tracing.py`.

### Eight-configuration expansion

The target matrix combines fixed/recursive chunking, semantic/hybrid retrieval, and top K values 3/5. It is being introduced in isolated checkpoints so the recursive production default remains usable throughout.

| Step | Scope | Status | Verification |
| --- | --- | --- | --- |
| 1 | Prepare and validate fixed BM25 corpus | Complete | 883 chunks, 883 unique IDs, 46 sources, exact alignment passed |
| 2 | Upload fixed embeddings to isolated Pinecone namespace | Complete | 883 fixed vectors; recursive/default remained 3,593; three self-query readbacks and a semantic sanity query passed |
| 3 | Add configurable retrieval profiles | Complete | Validated strategy/mode/top-K profiles route matching BM25 artifacts and Pinecone namespaces; recursive/hybrid/top-5 remains default |
| 4 | Retrieval-only fixed profile smoke tests | Complete | All four semantic/hybrid top-3/top-5 profiles passed against namespace `fixed`; nested MLflow traces verified |
| 5 | Expand evaluation matrix to eight configurations | Complete | Exact recursive/fixed × semantic/hybrid × top-3/top-5 Cartesian matrix validated; real runs deferred to Step 6 |
| 6 | Run and compare the final eight-run suite | Complete | 8/8 runs `FINISHED`; 24/24 queries and nested traces passed; best run tagged and artifact updated |

Step 1 command:

```bash
python -m src.prepare_bm25 --strategy fixed
```

Step 1 real-data validation produced `output/bm25_corpus_fixed.json` with SHA-256 `e878377705440dc9e599ecd3e3c2b55cd69299a072d23b7f1d806fd9736a7f15`. Fixed chunk texts range from 52 to 5,314 characters with an average of 1,783.25. The existing recursive `output/bm25_corpus.json` was not rewritten; it still aligns with 3,593 recursive chunks and has SHA-256 `51c8b6cd081f954f3e5d94d4676e98ffccb46556c6bc5fd88a736207447dcdc2`.

There are 40 IDs shared between the fixed and recursive chunk files because the original stable-ID function does not include the strategy name. This is safe for separate local BM25 artifacts, but it makes the isolated `fixed` Pinecone namespace mandatory in Step 2; uploading fixed vectors to the default namespace would risk overwriting recursive vectors.

Step 2 command:

```bash
python -m src.embed_upsert --strategy fixed
```

Step 2 preserved the original recursive defaults while adding an explicit
fixed profile backed by `output/embeddings_fixed.json` and Pinecone namespace
`fixed`. The implementation validates the index dimension and metric, chunk
and embedding alignment, unique Pinecone IDs, full-text metadata size, exact
namespace vector count, three representative self-query readbacks, and a
namespace-scoped semantic sanity query. The verified idempotent rerun reported
883 fixed vectors, 3,593 recursive/default vectors before and after the upload,
and 4,476 vectors total. Its three sanity-query results all carried
`strategy=fixed`.

Step 3 added immutable `RetrievalProfile` configurations spanning
`recursive`/`fixed` chunking, `semantic`/`hybrid` retrieval, and positive top-K
values. `build_retriever()` now selects the matching BM25 artifact and returns
the matching Pinecone namespace, while its no-argument behavior remains the
original recursive corpus and default namespace. Pure semantic mode preserves
Pinecone's dense order without BM25, RRF, or cross-encoder reranking; hybrid
mode preserves the original BM25 + dense + RRF + reranking path. Both the root
CHAIN span and child RETRIEVER span record the resolved strategy, mode,
namespace, and top-K. Profile/component strategy or namespace mismatches fail
before a query reaches an external service.

Step 3 regression verification passed on 2026-10-23: `33 passed in 70.89s`
across data preparation, embedding/upsert, retrieval profiles, evaluation,
generation fallback, and pipeline tracing tests. Live fixed-profile retrieval
is intentionally the separate Step 4 checkpoint.

Step 4 command:

```bash
python -m src.retrieval_smoke_test
```

The retrieval-only command reuses one fixed BM25/model bundle and does not call
Hugging Face generation, Groq generation, or the Groq judge. On 2026-10-24 all
four profiles returned the exact requested number of non-empty, unique, fixed
chunks from Pinecone namespace `fixed`:

| Profile | BM25 | Semantic | Fused | Returned | Latency | Trace ID |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `fixed-semantic-top3` | 0 | 20 | 20 | 3 | 8,917.83 ms | `tr-16bab80442459f2254969c872c1a390c` |
| `fixed-semantic-top5` | 0 | 20 | 20 | 5 | 590.65 ms | `tr-884b12d2036c6a7cf570125ef276dd6d` |
| `fixed-hybrid-top3` | 20 | 20 | 38 | 3 | 11,540.70 ms | `tr-94b7ab775813e06c3e57a2a011297335` |
| `fixed-hybrid-top5` | 20 | 20 | 38 | 5 | 10,989.34 ms | `tr-477c416ffed26dff909042d956c8735a` |

Immediate MLflow readback verified that every trace contains exactly one
`fixed_retrieval_smoke` CHAIN root and one `hybrid_legal_retrieval` RETRIEVER
child whose `parent_id` equals the root `span_id`. The ignored detailed report
is `output/fixed_retrieval_smoke.json`.

Step 4 regression verification passed on 2026-10-24: `40 passed in 46.82s`
across all seven focused test modules. The smoke test made no Pinecone writes
and no generation or judge requests.

Step 5 replaced the recursive-only evaluation orchestration with eight
validated `RetrievalProfile` configurations:

| Chunking | Retrieval | Top K values | Namespace |
| --- | --- | --- | --- |
| Recursive | Semantic | 3, 5 | `__default__` |
| Recursive | Hybrid + reranking | 3, 5 | `__default__` |
| Fixed | Semantic | 3, 5 | `fixed` |
| Fixed | Hybrid + reranking | 3, 5 | `fixed` |

Every evaluation run now derives strategy, mode, top-K, and namespace from its
profile and sends that same profile through the traced application boundary.
MLflow parameters and tags therefore reflect the real retriever rather than a
hardcoded recursive label. Suite startup requires components for both
strategies. The two BM25 indexes are built independently, while one Pinecone
client, embedding model, and cross-encoder are shared to avoid duplicate model
memory. The existing three-run summary artifact was intentionally not changed
in this implementation-only checkpoint; Step 6 replaced it after all eight
real runs completed successfully.

Step 5 regression verification passed on 2026-10-24: `43 passed in 123.27s`
across all seven focused test modules. No Pinecone, Hugging Face, or Groq call
was made by the Step 5 matrix tests.

Step 6 command:

```bash
python -m src.evaluation
```

Step 6 completed all eight runs and all 24 query evaluations on 2026-10-24.
Direct tracking-store readback verified the exact chunking strategy, retrieval
mode, top-K, and Pinecone namespace on every run, along with the aggregate
metrics and pass tags. The selected run is
`ba6464e78f014a6db1d21af5c39d508f` (`fixed-hybrid-top3`). The tracked compact
result is `artifacts/rag_evaluation_summary.json`; full per-query answers remain
in the ignored `output/rag_evaluation_details.json` file.

Step 6 regression verification passed on 2026-10-24: `43 passed in 133.54s`
across all seven focused test modules after the live evaluation artifact and
documentation were updated.

## Phase 3 — MLflow GenAI datasets and judges

### Objective

Use MLflow's native GenAI evaluation features without changing the existing
RAG retrieval or generation architecture: manage a reusable held-out dataset,
register versioned judges, and run reproducible MLflow GenAI evaluations.

### Step plan

| Step | Scope | Status | Verification |
| --- | --- | --- | --- |
| 1 | Register and label the shared held-out dataset | Complete | 3 human-reviewed records and expectations visible in UI; digest `1f570963`; idempotent label rerun updated 0 |
| 2 | Define and register versioned judges | Complete | 6 judges registered; latest pinned versions reuse cleanly; 0 setup inference calls |
| 3 | Run GenAI evaluation on `fixed-hybrid-top3` | Complete | Run `9fcc1af514b046078f04bc2202c5f60a` confirmed in UI; 3 nested traces; 6 successful assessments per row |
| 4 | Compare additional profiles with the same dataset and judges | Complete (8/8 valid) | All eight profiles are complete; `fixed-hybrid-top5` is the uniquely tagged best run |

### Step 1 — managed dataset

`src/genai_dataset.py` creates or reuses
`legal-aid-rag-held-out-v1`, links it to the active `legal-aid-rag`
experiment, applies provenance and label-status tags, merges only missing
records, and reads the dataset back for verification. The shared query list
was moved to lightweight `src/evaluation_queries.py` so dataset setup does not
initialize embedding, reranking, retrieval, or generation dependencies.

Run:

```bash
python -m src.genai_dataset --creator "Karam Hussain"
```

Verified locally on 2026-10-31:

- dataset ID: `d-fee7335515b9445dbb7feddfd68de247`;
- initial unlabeled dataset digest: `b4225320`;
- linked experiment ID: `1` (`legal-aid-rag`);
- three expected queries persisted;
- `label_status=unlabeled`;
- `created_by=Karam Hussain` and `last_updated_by=Karam Hussain`;
- immediate rerun returned `created=false` and `records_added=0`;
- final full regression suite: `48 passed` in 152.657 seconds after adding the
  supported creator-recreation path.

The dataset contains no model-generated ground truth. On 2026-10-31, Karam
Hussain approved version-1 structured expected facts for all three queries.
`src/genai_expectations.py` upserted those facts into the existing input-keyed
records, changing the digest to `1f570963` without changing the dataset ID or
record count. The dataset is now tagged `label_status=human-reviewed`,
`expectation_version=1`, and `reviewed_by=Karam Hussain`. The first invocation
updated three records; an immediate rerun updated zero.

The first UI launch was diagnosed as pointing at a different `mlflow.db` in
the parent workspace: that database contained 32 experiment traces and no
evaluation dataset. The correct project database contained all 89 traces and
the managed dataset. Manual inspection confirmed the 89 traces and all three
query inputs. The UI instructions now resolve the absolute project tracking
URI. Because MLflow 3.17.2 has no supported in-place creator update, the still
unscored dataset was recreated through MLflow's API with the reserved
`mlflow.user` tag; no trace was deleted or changed.

### Step 2 — versioned judges

`src/genai_judges.py` defines and registers these scorers under experiment ID
`1` (`legal-aid-rag`):

- `relevance_to_query`;
- `retrieval_relevance`;
- `retrieval_groundedness`;
- `correctness`;
- `retrieval_sufficiency`;
- `legal_answer_quality`, a custom judge for evidence use, uncertainty, and
  responsible legal-information framing.

All six use `groq:/openai/gpt-oss-20b`. Correctness and retrieval sufficiency
were registered only after the dataset received approved expectations. The
setup is registration-only and recorded `inference_calls=0`, so it did not query Groq,
Pinecone, Hugging Face, or the RAG application.

Run:

```bash
python -m src.genai_judges
```

Initial SDK verification on 2026-10-31 registered all six judges as version 1.
Step 3 uses `relevance_to_query:v1`, `retrieval_groundedness:v2`,
`retrieval_relevance:v2`, `correctness:v1`, `retrieval_sufficiency:v1`, and
`legal_answer_quality:v2`. The custom v2 removes trace-tool calling because the
direct Groq provider rejects JSON mode combined with tool calling. The two
retrieval scorers were reserialized after live use. Registration now explicitly
selects the highest version regardless of API list order; an immediate invocation
returned `action=reused` for all six, proving
that an unchanged setup does not create redundant versions. Manual inspection
confirmed that all six version-1 judges are visible in **GenAI -> Judges** and
that the approved `expected_facts` are visible on all three dataset records.
Their **Evaluating traces** setting remains intentionally `OFF`: Step 3 will
invoke the registered versions explicitly against the managed dataset instead
of enabling continuous background sampling and unplanned provider usage. The
earlier four-judge regression suite passed with `53 passed` in 116.38 seconds;
the final expectation-and-six-judge suite passed with `58 passed` in 194.99
seconds.

### Step 3 — controlled managed-dataset evaluation

`src/genai_evaluation.py` validates the exact three human-reviewed records,
loads the latest six registered judges, builds only the fixed retriever, and
runs the existing RAG boundary with profile `fixed-hybrid-top3`. It forces one
evaluation worker to protect Groq quota and skips MLflow's extra prevalidation
prediction so the final run contains exactly three evaluated generation calls.

The first live attempt, run `390e08774cc0409788bcf79fb1728e45`, diagnosed two
adapter issues and is tagged `evaluation.status=diagnostic-invalid`: the
project's RETRIEVER span originally serialized a wrapper object instead of the
document list MLflow judges require, and the trace-aware custom judge triggered
a Groq JSON-mode/tool-calling conflict. The fixes changed trace serialization
only—not retrieval ranking or returned application data—and made the custom
judge expectation-free without trace tools.

Corrected run `9fcc1af514b046078f04bc2202c5f60a` completed on 2026-11-01:

- linked managed dataset: `legal-aid-rag-held-out-v1`, digest `1f570963`;
- exactly three traces, each nested as `legal_aid_rag_query` →
  `hybrid_legal_retrieval` → `legal_generation`;
- all six scorer invocations succeeded for all three rows;
- query relevance `1.0000`;
- retrieval groundedness `1.0000`;
- retrieval relevance and precision `0.7778`;
- retrieval sufficiency `0.3333`;
- strict all-facts correctness `0.0000`;
- custom legal-answer quality passed `1/3` rows;
- Hugging Face remained primary but was unavailable, so Groq
  `qwen/qwen3.8-27b` handled all three generations (`fallback_rate=1.0`);
- mean retrieval, generation, and total RAG latency were `11,307.0 ms`,
  `2,204.7 ms`, and `13,511.7 ms` respectively;
- compact tracked evidence is stored in
  `artifacts/genai_evaluation_fixed_hybrid_top3.json`;
- final regression suite: `65 passed` in 90.85 seconds.

The low sufficiency and correctness values are valid findings, not scorer
errors: the retrieved chunks fully grounded the generated claims, but did not
cover every approved fact for two queries, and no response covered every strict
expected fact. Manual UI inspection confirmed the corrected evaluation run,
all three rows, six assessment columns, and nested retrieval/generation spans.

### Step 4 — resumable eight-profile GenAI comparison

`src/genai_comparison.py` applies the same managed dataset, approved
expectations, generation temperature, and pinned six-judge lineage to the exact
recursive/fixed × semantic/hybrid × top-3/top-5 matrix. It reuses only reports
whose dataset ID, digest, expectation version, judge versions, row count, and
completion status all match. Each successful profile is checkpointed before the
next begins, and the final best profile will be selected by the documented
quality-first rule with latency used only as a later tie-breaker.

The runner keeps both dataset rows and judge calls sequential
(`MLFLOW_GENAI_EVAL_MAX_WORKERS=1` and
`MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS=1`) and uses bounded per-judge retries for
transient structured-output failures. The existing `fixed-hybrid-top3` Step 3
result was reused rather than rerun.

Verified completed profiles through 2026-11-03:

| Profile | Run ID | Correctness | Retrieval sufficiency | Legal quality pass rate | Retrieval groundedness | Retrieval relevance | Query relevance | Mean total latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed-hybrid-top3` | `9fcc1af514b046078f04bc2202c5f60a` | 0.0000 | 0.3333 | 0.3333 | 1.0000 | 0.7778 | 1.0000 | 13,511.7 ms |
| `recursive-hybrid-top3` | `98e7838d135f43fcb830c1d824e9581a` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.8889 | 1.0000 | 9,424.3 ms |
| `recursive-semantic-top5` | `a130d951e9ce446d9238848c24dd8986` | 0.0000 | 0.0000 | 0.3333 | 1.0000 | 0.4667 | 0.3333 | 4,557.7 ms |
| `recursive-semantic-top3` | `5e585a134b844a0a8e50ae4693cee229` | 0.0000 | 0.0000 | 0.3333 | 0.3333 | 0.4444 | 0.3333 | 3,933.7 ms |
| `recursive-hybrid-top5` | `63fe861d46e740a0893c7d13041152ba` | 0.0000 | 0.0000 | 0.3333 | 1.0000 | 0.8667 | 1.0000 | 7,532.3 ms |
| `fixed-semantic-top3` | `91c3ff85269a47ad8a1131b5e28815dc` | 0.0000 | 0.3333 | 0.6667 | 1.0000 | 0.4444 | 0.6667 | 4,589.7 ms |
| `fixed-semantic-top5` | `68e216b95fc3455bad64fd8f516c2902` | 0.0000 | 0.3333 | 0.6667 | 1.0000 | 0.4667 | 0.6667 | 3,807.7 ms |
| `fixed-hybrid-top5` | `e6034e3cda2f4582ab5fac02132b4687` | 0.3333 | 0.3333 | 1.0000 | 0.6667 | 0.6667 | 1.0000 | 29,517.0 ms |

The comparison is complete. Earlier partial attempts remain excluded from
reuse and ranking. The original live continuation was stopped after Groq
repeatedly rate-limited the registered `openai/gpt-oss-20b` judges and a scorer
exceeded the bounded timeout; one row took more than 90 minutes and could no
longer produce a complete profile. This was an external judge-quota blocker,
not a RAG retrieval or generation failure.

A continuation check later on 2026-11-01 showed that a minimal
`relevance_to_query:v1` request could succeed in 11.11 seconds, but that was not
sufficient evidence of full quota recovery: the first full-context assessment
in `recursive-hybrid-top5` still exceeded the 300-second scorer timeout. The
attempt was stopped immediately and MLflow run
`1269f225f5c04632a40aa60442a10edc` was marked `KILLED` with
`evaluation.status=interrupted-provider-quota`. No completed profile was
repeated or replaced. Wait for the Groq judge quota window to reset before the
next resume.

On 2026-11-02, a fresh minimal judge preflight succeeded in 19.53 seconds. The
resumable runner then completed and checkpointed `recursive-hybrid-top5`,
`fixed-semantic-top3`, and `fixed-semantic-top5`, increasing valid coverage
from 4/8 to 7/8 without repeating earlier completed profiles. The final
`fixed-hybrid-top5` attempt generated all three answers but finished partial
after six scorer invocations timed out at 300 seconds. Its run
`090c1764909a410a95bb9ef52e37bb18` is marked `KILLED`, tagged
`evaluation.status=interrupted-provider-quota` and
`evaluation.valid_for_comparison=false`, and excluded from reuse and ranking.
At that point, only that profile needed another quota-window retry.

On 2026-11-03, a fresh preflight succeeded in 21.86 seconds and the runner
retried only `fixed-hybrid-top5`. All three rows and all six judges completed
with zero scorer failures in run `e6034e3cda2f4582ab5fac02132b4687`.
The comparison now reports `status=completed`, eight completed profiles, no
remaining profiles, and `error=null`. MLflow SDK readback confirmed all eight
runs are `FINISHED`, the winner alone has `evaluation.best_run=true`, and the
other seven have `evaluation.best_run=false`. `fixed-hybrid-top5` wins under
the documented quality-first rule because it is the only profile with nonzero
correctness (0.3333); latency is considered only after the quality metrics.
A final unfiltered runner invocation reused all eight compatible artifacts
without inference and rewrote `requested_profiles` as the complete matrix.

Implementation verification passes all 11 focused comparison and
controlled-evaluation tests. The final full local regression after the
quota-safe resume and cleanup changes passed all 70 tests in 223.25 seconds.
After the eighth profile completed, the focused comparison/evaluation suite
passed again: 11 tests in 51.54 seconds with only the existing SQLAlchemy
deprecation warning.

## Phase 4 — FastAPI prediction service

### Objective

Serve the unchanged MLflow-traced RAG pipeline through a validated HTTP API
without coupling deployment to the deferred GenAI comparison winner.

### Implementation checkpoint

- Added `src/app.py` with `/predict`, `/health`, generated `/docs`, and an
  independently testable application factory.
- Kept the current serving default at recursive + hybrid + top-5; requests may
  select semantic/hybrid mode, top-K from 3 to 10, and temperature from 0 to 1.
- Loads and validates the recursive retriever once during application startup,
  then reuses the same components for every request.
- Calls the existing `src.pipeline.run_rag_query` boundary, preserving nested
  MLflow query/retrieval/generation tracing and the Hugging Face-to-Groq
  generation fallback.
- Returns compact provider, model, fallback, source, chunk-count, and latency
  metadata without returning prompts, full retrieved documents, or secrets.
- Maps missing context to HTTP 404, provider generation failure to HTTP 502,
  unavailable startup state to HTTP 503, unexpected pipeline failures to HTTP
  500, and Pydantic validation failures to HTTP 422.
- Added explicit FastAPI, Uvicorn, and HTTPX dependencies.
- Focused credential-free verification initially passed `4` tests in 3.53
  seconds; component completeness coverage was then added before the final
  regression run.
- Real startup-loader verification passed without a retrieval or generation
  call: 3,593 recursive BM25 documents, recursive/default Pinecone namespace,
  embedding model, reranker, and Pinecone index client all initialized.
- Full local regression after the API implementation passed all 75 tests in
  172.52 seconds with only the existing SQLAlchemy deprecation warning.
- Real Uvicorn startup passed on `127.0.0.1:8000`: `/health` returned HTTP 200
  with `retriever_loaded=true` and `chunking_strategy=recursive`, `/docs`
  returned HTTP 200, and `/openapi.json` identified version `1.0.0` with both
  `/health` and `/predict` paths.
- Real `/predict` verification passed for "What share does a daughter receive
  in Islamic inheritance?": HTTP 200, recursive hybrid top-5, five chunks from
  two unique sources, Groq fallback `qwen/qwen3.8-27b`, retrieval 11,333 ms,
  generation 2,789 ms, and total RAG time 14,122 ms.
- MLflow readback confirmed trace `tr-27666fc0cfcd18e5d7811d72fda40089`
  is `OK` with one `legal_aid_rag_query` root and nested
  `hybrid_legal_retrieval` and `legal_generation` spans.

The live service and prediction endpoint are verified. On 2026-11-02, the user
also manually confirmed both `/health` and `/predict` through the generated
`/docs` interface. Predictions require the existing Pinecone and generation
credentials but do not invoke the deferred MLflow judge suite.

## Phase 5 — milestone documentation and rubric audit

### Objective

Make the repository entry point independently reproducible and map the current
RAG implementation to the Milestone 1 deliverables without misrepresenting
index construction as supervised training.

### Implementation checkpoint

- Confirmed the required unit-test categories already exist across
  `tests/test_prepare_bm25.py` and `tests/test_api.py`: preparation and invalid
  data, cached retriever loading, request validation, prediction success, and
  provider-failure handling.
- Added a one-line project pitch and a compact Mermaid architecture diagram
  covering legal data, validated preparation/indexing, RAG evaluation, MLflow,
  and FastAPI serving.
- Added copyable Windows and Linux/macOS quick-start instructions using a
  virtual environment, `requirements.txt`, `.env.example`, `pytest`, and
  Uvicorn.
- Added a concise dataset/task statement recording the five-folder corpus,
  46/50 accepted PDFs, 3,593 recursive chunks, 883 fixed chunks, held-out RAG
  task, and repository data-exclusion policy.
- Added `.env.example` with variable names and non-secret defaults only.
- Added `CONTRIBUTION.md` with the three student names, ERP IDs, milestone task
  division, and detailed member responsibilities.

## Phase 6 - local Gradio Miras interface

### Objective

Replace the deprecated Hugging Face Streamlit Space path with a locally
runnable Gradio interface in the same visual style, without changing the RAG
retrieval, generation fallback, or MLflow tracing architecture.

### Implementation and verification

- Added `gradio_app.py` with a responsive parchment, gold, and dark Miras UI,
  example questions, fixed/recursive chunking, hybrid/semantic retrieval,
  top-K and temperature controls, answer metadata, optional retrieved context,
  and safe errors.
- Reused `src.app.load_retriever_components`, `build_profile`, and
  `run_rag_query`; the recursive default, Pinecone/BM25 retrieval, Hugging
  Face-to-Groq fallback, and nested MLflow spans remain unchanged.
- Added Gradio 6 to `requirements.txt` and kept generated Matplotlib cache data
  under the ignored project `.cache` directory for reliable Windows startup.
- Added six credential-free UI-adapter tests. Together with the five API
  tests, the focused suite passed all 11 tests after chunking selection was
  added.
- The initial complete repository regression passed all 80 tests. After adding
  chunking selection, the complete suite passed all 81 tests in 295.53 seconds;
  the only warning is the pre-existing MLflow SQLAlchemy deprecation warning.
- Dependency validation passed for all 164 installed packages.
- Verified `/` and `/config` over HTTP and confirmed the prediction API is
  exposed as `/predict`.
- Completed a real recursive/hybrid/top-5 query. It returned five chunks from
  two sources; Hugging Face exhausted/unavailable models cleanly fell back to
  Groq `qwen/qwen3.8-27b`; retrieval took 10,080 ms, generation 2,825 ms, and
  total RAG time 12,905 ms.
- MLflow readback confirmed trace `tr-c5dadc5827581e50333df3c143a806c7`
  is `OK`, with `hybrid_legal_retrieval` and `legal_generation` nested under
  the `legal_aid_rag_query` root.
- Browser visual acceptance remains for the user at
  `http://127.0.0.1:7860`; the local service is ready for that manual check.
- Extended the interface with fixed/recursive chunking selection. Live
  fixed/hybrid/top-5 verification loaded the 883-document fixed BM25 artifact,
  queried Pinecone namespace `fixed`, returned five chunks from five sources,
  and generated through the expected Groq fallback in 57,629 ms total.

## Phase 7 - prompt, model, and temperature inference ablation

### Objective

Meet the updated RAG requirement with controlled inference evaluations rather
than training cycles. Reuse the completed eight-profile retrieval parameter
comparison and isolate prompt, generation-model, and temperature effects on
the same held-out MLflow dataset.

### Implementation checkpoint

- Froze retrieval to the GenAI winner: fixed chunks, hybrid retrieval, top-5,
  and Pinecone namespace `fixed`.
- Added versioned `strict_grounded_v1` and `structured_legal_v2` prompts.
- Added backward-compatible direct-Groq evaluation mode while preserving the
  production Hugging Face-to-Groq `auto` fallback default.
- Added a resumable four-run matrix sharing one baseline and changing exactly
  one generation variable per comparison.
- Added MLflow params/tags for requested and actual provider/model, prompt
  variant, temperature, dataset/judge lineage, quality metrics, and latency.
- Groq model discovery on 2026-11-05 confirmed `qwen/qwen3.8-27b` and
  `openai/gpt-oss-120b`; both passed real direct-inference preflights in 1,738
  ms and 2,436 ms total respectively.
- The initial focused regression passed all 21 generation, tracing, managed
  evaluation, and ablation tests; the complete repository suite passed all 91
  tests with only the pre-existing SQLAlchemy deprecation warning.
- Baseline run `f8eb6177b0a44c2e91cbd27926ca3efd` completed all three rows and
  six-judge lineage at correctness 0.000, legal quality 0.667, retrieval
  relevance 0.667, and 28,756 ms average total latency.
- Structured-prompt run `1528a96621d54a93819f01eab6b600a0` completed and
  improved correctness to 0.333, legal quality to 1.000, retrieval relevance
  to 0.733, and average total latency to 21,272 ms.
- Model-B run `6db207d1847d4d96af0f42905fcf1bf1` completed. GPT-OSS-120B was
  8,611 ms faster than baseline but reduced legal quality to 0.333 and
  retrieval relevance to 0.600, so Qwen remains the stronger quality choice.
- SDK readback confirmed representative baseline, prompt, and model traces are
  `OK`, each with `hybrid_legal_retrieval` and `legal_generation` nested under
  `legal_aid_rag_query`: `tr-88649357783352dd0f7961f097355771`,
  `tr-595be916ac6bf44619393466c55f8d8d`, and
  `tr-f1a17e1112afdbccf3fb4d94c1e0d466`.
- A replacement Groq key was selected locally without committing credentials.
  The first retry confirmed that Groq generation worked but encountered a
  transient Pinecone DNS failure on row three; strict validation excluded that
  partial run. Pinecone resolution recovered before the final retry.
- High-temperature run `caf08bcd4194455f9efc030ffc2a9634` completed all three
  rows and six judges. Versus the 0.1 baseline, temperature 0.7 changed
  retrieval sufficiency by -0.333, groundedness by +0.333, retrieval relevance
  by -0.067, and average total latency by -12,058 ms; correctness and legal
  quality were unchanged.
- The comparison is complete at 4/4 valid. `prompt-structured` remains the
  selected run because it has the best correctness (0.333), legal quality
  (1.000), and retrieval relevance (0.733) in this controlled dataset.

## Phase 8 - post-M1 correctness and latency fixes

### Objective

Address instructor feedback about 20-30 second inference and misleading
all-or-nothing correctness results while preserving the RAG architecture.

### Final validation

- Replaced Hugging Face generation with direct Groq because the original Hub
  model was removed; retrieval and nested MLflow tracing remain unchanged.
- Added stage-level retrieval timing and fractional expected-fact coverage.
  Strict binary correctness remains available as a diagnostic, while profile
  ranking uses fact coverage so one omitted fact does not collapse a partially
  correct answer to zero.
- Completed all eight core-v2 profile evaluations. `fixed-hybrid-top5` remains
  the quality champion at 0.7222 fact coverage; `fixed-semantic-top3` is the
  speed challenger at 0.5833 fact coverage.
- Verified the Gradio interface end to end on 2026-11-13 after restarting it
  with Pinecone/Groq network access. A fixed/hybrid/top-5 request returned five
  chunks from three sources through Groq `qwen/qwen3.8-27b` with zero retries
  in 10,699 ms total: 9,171 ms retrieval and 1,528 ms generation.
- Ran matching 75-request warm retrieval benchmarks across 15 questions and
  five repeats. Champion run `ae428d443faa4487a2c01659b2923281`
  recorded 6,975 ms mean, 6,754 ms median, and 12,211 ms p95. Challenger run
  `c672c620f2b242a9a7a3baf6d4614b1a` recorded 528 ms mean, 394 ms median,
  and 1,341 ms p95.
- The challenger is 92.4% faster by mean and 94.2% faster by median retrieval,
  while giving up 0.1389 absolute fact coverage. The deployment choice is
  therefore explicit: hybrid top-5 for maximum coverage, semantic top-3 for
  interactive latency.
- Rechecked all four inference-ablation settings with two repeats across the
  same three held-out questions. Retrieval context was shared within each
  question/repeat, call order was rotated, judges were disabled, and the
  20-second quota cooldown was excluded from timing. All 24 generation calls
  succeeded with zero retries or failures in MLflow run
  `0c4c66a00d1c41b2add2d4ba6791d7b9`.
- Generation medians were 759 ms (baseline), 1,377 ms (structured prompt),
  1,590 ms (GPT-OSS-120B), and 937 ms (temperature 0.7); total medians were
  5,330, 5,746, 6,091, and 5,370 ms respectively. The earlier 52.7-106.6
  second averages were not reproduced and are attributed to transient
  Pinecone/Groq stalls and retries rather than stable configuration cost.

## Change history

| Date | Branch | Change |
| --- | --- | --- |
| 2026-10-22 | `1-mlflow-integration` | Created the phase log and recorded implementation plus pending manual tests. |
| 2026-10-22 | `1-mlflow-integration` | Passed the credential-free smoke test and recorded its run ID, trace ID, and verified spans. |
| 2026-10-22 | `1-mlflow-integration` | Verified the MLflow UI server and tracking API; browser inspection remains pending. |
| 2026-10-22 | `1-mlflow-integration` | Team confirmed the session and evaluation run in the MLflow UI; Test C passed. |
| 2026-10-22 | `1-mlflow-integration` | Test D prerequisite audit completed; real-query verification is blocked by missing local keys and corpus artifacts. |
| 2026-10-23 | `1-mlflow-integration` | Regenerated and validated 3,593 BM25 entries; added missing tokenizer dependency and Windows UTF-8 handling. Test D is now blocked only by the absent configured Pinecone index. |
| 2026-10-23 | `1-mlflow-integration` | Created and validated `legal-inheritance-rag` with 3,593 vectors. Real-query generation is blocked by Hugging Face credit/model availability. |
| 2026-10-23 | `1-mlflow-integration` | Added tested Hugging Face-to-Groq fallback and MLflow provider/test metadata. Live Test D reached Groq but is blocked by an invalid Groq credential. |
| 2026-10-23 | `1-mlflow-integration` | Updated the configurable Groq fallback to active model `qwen/qwen3.8-27b`, added bounded transient-error retries, made local SQLite trace writes synchronous, and passed full Test D with verified run and trace IDs. |
| 2026-10-23 | `2-rag-evaluation-experiments` | Team accepted the supervised-training exemption, closed the MLflow-foundation phase, and created the RAG experiment branch with documented acceptance criteria. |
| 2026-10-23 | `2-rag-evaluation-experiments` | Added three comparable RAG experiment configurations, explicit failure accounting, richer MLflow metrics/tags, deterministic best-run selection, and focused unit tests. |
| 2026-10-23 | `2-rag-evaluation-experiments` | Diagnosed judge rate limits, hidden-reasoning truncation, and Windows console encoding; separated the GPT-OSS judge from Qwen generation, batched claim verdicts, and made output safe. |
| 2026-10-23 | `2-rag-evaluation-experiments` | Completed three 3/3-query MLflow runs, selected `C-hybrid-top5`, saved the compact result artifact, and verified all parameters, metrics, status tags, and the best-run tag through MLflow readback. |
| 2026-10-23 | `2-rag-evaluation-experiments` | Passed the final 11-test focused suite and pre-push secret/ignore/diff audit; partial configurations are explicitly excluded from best-run selection. |
| 2026-10-23 | `2-rag-evaluation-experiments` | Corrected evaluation tracing to reuse the application CHAIN boundary; live MLflow parent-ID readback confirmed query, retrieval, and generation are nested in trace `tr-83738135bb07f35ce078e9be0c71ae88`. |
| 2026-10-23 | `2-rag-evaluation-experiments` | Began the eight-configuration expansion. Step 1 added validated strategy-specific BM25 preparation, generated the 883-entry fixed corpus, preserved the recursive default, and documented the 40 cross-strategy ID collisions requiring namespace isolation. |
| 2026-10-23 | `2-rag-evaluation-experiments` | Completed Step 2: cached and uploaded 883 fixed embeddings to isolated Pinecone namespace `fixed`, verified representative vectors and a semantic query, and confirmed the recursive/default namespace remained at 3,593 vectors. |
| 2026-10-23 | `2-rag-evaluation-experiments` | Completed Step 3 implementation: added validated retrieval profiles, strategy-specific BM25/Pinecone routing, true semantic versus hybrid execution paths, mismatch guards, MLflow profile metadata, and focused unit coverage while preserving recursive/hybrid/top-5 defaults. |
| 2026-10-24 | `2-rag-evaluation-experiments` | Completed Step 4: all four fixed semantic/hybrid top-3/top-5 retrieval profiles passed live namespace, strategy, count, content, and nested MLflow trace validation without generation or judge calls. |
| 2026-10-24 | `2-rag-evaluation-experiments` | Completed Step 5 implementation: expanded evaluation to the exact eight-profile Cartesian matrix, removed hardcoded recursive run metadata, routed profiles through nested RAG traces, shared dense resources across strategy-specific BM25 retrievers, and kept real execution isolated to Step 6. |
| 2026-10-24 | `2-rag-evaluation-experiments` | Completed Step 6: executed and verified all eight MLflow runs and 24 nested RAG traces, selected `fixed-hybrid-top3`, updated the tracked evaluation artifact, and recorded the final comparison. |
| 2026-10-31 | `3-mlflow-genai-datasets-judges` | Began Phase 3 Step 1: created and verified the reusable three-record MLflow GenAI evaluation dataset, added idempotent setup and tests, and preserved the existing RAG pipeline. |
| 2026-10-31 | `3-mlflow-genai-datasets-judges` | Completed Step 1 UI verification, diagnosed an incorrect parent tracking database, preserved all 89 project traces, and recreated the unscored dataset with `Karam Hussain` as its MLflow creator. |
| 2026-10-31 | `3-mlflow-genai-datasets-judges` | Implemented Step 2 registration for four expectation-free, Groq-backed MLflow judges; SDK verification registered version 1 and an idempotent rerun reused every version without inference. |
| 2026-10-31 | `3-mlflow-genai-datasets-judges` | Completed Step 2 UI verification: all four version-1 judges are visible; automatic trace evaluation remains intentionally off until controlled dataset evaluation in Step 3. |
| 2026-10-31 | `3-mlflow-genai-datasets-judges` | Added Karam Hussain's approved version-1 expected facts to the same three dataset records, marked them human-reviewed, and registered correctness plus retrieval-sufficiency judges without inference. |
| 2026-10-31 | `3-mlflow-genai-datasets-judges` | Completed the label-extension UI verification: expectations are visible on all three records and all six version-1 judges are visible before Step 3. |
| 2026-11-01 | `3-mlflow-genai-datasets-judges` | Implemented Step 3 managed-dataset evaluation, used a diagnostic run to correct MLflow retriever-output and Groq custom-judge adapters, then completed and SDK-verified the three-trace six-judge run `9fcc1af514b046078f04bc2202c5f60a`. |
| 2026-11-01 | `3-mlflow-genai-datasets-judges` | Completed Step 3 UI verification: the successful evaluation run, all three rows, all six assessment columns, and nested RAG spans are visible in MLflow. |
| 2026-11-01 | `3-mlflow-genai-datasets-judges` | Began Step 4 with a resumable eight-profile runner, strict lineage checks, deterministic ranking, sequential judge execution, and bounded scorer retries; 4/8 profiles are valid and checkpointed, while the remaining four are paused after sustained Groq judge rate limiting. |
| 2026-11-01 | `3-mlflow-genai-datasets-judges` | Retested Step 4 continuation after an 11.11-second minimal judge preflight; the first full-context assessment still timed out at 300 seconds, so the attempt was marked `KILLED` and the recommendation remains to wait for the Groq quota window to reset. |
| 2026-11-01 | `4-fastapi-service` | Began the API phase: added a startup-cached retriever, validated `/predict`, `/health`, generated `/docs`, compact response/error contracts, explicit service dependencies, documentation, and four credential-free tests while preserving the existing traced RAG pipeline. |
| 2026-11-02 | `4-fastapi-service` | Completed manual API acceptance: the user confirmed that both `/health` and `/predict` work through the generated FastAPI `/docs` interface. |
| 2026-11-02 | `3-mlflow-genai-datasets-judges` | Resumed after a successful 19.53-second preflight and completed three additional profiles, reaching 7/8 valid; `fixed-hybrid-top5` remained partial after six scorer timeouts and was explicitly marked invalid for comparison. |
| 2026-11-02 | `5-milestone-documentation` | Began the rubric audit: confirmed required preparation/loading/API tests, added the README pitch, architecture, reproducible quick start, dataset/task note, and safe environment template; contribution details await the team names and ERP IDs. |
| 2026-11-03 | `3-mlflow-genai-datasets-judges` | Completed `fixed-hybrid-top5` with all 18 assessments valid, closed the comparison at 8/8, and SDK-verified `fixed-hybrid-top5` as the only MLflow best run. |
| 2026-11-03 | `6-gradio-interface` | Added and verified the local Gradio Miras interface, including a real Groq-fallback prediction and nested MLflow trace readback. |
| 2026-11-05 | `7-inference-ablation-evaluation` | Added the controlled four-run prompt/model/temperature suite; completed baseline, prompt, and model comparisons, while the high-temperature judge pass remains checkpointed after quota throttling. |
| 2026-11-05 | `7-inference-ablation-evaluation` | Completed the high-temperature retry after replacing the local Groq key and recovering from a transient Pinecone DNS failure; the controlled comparison is now 4/4 valid and selects the structured prompt. |
| 2026-11-05 | `8-m1-submission` | Finalized `CONTRIBUTION.md` with student identities and a milestone-aligned division of modeling/evaluation, pipeline/API, and quality/operations responsibilities. |
| 2026-11-05 | `8-m1-submission` | Generated and visually verified the 17-page illustrated Milestone 1 PDF report using final tracked artifacts plus live proof of MLflow, FastAPI, Gradio, and public corpus access. |
| 2026-11-11 | `8-m1-submission` | Removed the retired Hugging Face generation path and dependency, made Groq the sole production provider with bounded retries, preserved retrieval and nested MLflow tracing, updated current setup/architecture documentation, and passed 31 focused generation, tracing, API, Gradio, and evaluation-adapter tests. Historical artifacts remain unchanged as evidence of their original runs. |
| 2026-11-12 | `8-m1-submission` | Added per-stage retrieval timings, configurable 10/10/12 hybrid candidate limits, a repeatable warm latency benchmark, and the version-1 `fact_coverage` judge while retaining strict binary correctness. Registration reused six judges and added one without inference; a one-call partial-answer smoke test correctly returned 3/4 = 0.75. The 45-request fixed-hybrid-top5 benchmark recorded 3,730 ms mean, 3,451 ms median, and 5,469 ms p95 retrieval latency; reranking accounted for 87.6% of mean time. |
| 2026-11-12 | `8-m1-submission` | Ran the separate `fixed-hybrid-top5-improved-v2` managed evaluation (`518b7c4824cb4a74bc9dcbd676058b3a`). Fact coverage succeeded on all rows with values 0.75, 0.75, and 0.333 (mean 0.611), while strict correctness remained 0.333. Mean end-to-end inference fell to 7,434 ms (6,099 ms retrieval + 1,335 ms generation). The run is correctly marked partial because the legacy retrieval-sufficiency judge returned malformed JSON on two rows; historical artifacts were not overwritten. |
| 2026-11-12 | `8-m1-submission` | Reworked the unreliable seven-call-per-row judge path into one quota-aware, schema-validated GPT-OSS core scorer per query. Valid run `96c173dc72b8408c8541e49ddff7ae7f` completed 3/3 rows in 1:23 with fact coverage 0.7222, relevance 0.8333, retrieval relevance 0.8333, groundedness 0.6000, and 7,509 ms mean RAG inference. Failed diagnostics remain partial or killed; new comparison artifacts use a separate `core_v2` namespace so historical evidence is preserved. |
| 2026-11-12 | `9-post-m1-fixes` | Completed the quota-aware core-v2 evaluation matrix at 8/8 valid profiles and preserved every original milestone artifact. `fixed-hybrid-top5` remains the quality champion at 0.7222 fact coverage; `fixed-semantic-top3` is the primary latency challenger at 2,492 ms, 0.8833 groundedness, 0.8667 retrieval relevance, and 0.5833 fact coverage. The canonical comparison is stored in `artifacts/genai_evaluation_comparison_core_v2.json`. |
| 2026-11-13 | `9-post-m1-fixes` | Completed matching 75-request warm retrieval benchmarks for the champion and challenger. `fixed-hybrid-top5` recorded 6,975/6,754/12,211 ms mean/median/p95; `fixed-semantic-top3` recorded 528/394/1,341 ms and is 92.4% faster by mean. A live Gradio request also passed end to end through Pinecone and Groq in 10,699 ms. |
| 2026-11-13 | `9-post-m1-fixes` | SDK readback verified both latency runs as `FINISHED` with the expected 15-query, five-repeat parameters and matching stored metrics. Generated and visually verified a revised 18-page PDF report and seven-slide editable PPTX covering core-v2 fact coverage, the 150-request latency comparison, 108 passing tests, and the champion-challenger deployment recommendation. |
| 2026-11-13 | `9-post-m1-fixes` | Re-ran all four prompt/model/temperature configurations with the corrected core-v2 lineage and preserved the original six-judge artifacts. The quality-first rule selects temperature 0.7 at 0.8333 fact coverage, while groundedness falls to 0.6667; the structured prompt is the balanced challenger at 0.8667 groundedness/retrieval relevance and 0.9333 query relevance. Two model-run attempts remain partial evidence because transient Pinecone DNS/read failures produced only 1/3 predictions. Added bounded Pinecone transport retries and passed the complete 109-test regression. |
| 2026-11-13 | `9-post-m1-fixes` | Updated the README, concise brief, 18-page report, and seven-slide core-v2 deck with the corrected inference ablation. The PDF passed full 18-page render inspection. The new PPTX passed package and layout validation with seven slides, two editable charts, and zero findings; native PowerPoint screenshot capture was unavailable in this session, so that limitation remains explicit. |
| 2026-11-13 | `9-post-m1-fixes` | Completed a controlled 24-call inference-latency rerun across the four prompt/model/temperature configurations. Shared warm retrieval, rotated order, disabled judges, and unmeasured quota cooldown removed major confounders. All calls passed without retries; generation medians were 0.759-1.590 seconds and total medians 5.330-6.091 seconds, confirming the original 52.7-106.6 second outliers were transient network/retry effects. |
| 2026-11-22 | `10a-m2-docker-cpu-image` | Moved the CPU-only image fix onto branch 10 and verified it locally at `8a531de`: 784 MB content / 3.18 GB unpacked (versus 10.3 GB), 512.6 s build, first start stopped by a transient Pinecone DNS error, recreated container ready in 16.4 s, `/health` ok and `/docs` 200, `/predict` HTTP 200 for `recursive` (3,618 ms, 2 sources) and `fixed` (3,684 ms, 3 sources), 600.2 MiB current and 756.1 MiB peak memory under a 2 GiB limit with 0 OOM events, no `.env` in the image. Fixed the `HANDOUT.md` request examples to use `query`. |
| 2026-11-22 | `12-m2-render-deployment` | Recorded the Render free-plan out-of-memory failure, added the Azure Container Apps deploy script (1 vCPU, 2 GiB, scale to zero, secrets prompted in Cloud Shell), documented the local build and Docker Hub push, and added a bounded Pinecone startup retry with two unit tests. Live Azure checks pending. |
| 2026-11-22 | `12-m2-render-deployment` | Deployed image `75571d7` from Docker Hub (`rayyankhan007/legal-aid-rag-api`) to Azure Container Apps in `swedencentral`. The first revision crash-looped on a Pinecone 401 from a mistyped secret; after re-entering both keys and rolling a new revision, public `/health`, `/docs`, and `/predict` for `recursive` (1,573 ms server total) and `fixed` (3,402 ms) passed through Groq without fallback, and invalid input returned 422. Documented the live URL and the key-replacement procedure. |
| 2026-11-22 | `12-m2-render-deployment` | Hardened key entry in the Azure deploy script: `deploy/azure/read-key.sh` cleans quotes, `NAME=` prefixes, spaces, and line endings, checks the `pcsk_`/`gsk_` prefixes, and re-prompts; `UPDATE_KEYS=1` replaces both keys on the live app and rolls a new revision. Twelve new tests passed locally with Git Bash, and a stubbed dry run confirmed the Azure calls for four scenarios. |
| 2026-11-22 | `13-m2-monitoring-runbook` | Added in-memory request metrics, one safe JSON access-log line per counted request with `X-Request-ID`, the `GET /metrics` endpoint, `--no-access-log` in the image, and `RUNBOOK.md`. Eighteen new monitoring tests and the existing API, container, and deploy-key tests passed locally (42), and Ruff passed. Redeployment evidence pending. |
| 2026-11-22 | `13-m2-monitoring-runbook` | Deployed `996bf7f` and verified live metrics (0 to 4 requests, 3/1 success/failure) with `X-Request-ID` on every response; recorded a scale-to-zero 502 seen after an idle period. At the team's request, mounted the branch-8/9 Miras Gradio interface at `/ui` with `/` redirecting to it (`src/web.py`), sharing the API's loaded retrievers and counting interface questions as `POST /ui/predict`; 58 local tests and Ruff passed. |
| 2026-11-22 | `13-m2-monitoring-runbook` | Deployed `3062327`: the public URL now opens the Miras interface over HTTPS, a browser question was answered, `/docs` and `/predict` still work, `/metrics` counted interface questions (3: 2 answered, 1 empty question rejected), and the Azure log stream showed the JSON access lines without question text. Recorded `3062327` as a known-good release. Phase 12 is complete pending team acceptance. |
| 2026-11-22 | `14-m2-submission` | Added the Milestone 2 requirement/evidence table to the README, labelled Render as the failed capacity experiment across the docs, and added six live and CI screenshots in `report/m2-evidence/`. A fresh clone passed Ruff and 58 focused tests, built an image that ran healthy and answered through the API and the interface, the public service passed its smoke test, and CI passed on `a48712c`. Measured a 15.1 s cold start. The Milestone 2 `CONTRIBUTION.md` table awaits the team. |
| 2026-11-25 | `14-m2-submission` | Audited the revised instructor requirements; added the explicit MLflow/LLM-as-a-Judge Project Focus, the M2 contribution allocation, `AI_USAGE.md`, and a timed video script. Confirmed `/metrics` is live but `/stats` returns 404, so the alias, redeployment, and final recording remain pending. |
| 2026-11-25 | `14-m2-submission` | Added a seven-slide Markdown presentation source for the live demo, with timed speaker notes, Docker/CI/Azure demonstration cues, evaluation results, and optional backup slides. |
| 2026-11-25 | `14-m2-submission` | Added a top-level README summary of the post-M1 latency, correctness, judge-reliability, and provider problems, the branch-9 fixes, and the verified results. |
| 2026-11-25 | `13-m2-monitoring-runbook` | Added the required `GET /stats`: the `/metrics` fields plus fail-safe, privacy-safe question aggregates (error rate, rate per minute, status counts, latency p50/p90/p95/p99), a `/stats` healthcheck and deploy smoke check, and `scripts/stats_demo.py`. 21 new tests; 79 focused tests and Ruff passed; the demo passed against the rebuilt image with real keys. Live redeploy pending. |
