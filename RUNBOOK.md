# Legal-Aid RAG API Runbook

How to check, diagnose, redeploy, roll back, and shut down the public API.
`MLOPS_LOG.md` (Phases 11 and 12) records which of these steps have been run
against the live service and with what result. Commands marked **Cloud Shell** run in the Azure Portal's Cloud Shell in
**Bash** mode (type `bash` first if the prompt starts with `PS`). Commands
marked **PowerShell** run on a team laptop with Docker Desktop.

## 1. System summary

```text
client ──HTTPS──> Azure Container Apps: legal-aid-rag-api (1 vCPU, 2 GiB, 0-1 replicas)
                    │  image docker.io/rayyankhan007/legal-aid-rag-api:<git-sha>
                    │  (BM25 corpora and both Hugging Face models are baked in)
                    ├──> Pinecone index legal-inheritance-rag (AWS us-east-1;
                    │      default namespace = recursive, "fixed" = fixed chunks)
                    └──> Groq API, model qwen/qwen3.8-27b (answer generation)
```

| Item | Value |
| --- | --- |
| Public URL | <https://legal-aid-rag-api.whitesand-d7432c32.swedencentral.azurecontainerapps.io> (redirects to the Miras interface at `/ui/`) |
| JSON API | `/health`, `/predict`, `/metrics`, `/stats`; interactive docs at `/docs` |
| Entry point | `uvicorn src.web:app`: the API from `src/app.py` with `gradio_app.py` mounted at `/ui`, sharing one set of loaded retrievers |
| Azure subscription | Azure for Students |
| Region | `swedencentral` (allowed by the student policy; nearest to Pinecone `us-east-1`) |
| Resource group / environment / app | `legal-aid-rag-rg` / `legal-aid-rag-env` (Express) / `legal-aid-rag-api` |
| Image registry | Docker Hub, public repository `rayyankhan007/legal-aid-rag-api` |
| Image tags | Short Git SHA of the commit the image was built from (immutable by convention) |
| Deploy script | [`deploy/azure/deploy-containerapp.sh`](deploy/azure/deploy-containerapp.sh) |
| Source branch | `14-m2-submission` (final Milestone 2 branch) on `karnage4/MLOps-for-Legal-Aid-RAG-pipeline` |

The app scales to zero about five minutes after its last request and starts
again on the next request. A cold start pulls the image and loads both
retrievers: on 2026-11-22 the first `/health` request after scale to zero
returned 200 after 15.1 s. Callers may see HTTP 503 or a slow first response
until it finishes.

MLflow traces go to a SQLite file inside the container (`/app/mlflow.db`) and
are lost when the replica stops; production tracing is not retained.

### Known-good releases

Roll back only to an image listed here.

| Image tag | Image digest (manifest list) | Deployed | Verified |
| --- | --- | --- | --- |
| `75571d7` | `sha256:40d92abae90670d2c624b6a87a11bd5c6bbc9a49082e0ec3a2ce7ba2ba5367bc` | 2026-11-22 | `/health`, `/docs`, `/predict` (both strategies); no `/metrics`, no interface |
| `996bf7f` | `sha256:471c12f8bf1b5b64e9b25ad6816a55ea60c080cd951c28593ccc6b3095f54421` | 2026-11-22 | Adds `/metrics` and JSON access logs: `/health`, `/predict` (both strategies), 422 rejection, and changing counters verified live; no interface |
| `3062327` | `sha256:e0969395e040a38a92369afa4990f72fc42e2ee34d2b8674cbbbdc238c12c125` | 2026-11-22 | Adds the Miras interface at `/ui` (`/` redirects): interface question answered in a browser, `/docs` and `/predict` 200, `POST /ui/predict` counted, JSON log lines in Azure |
| `a5d818a` | `sha256:8160d38205527adfd28796a99005b2ae7d130b1198e6ec9842d144d4d739e0db` | 2026-11-25 (running) | Adds `GET /stats` and fail-safe monitoring: deploy-script `/stats` check passed; `scripts/stats_demo.py` passed live (0 to 4 questions, 3 answered, 1 rejected with 422) |

## 2. Configuration

Secrets are Container Apps secrets referenced by environment variables. Never
write their values into the repository, chat, screenshots, or tickets.

| Variable | Source | Purpose |
| --- | --- | --- |
| `PINECONE_API_KEY` | secret `pinecone-api-key` | Pinecone access (keys start `pcsk_`) |
| `GROQ_API_KEY` | secret `groq-api-key` | Groq access (keys start `gsk_`) |
| `GROQ_MODEL` | plain | Generation model, `qwen/qwen3.8-27b` |
| `GROQ_MAX_ATTEMPTS` | plain | Retries for transient Groq errors (3) |
| `BM25_CANDIDATE_K`, `SEMANTIC_CANDIDATE_K`, `RERANK_CANDIDATE_K` | plain | Hybrid retrieval candidate sizes (10/10/12) |
| `PINECONE_QUERY_MAX_ATTEMPTS`, `PINECONE_RETRY_BASE_SECONDS` | plain | Per-query Pinecone retries (3, 1.5 s) |
| `PINECONE_INIT_MAX_ATTEMPTS` | plain | Startup Pinecone retries (5) |
| `MLFLOW_EXPERIMENT_NAME` | plain | `legal-aid-rag-azure` |
| `KEYS_ROTATED_AT` | plain | Timestamp of the last key replacement; changing it starts a new revision |

Show the live configuration without secret values (**Cloud Shell**):

```bash
az containerapp show -n legal-aid-rag-api -g legal-aid-rag-rg \
  --query "{image: properties.template.containers[0].image, env: properties.template.containers[0].env, scale: properties.template.scale}" -o json
```

## 3. Routine checks

**PowerShell** (or any browser for the GET requests):

```powershell
$url = "https://legal-aid-rag-api.whitesand-d7432c32.swedencentral.azurecontainerapps.io"
Invoke-RestMethod "$url/health"
Invoke-RestMethod "$url/stats" | ConvertTo-Json -Depth 5
$body = @{ query = "Who inherits property when a person dies without a will?"; chunking_strategy = "fixed" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$url/predict" -ContentType "application/json" -Body $body
```

Healthy means: `/health` returns `status: ok` with `retriever_loaded: true`
and both strategies; `/predict` returns HTTP 200 with `provider: groq` and
`fallback_used: false`; `/stats` returns HTTP 200.

To demonstrate monitoring end to end, run
`python scripts/stats_demo.py --url $url` (standard library only): it prints
`/stats`, sends three questions and one rejected question, and prints
`/stats` again with a before/after summary.

### Metrics and stats

`GET /metrics` reports, since the current replica started: `started_at`,
`uptime_seconds`, `request_count`, `success_count`, `failure_count`,
`average_latency_ms`, and the same counts per route (for example
`POST /predict`).

`GET /stats` returns the same fields plus `queries`, aggregated over
questions only (`POST /predict` and `POST /ui/predict`): `total`,
`succeeded`, `failed`, `error_rate` (null with no questions), `per_minute`,
`by_status`, `by_endpoint`, and `latency_ms` with `mean`, `min`, `max`,
`p50`, `p90`, `p95`, and `p99` over the last 1,000 successful answers (null
until there is one). Watch `queries.error_rate` and `queries.by_status`: a
rise in `502` points at Groq or retrieval failures, `422` at client input.

Rules (see `src/monitoring.py`):

- every request to an API route is counted except reads of `GET /metrics`
  and `GET /stats`;
- the interface's page, assets, and event stream under `/ui` are not counted;
  each question asked in the interface is counted as `POST /ui/predict`
  (status 200, 422 for invalid input, 502 for a retrieval or generation
  failure, 500 otherwise);
- status 400 or higher, or an unhandled exception, is a failure;
- unknown paths are grouped under the route `unmatched`;
- counters are in memory and reset on every restart, redeploy, or scale to
  zero, so a small `uptime_seconds` after an idle period is normal;
- monitoring is fail-safe: if a collector fails, the request still completes
  and a warning from the `src.monitoring` logger (without request data)
  appears in the log stream. The Docker healthcheck reads `/stats`, so a
  broken endpoint shows up as an unhealthy local container.

### Logs

Each counted request writes one JSON line to stdout and nothing else about
the request; questions, answers, headers, query strings, and keys are never
logged:

```json
{"event": "http_request", "timestamp": "2026-11-22T14:05:01.123+00:00", "request_id": "3f2c...", "method": "POST", "route": "/predict", "status": 200, "duration_ms": 1573.2}
```

The same `request_id` is returned in the `X-Request-ID` response header, so a
caller's failing request can be found in the log. Read logs (**Cloud Shell**):

```bash
az containerapp logs show -n legal-aid-rag-api -g legal-aid-rag-rg --tail 100
az containerapp logs show -n legal-aid-rag-api -g legal-aid-rag-rg --tail 300 | grep http_request
az containerapp logs show -n legal-aid-rag-api -g legal-aid-rag-rg --follow
```

Startup messages and Python tracebacks appear in the same stream. If the CLI
returns `Got bad status from the logstream API`, open the app in the Azure
Portal and use **Monitoring > Log stream** instead.

## 4. Diagnosis

| Symptom | Likely cause | Check | Fix |
| --- | --- | --- | --- |
| HTTP 503 for under a minute after a pause | Cold start after scale to zero | Wait 60 s and retry `/health` | None needed |
| HTTP 502 with no `X-Request-ID` header, about five minutes after the last request | The replica was being scaled to zero as the request arrived (seen once on 2026-11-22: `/metrics` then showed a new `started_at` about 20 s later) | Retry after 20-30 s; `/metrics` shows a fresh `started_at` | None needed; repeated 502s while requests keep flowing are not this case |
| Interface says the legal search index is still loading | A question arrived before startup finished loading the retrievers | `/health` | Wait until `/health` returns 200, then ask again |
| HTTP 503 for several minutes | Container crash-loops at startup | Logs end with `Application startup failed` | Read the traceback above it; see the rows below |
| `pinecone ... 401 UNAUTHENTICATED Invalid API key` | Wrong or mistyped Pinecone secret (the 2026-11-22 first-deployment incident) | Logs | Replace the keys (section 6) |
| Pinecone DNS or timeout errors at startup | Transient network fault | Logs show retries up to `PINECONE_INIT_MAX_ATTEMPTS` | Usually recovers on its own; if it persists, check Pinecone's status page and console |
| `/predict` HTTP 502 `Answer generation failed` | Groq quota, rate limit, outage, or invalid Groq key | `failure_count` on `POST /predict` rises; the response `detail` names the error | Wait for the quota window; check the Groq console; replace the key if it is invalid |
| `/predict` HTTP 404 `No relevant legal context` | Retrieval returned nothing | Try another question; check the Pinecone index and namespaces | Re-upload vectors only if the index was changed |
| `/predict` HTTP 422 | Invalid request body (too short, unknown field, bad option) | Response `detail` | Client-side fix |
| Container restarts, exit code 137 | Out of memory | Portal **Metrics > Memory Working Set**; the measured local peak is 931.7 MiB of the 2 GiB limit | Raise memory with `az containerapp update --memory`, or roll back |

## 5. Redeploy a new version

1. Commit and push the change, then build and push an image tagged with that
   commit (**PowerShell**, signed in to Docker Hub in Docker Desktop):

   ```powershell
   $tag = git rev-parse --short HEAD
   docker build --tag docker.io/rayyankhan007/legal-aid-rag-api:$tag .
   docker push docker.io/rayyankhan007/legal-aid-rag-api:$tag
   ```

   Never move an existing tag to a different build; a new build gets a new
   commit and tag.
2. Optionally smoke-test the image locally first:
   `docker run --rm --env-file .env -p 8000:8000 docker.io/rayyankhan007/legal-aid-rag-api:$tag`,
   then run the section 3 checks against `http://localhost:8000`.
3. Roll the app to the new image (**Cloud Shell**, from a clone of the branch):

   ```bash
   git clone --branch 14-m2-submission --depth 1 https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline.git
   cd MLOps-for-Legal-Aid-RAG-pipeline
   IMAGE=docker.io/rayyankhan007/legal-aid-rag-api:<tag> LOCATION=swedencentral bash deploy/azure/deploy-containerapp.sh
   ```

   The script updates the image (a new revision), waits for `/health`, and
   prints the URL.
4. Run the section 7 verification and add the tag to the known-good table.

## 6. Replace the API keys

**Cloud Shell**, passing the image that is currently running:

```bash
UPDATE_KEYS=1 IMAGE=docker.io/rayyankhan007/legal-aid-rag-api:<current-tag> LOCATION=swedencentral bash deploy/azure/deploy-containerapp.sh
```

Paste each key when asked. The script removes quotes and `NAME=` prefixes,
rejects keys without the `pcsk_`/`gsk_` prefix, stores the secrets, and sets
`KEYS_ROTATED_AT` so a new revision reads them. Express environments do not
support `az containerapp revision restart`, and the Cloud Shell CLI has no
`az containerapp stop`/`start`.

## 7. Roll back

1. Choose the newest tag in the known-good table that predates the problem.
2. **Cloud Shell**:

   ```bash
   IMAGE=docker.io/rayyankhan007/legal-aid-rag-api:<known-good-tag> LOCATION=swedencentral bash deploy/azure/deploy-containerapp.sh
   ```

   To pin the exact bytes, use the digest instead of the tag:
   `IMAGE=docker.io/rayyankhan007/legal-aid-rag-api@sha256:<digest>`.
3. Confirm the running image:

   ```bash
   az containerapp show -n legal-aid-rag-api -g legal-aid-rag-rg --query "properties.template.containers[0].image" -o tsv
   ```

4. Verify after any redeploy or rollback:
   - the public URL opens the Miras interface and one question gets an answer
     (only on images that include the interface);
   - `/health` returns HTTP 200 with `retriever_loaded: true`;
   - `/docs` returns HTTP 200;
   - `/predict` returns HTTP 200 for `chunking_strategy` `recursive` and `fixed`;
   - `/metrics` shows a fresh `started_at` and counts that rise with your
     requests (only on images that include monitoring);
   - `/stats` returns HTTP 200 and its `queries.total` rises with your
     questions (only on images that include `/stats`; the deploy script
     checks the 200 itself);
   - the logs show one `http_request` line per request and no question text.

Rolling back the image does not roll back secrets; if the keys changed, also
run section 6.

## 8. Shut down after grading, and restore

- **Idle (default):** the app already scales to zero and charges no compute
  time while idle. It wakes automatically on the next request.
- **Stop (no automatic wake-up):** Azure Portal > `legal-aid-rag-api` >
  **Overview** > **Stop**. Requests then fail until **Start** is clicked. The
  configuration and secrets are kept.
- **Delete everything (cannot be undone):**
  `az group delete -n legal-aid-rag-rg` (**Cloud Shell**). To restore, re-run
  the first-deployment command from the README with a known-good tag and
  re-enter both keys; the URL will change.

When the service is stopped or deleted, record the date, the last verified URL,
and the last image tag in `README.md` and `MLOPS_LOG.md`.

## 9. Ownership

| Area | Owner |
| --- | --- |
| Azure for Students subscription, Docker Hub `rayyankhan007`, deployments | Muhammad Rayyan Khan |
| Retrieval, Pinecone index, evaluation | Team (see `CONTRIBUTION.md`) |
| Groq and Pinecone API keys | Key holders on the team; rotate with section 6 |

Escalate by contacting the team members listed in `CONTRIBUTION.md`.
