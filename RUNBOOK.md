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
| Public URL | <https://legal-aid-rag-api.whitesand-d7432c32.swedencentral.azurecontainerapps.io> |
| Interactive docs | `/docs` on the URL above |
| Azure subscription | Azure for Students |
| Region | `swedencentral` (allowed by the student policy; nearest to Pinecone `us-east-1`) |
| Resource group / environment / app | `legal-aid-rag-rg` / `legal-aid-rag-env` (Express) / `legal-aid-rag-api` |
| Image registry | Docker Hub, public repository `rayyankhan007/legal-aid-rag-api` |
| Image tags | Short Git SHA of the commit the image was built from (immutable by convention) |
| Deploy script | [`deploy/azure/deploy-containerapp.sh`](deploy/azure/deploy-containerapp.sh) |
| Source branch | `13-m2-monitoring-runbook` on `karnage4/MLOps-for-Legal-Aid-RAG-pipeline` |

The app scales to zero about five minutes after its last request and starts
again on the next request. A cold start pulls the image and loads both
retrievers (about 15 seconds locally once the image is present); callers may
see HTTP 503 or a slow first response until it finishes.

MLflow traces go to a SQLite file inside the container (`/app/mlflow.db`) and
are lost when the replica stops; production tracing is not retained.

### Known-good releases

Roll back only to an image listed here.

| Image tag | Image digest (manifest list) | Deployed | Verified |
| --- | --- | --- | --- |
| `75571d7` | `sha256:40d92abae90670d2c624b6a87a11bd5c6bbc9a49082e0ec3a2ce7ba2ba5367bc` | 2026-11-22 | `/health`, `/docs`, `/predict` (both strategies); no `/metrics` |

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
Invoke-RestMethod "$url/metrics"
$body = @{ query = "Who inherits property when a person dies without a will?"; chunking_strategy = "fixed" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$url/predict" -ContentType "application/json" -Body $body
```

Healthy means: `/health` returns `status: ok` with `retriever_loaded: true`
and both strategies; `/predict` returns HTTP 200 with `provider: groq` and
`fallback_used: false`.

### Metrics

`GET /metrics` reports, since the current replica started: `started_at`,
`uptime_seconds`, `request_count`, `success_count`, `failure_count`,
`average_latency_ms`, and the same counts per route (for example
`POST /predict`). Rules (see `src/monitoring.py`):

- every request is counted except `GET /metrics` itself;
- status 400 or higher, or an unhandled exception, is a failure;
- unknown paths are grouped under the route `unmatched`;
- counters are in memory and reset on every restart, redeploy, or scale to
  zero, so a small `uptime_seconds` after an idle period is normal.

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
   git clone --branch 13-m2-monitoring-runbook --depth 1 https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline.git
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
   - `/health` returns HTTP 200 with `retriever_loaded: true`;
   - `/docs` returns HTTP 200;
   - `/predict` returns HTTP 200 for `chunking_strategy` `recursive` and `fixed`;
   - `/metrics` shows a fresh `started_at` and counts that rise with your
     requests (only on images that include monitoring);
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
