# Branch 13 Handoff: Monitoring and Runbook

## Start Condition

Create `13-m2-monitoring-runbook` only after branch 12 has a healthy public deployment. Start it from the verified branch-12 tip:

```powershell
git switch 12-m2-render-deployment
git pull --ff-only origin 12-m2-render-deployment
git switch -c 13-m2-monitoring-runbook
```

If deployment is still blocked by memory, remain on branch 12. Monitoring code does not fix insufficient hosting resources.

## Required Outcome

Demonstrate request count and average latency during grading through safe structured logs and a small operations endpoint. Add an executable `RUNBOOK.md` for redeploy and rollback, then redeploy and capture public evidence.

## Monitoring Design

Add lightweight FastAPI middleware that records, at minimum:

- timestamp;
- HTTP method and normalized route/path;
- response status;
- elapsed milliseconds; and
- a generated/request correlation ID if useful.

Do not log request bodies, questions, retrieved legal text, answers, headers, API keys, cookies, or environment values. Keep this operational logging separate from MLflow GenAI traces.

Add a read-only operations endpoint, preferably `/metrics` or `/operations/metrics`, returning:

```json
{
  "uptime_seconds": 123.4,
  "request_count": 10,
  "success_count": 9,
  "failure_count": 1,
  "average_latency_ms": 456.7
}
```

Decide and document whether health/metrics requests count toward totals. Apply the same rule consistently in code and tests.

An in-memory, lock-protected counter is acceptable for the milestone because the service intentionally uses one worker. Document that counters reset on restart, redeploy, crash, or scale-to-zero and are not aggregate multi-instance telemetry.

## Implementation Checklist

1. Add a small metrics state object with monotonic start time, counters, total elapsed time, and safe concurrency protection.
2. Add middleware using `time.perf_counter()` and `try/finally` so failures are counted.
3. Emit one machine-readable log line per request without sensitive content.
4. Add the operations endpoint and Pydantic response schema.
5. Ensure middleware does not break streaming/errors and does not hide original exceptions.
6. Add unit tests before redeployment.
7. Update README with endpoint examples and limitations.
8. Add `RUNBOOK.md` using the actual chosen host, service name, and verified commands.

## Minimum Tests

- successful request increments total and success counts;
- 4xx/5xx response increments failure count according to the documented rule;
- average latency equals accumulated duration divided by counted requests;
- uptime is nonnegative and increases;
- metrics endpoint schema is stable;
- request bodies and secret-like values do not appear in captured logs;
- concurrent requests do not lose increments;
- existing `/health`, `/predict`, chunking validation, MLflow, and retrieval tests still pass.

Run:

```powershell
.venv\Scripts\ruff.exe check src tests gradio_app.py
.venv\Scripts\python.exe -m pytest -q
docker build --tag legal-aid-rag-api:m2-monitoring .
```

## RUNBOOK.md Content

The runbook must be specific enough for another team member to operate the service without oral context:

- system summary and dependency diagram/reference;
- live service URL and hosting project/service/revision identifiers;
- required environment-variable names, but never values;
- normal health and monitoring checks;
- log location and safe filters;
- diagnosis for startup timeout, OOM, Pinecone failure, Groq quota/failure, and unhealthy checks;
- redeployment from a verified Git SHA/image tag;
- rollback to the last known-good immutable revision/image tag;
- post-redeploy/rollback `/health`, `/docs`, `/predict`, and metrics verification;
- escalation/contact ownership;
- how to shut down after grading and how to restore it.

Rollback must name an immutable Git SHA, container digest/tag, or managed-platform revision. “Push the old code again” is not a sufficient rollback procedure.

## Public Demonstration

After redeploying:

1. capture the initial metrics response;
2. send several `/health` and `/predict` requests, including one safe failure such as invalid validation input;
3. capture the changed counter and average latency;
4. capture host logs showing safe structured entries;
5. verify no query text, legal response, or credential is exposed;
6. record endpoint, Git SHA, deployment/revision ID, UTC/local timestamp, and screenshots in the repository evidence area.

## Acceptance Gate

- monitoring values visibly change after requests;
- request count and average latency are demonstrable;
- logs contain one safe structured entry per counted request;
- all automated tests and CI jobs pass;
- the public service remains healthy;
- redeploy and rollback instructions are tested or dry-run to the furthest safe point and are unambiguous;
- README and `MLOPS_LOG.md` are current.

When all conditions pass, push branch 13 and create branch 14 from its verified tip.
