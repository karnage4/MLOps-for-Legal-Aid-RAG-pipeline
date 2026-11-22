# Milestone 2 evidence

Screenshots of the deployed service and CI, captured on 2026-11-22 (PKT,
UTC+5) from the public URL
<https://legal-aid-rag-api.whitesand-d7432c32.swedencentral.azurecontainerapps.io>
running image `docker.io/rayyankhan007/legal-aid-rag-api:3062327`. None shows
an API key, a secret value, or a private question. The matching measurements
are in `MLOPS_LOG.md` (Phases 11-13).

| File | Shows | Requirement |
| --- | --- | --- |
| [`00-live-interface.png`](00-live-interface.png) | The public URL opening the Miras Gradio interface at `/ui/` (headless Edge, 23:07) | Deployment |
| [`01-live-interface-answer.jpg`](01-live-interface-answer.jpg) | The live interface answering "What share does a daughter receive in Islamic inheritance?" with quoted, sourced context (23:06) | Deployment |
| [`02-live-metrics.png`](02-live-metrics.png) | `GET /metrics` about 71 s after a cold start: 8 requests, `POST /ui/predict` counted with its latency, failures grouped under `unmatched` | Monitoring |
| [`03-live-docs.png`](03-live-docs.png) | The live OpenAPI docs listing `/metrics`, `/health`, and `/predict` | Deployment, README |
| [`04-github-actions-run.png`](04-github-actions-run.png) | GitHub Actions run 45033847322 for `3062327` (the deployed image): Ruff lint, Pytest, and Docker build passed | CI |
| [`05-github-actions-main-run.png`](05-github-actions-main-run.png) | GitHub Actions run 43652828766 on `main` (`3a3c422`): the required Docker build on `main` passed | CI build on `main` |

Before this capture the app had scaled to zero; the first `/health` request
woke it and returned 200 after 15.1 s (`started_at` 18:05:40 UTC).

## Still to add (needs the Azure account holder)

Signed-in Azure Portal pages cannot be captured from outside the account.
Hide the subscription ID and never open the **Secrets** values.

1. `legal-aid-rag-api` > **Overview**: status Running, the application URL,
   and region Sweden Central.
2. **Application** > **Revisions and replicas**: the active revision and its
   image `rayyankhan007/legal-aid-rag-api:3062327`.
3. **Monitoring** > **Log stream**: a few `"event": "http_request"` lines
   after using the interface.
