# Branch 14 Handoff: Milestone 2 Submission

## Start Condition

Create `14-m2-submission` only after deployment, monitoring, public tests, and runbook work on branch 13 are complete and pushed:

```powershell
git switch 13-m2-monitoring-runbook
git pull --ff-only origin 13-m2-monitoring-runbook
git switch -c 14-m2-submission
```

This branch is for final evidence, documentation, contribution attribution, and reproducibility fixes. Do not introduce broad architecture changes here.

## Milestone 2 Rubric Tally

Create a table in README or a final M2 report mapping every requirement to code and evidence:

| Requirement | Required evidence |
|---|---|
| Dockerfile | Clean build, correct API command/port, source + artifacts included, `/health` Docker healthcheck |
| CI lint | Visible green Ruff job |
| CI tests | Visible green full pytest job |
| CI build | Visible green Docker build on `main` |
| Deployment | Reachable public containerized API on the chosen host |
| Monitoring | Request count and average latency in endpoint/log/dashboard |
| Runbook | Redeploy and immutable rollback instructions |
| README | Deployment steps, live/shutdown status, and public curl examples |
| Contribution | Names, ERP IDs, and M2 member-to-task table |

If the service is shut down after grading to avoid cost, state that explicitly in README with the last verified endpoint, date, Git SHA/revision, and restoration procedure.

## Contribution Information

Use the actual work completed during M2 and these identities:

- Karam Hussain — ERP `27857`
- Aadesh Panjwani — ERP `29303`
- Muhammad Rayyan Khan — ERP `28410`

Update `CONTRIBUTION.md` to an M2-specific member-to-task table. Do not copy percentages into the submitted file. Do not claim tasks a member did not perform; reconcile the table with Git history, logs, test evidence, and manual deployment work.

## Final Verification

From a fresh clone if time allows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
.venv\Scripts\ruff.exe check src tests gradio_app.py
.venv\Scripts\python.exe -m pytest -q
docker build --tag legal-aid-rag-api:m2-final .
```

Then run the container with runtime secrets, verify Docker health, and exercise:

- `/health`;
- `/docs`;
- `/predict` with recursive chunking;
- `/predict` with fixed chunking; and
- the monitoring endpoint.

Repeat the same smoke tests against the public URL. Record response status and latency without exposing secrets or private content.

## Evidence Checklist

- clean Docker build output/image inspection;
- healthy local container;
- GitHub Actions run showing all three jobs;
- host dashboard showing healthy deployment and immutable revision/image;
- public `/health`, `/docs`, and `/predict` results;
- public monitoring before/after requests;
- safe structured logs;
- rollback target and post-rollback verification evidence;
- no API keys in screenshots, shell history excerpts, logs, repository, or reports;
- README links and commands checked from a clean shell;
- `git status` clean and expected branch pushed.

Prefer screenshots with timestamps and visible service/run identifiers. Redact credentials, authorization headers, environment values, and any sensitive user query.

## Submission Archive

If a ZIP is required, archive the `Legal-Aid-RAG-MLflow` project content needed to reproduce the service. Exclude:

- `.env` and all credential files;
- `.venv/`;
- `mlflow.db` and `mlruns/`;
- caches and generated temporary files;
- raw corpus data not intended for submission;
- local IDE settings; and
- large unrelated legacy/reference directories.

Include source, tests, Docker/CI/host configuration, both versioned BM25 serving artifacts, README, RUNBOOK, MLOPS log, contribution document, requirements, and relevant portable evaluation/evidence artifacts. Test extraction into a new folder before submission.

## Documentation Consistency

Before finalizing, search for stale claims:

```powershell
rg -n "Hugging Face|Render|localhost|TODO|TBD|branch 12|512Mi|live endpoint" README.md RUNBOOK.md MLOPS_LOG.md CONTRIBUTION.md MILESTONE2_PLAN.md
```

Ensure the architecture reflects Groq-only generation and the actual selected deployment platform. If Render was abandoned for Cloud Run or another host, clearly label Render as a failed capacity experiment rather than the live deployment.

## Git Finalization

```powershell
git status
git diff --check
git log --oneline --decorate -15
git push -u origin 14-m2-submission
```

After final CI and public smoke tests pass, optionally create an annotated milestone tag and push it. Tag only the exact verified submission commit. Record the final commit SHA, CI URL, public revision/image, and submission archive checksum.

## Definition of Done

- every rubric item maps to implementation and visible evidence;
- Ruff, pytest, Docker build, local smoke tests, CI, and public smoke tests pass;
- deployment is healthy or its post-grading shutdown is documented;
- runbook redeploy/rollback instructions match the actual host;
- contribution information is accurate;
- no secret is committed or submitted;
- branch/tag/archive identifiers make the submission reproducible.
