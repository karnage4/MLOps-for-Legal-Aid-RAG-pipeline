# Milestone 2 Video Script and Checklist

Step-by-step recording script for the Milestone 2 video, following the
instructor's brief section by section. Companion slides, if wanted, are in
`M2_VIDEO_PRESENTATION.md`; the video itself must show a **real run**.

- **Target length: 6:00.** Required 4-6 minutes; **anything after 7:00 is not
  marked**. Rehearse once with a timer.
- **The last 0.5-1 minute must cover next steps for the focus area**
  (instructor note, 20 Nov).
- Each segment lists what to **SHOW**, what to **DO** (click or type), and what
  to **SAY**. Paraphrase the SAY lines in your own words.
- `/stats` is live as of 25 Nov (image `a5d818a`), so recording can go ahead.

| Time | Segment | Instructor requirement |
| --- | --- | --- |
| 0:00-0:40 | 1. Introduction | Project focus, overview, tech stack |
| 0:40-1:50 | 2. Clean start | GitHub repo, README `docker build` + `docker run`, `/docs`, `/health`, `/predict` locally |
| 1:50-2:30 | 3. Docker | Dockerfile briefly, `docker ps` showing `(healthy)` |
| 2:30-3:20 | 4. Monitoring | Several `/predict` calls, `/stats` request count and average latency rising |
| 3:20-4:15 | 5. CI | Lint, test, build green; a push triggers a run; a failing test turns it red |
| 4:15-5:10 | 6. Deployment | Live URL; `/health`, `/predict`, `/stats` on Azure with README cURL |
| 5:10-6:00 | 7. Focus and next steps | Next steps for the focus area (last 0.5-1 min) |

---

## 0. Setup (15 minutes before recording, not recorded)

Run everything in **PowerShell in the project folder**
(`...\MLOps-for-Legal-Aid-RAG-pipeline`) unless stated. `karnage4` is the
GitHub remote's name on the recording laptop; check yours with `git remote -v`
(often `origin`).

1. **Docker Desktop** is open and says **Engine running**.
2. **Latest code:**

   ```powershell
   git switch 14-m2-submission
   git pull
   ```

3. **Warm the build cache** so the on-camera build takes seconds instead of
   about 10 minutes. Wait for it to finish:

   ```powershell
   docker build --tag legal-aid-rag-api:local .
   ```

4. **Remove any old container** so the video starts clean (an error saying
   there is no such container is fine):

   ```powershell
   docker rm -f legal-aid-rag-api
   ```

5. **Prepare the red CI run** (about 3 minutes; do it early). This creates a
   throw-away branch with one deliberately failing test and pushes it:

   ```powershell
   git switch -c ci-demo
   Set-Content tests\test_ci_demo.py "def test_ci_demo_turns_the_run_red():`n    assert 1 + 1 == 3`n"
   git add tests\test_ci_demo.py
   git commit -m "CI demo: deliberately failing test"
   git push karnage4 ci-demo
   ```

   Stay on `ci-demo` for the recording; the Docker image is the same because
   tests are not copied into it. In the Actions tab, wait until this run shows
   **Pytest failed** (red) while Ruff and Docker build pass.
6. **Wake the Azure app and keep it awake.** It sleeps after about 5 idle
   minutes. Open a **second PowerShell window** (keep it off screen) and leave
   this running until the recording ends; stop it with **Ctrl+C**:

   ```powershell
   $url = "https://legal-aid-rag-api.whitesand-d7432c32.swedencentral.azurecontainerapps.io"; while ($true) { try { Invoke-RestMethod "$url/health" -TimeoutSec 60 | Out-Null; "$(Get-Date -Format T) awake" } catch { "$(Get-Date -Format T) waking..." }; Start-Sleep 60 }
   ```

7. **Browser tabs**, left to right:
   1. Repository on the submission branch:
      <https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline/tree/14-m2-submission>
   2. Dockerfile:
      <https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline/blob/14-m2-submission/Dockerfile>
   3. Actions: <https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline/actions>
   4. Live site:
      <https://legal-aid-rag-api.whitesand-d7432c32.swedencentral.azurecontainerapps.io>

   The local pages (`http://127.0.0.1:8000/docs` and `/stats`) are opened
   during the video.
8. **Terminals on screen:** the PowerShell window in the project folder, and
   **Git Bash** (for the README cURL commands in segment 6). Increase the font
   size (Ctrl + mouse wheel) so text is readable in the recording.
9. **Privacy:** never open `.env`; do not open Azure secrets or environment
   variable values; turn on Windows **Do not disturb** to hide notifications.
10. **Recorder:** Xbox Game Bar (**Win+Alt+R** starts and stops) or OBS, 1080p,
    microphone checked.

---

## 1. Introduction (0:00-0:40)

**SHOW:** tab 1, the README top: the title, **Project Focus**, and the
architecture diagram.

**SAY:**

> This is our Legal-Aid RAG system, Miras. It answers inheritance-law
> questions from a legal PDF corpus with retrieval-augmented generation: the
> corpus is chunked two ways, fixed and recursive, searched with BM25 and
> Pinecone vectors, reranked with a cross-encoder, and answered by a
> Groq-hosted Qwen model with the sources shown.
>
> Our project focus is extensive MLflow experimentation and LLM-as-a-Judge
> evaluation: we compared eight retrieval profiles and prompt, model, and
> temperature choices with versioned LLM judges, all tracked in MLflow.
>
> The stack is Python, FastAPI and a Gradio interface, Pinecone, Groq, and
> MLflow, packaged with Docker, tested with GitHub Actions, and deployed on
> Azure Container Apps.

---

## 2. Clean start (0:40-1:50)

**SHOW:** tab 1, scroll the README to **Deployment Notes** and the
`docker build` / `docker run` commands.

**SAY:**

> The README documents one way to build and run the service. I'll run exactly
> those commands on this machine.

**DO (PowerShell):**

```powershell
docker build --tag legal-aid-rag-api:local .
docker run --detach --name legal-aid-rag-api --env-file .env --publish 8000:8000 legal-aid-rag-api:local
```

**SAY (while it starts, about 30 seconds):**

> The build reuses cached layers from an earlier build; a cold build takes
> about ten minutes because it installs PyTorch and bakes in both models. The
> API keys come from a local `.env` file at run time and are never in the image.
> Startup loads both retrievers before the API answers.

**DO:** open a new tab: `http://127.0.0.1:8000/docs`. If it does not load yet,
wait 10 seconds and refresh.

1. Click **GET /health** > **Try it out** > **Execute**. Point at **200** and
   `"status": "ok"`, `"retriever_loaded": true`.
2. Click **POST /predict** > **Try it out**, replace the request body with:

   ```json
   {"query": "What share does a daughter receive in Islamic inheritance?", "chunking_strategy": "fixed"}
   ```

   Click **Execute**. Point at the **answer**, `sources`, `"provider": "groq"`,
   and `timings`.

**SAY:**

> Requests are validated, so a question that is too short or has an unknown
> field is rejected with 422. The answer comes with its sources and timings.

---

## 3. Docker (1:50-2:30)

**SHOW:** tab 2, the Dockerfile. Scroll slowly and point at:

- `FROM python:3.12-slim` and the **CPU-only PyTorch** install;
- the step that **preloads the embedding and reranking models**;
- `USER app` (it does not run as root);
- `COPY` of only `src`, `gradio_app.py`, and the two BM25 files (`.env` cannot
  get in because `.dockerignore` allows only listed files);
- `HEALTHCHECK`, which calls **`/health` and `/stats`**;
- `CMD` with uvicorn on port 8000, one worker.

**DO (PowerShell):**

```powershell
docker ps
```

Point at **`(healthy)`** in the STATUS column. If it still says
`(health: starting)`, wait 10 seconds and run `docker ps` again; the first
check runs 30 seconds after start.

**SAY:**

> Docker itself runs this health check every 30 seconds; healthy means both
> retrievers loaded and the monitoring endpoint answers.

---

## 4. Monitoring (2:30-3:20)

**DO:** open a new tab: `http://127.0.0.1:8000/stats` and tick
**Pretty-print**. Point at `request_count` and `average_latency_ms`.

**SAY:**

> `/stats` shows how many requests the service has handled since it started,
> successes and failures, and the average latency, overall and per route, plus
> question-level statistics: error rate and latency percentiles.

**DO (PowerShell):** send three questions and one invalid one, with before and
after printed:

```powershell
.venv\Scripts\python.exe scripts\stats_demo.py
```

(On a machine without the project `.venv`, use `python scripts\stats_demo.py`;
it needs only the Python standard library.) Then **refresh the `/stats` tab**.

**SAY (pointing):**

> The request count rose, the average latency changed, and under `queries` we
> see four questions: three answered and one rejected, so the error rate is
> 0.25, with the median and 95th-percentile latency. Monitoring only records the
> route, status, and timing; questions, answers, and personal details are never
> stored, and a monitoring failure can never break a user's request.

**Optional (5 seconds):** `docker logs legal-aid-rag-api --tail 4` shows one
JSON line per request with no question text.

---

## 5. CI (3:20-4:15)

**SHOW:** tab 3, Actions.

1. Click the latest run on **14-m2-submission** and point at the three green
   jobs: **Ruff lint**, **Pytest**, **Docker build**.

   > Every push runs lint, the full test suite, and a Docker build. On `main`
   > the Docker build is required, and it passes.

2. Go back and click the **"CI demo: deliberately failing test"** run on
   `ci-demo` (red). Click **Pytest** to show the failing
   `test_ci_demo_turns_the_run_red`.

   > Here a deliberately failing test turns the run red.

3. **DO (PowerShell)**, push the fix on camera:

   ```powershell
   git rm tests\test_ci_demo.py
   git commit -m "CI demo: remove the failing test"
   git push karnage4 ci-demo
   ```

   Refresh the Actions tab: a new run **"CI demo: remove the failing test"**
   appears **queued / in progress**.

   > The push started a new run automatically. We won't wait for it on camera.

---

## 6. Deployment (4:15-5:10)

**SHOW:** tab 4, the live site. It opens the **Miras** interface.

**SAY:**

> The same image runs on Azure Container Apps: one virtual CPU, two
> gigabytes, scaling to zero when idle. The keys are Azure secrets. We first
> tried Render's free plan, but its 512 MB limit was below our measured memory,
> so we moved to Azure.

**Optional (10 seconds):** click the example question **"Can a minor be an
heir?"**, wait 2 seconds, click **Ask Miras**, and show the answer.

**DO (Git Bash):** the cURL examples from the README's **Monitoring** section:

```bash
URL=https://legal-aid-rag-api.whitesand-d7432c32.swedencentral.azurecontainerapps.io
curl "$URL/health"
curl -X POST "$URL/predict" -H "Content-Type: application/json" -d '{"query":"Who inherits property when a person dies without a will?","chunking_strategy":"fixed"}'
curl "$URL/stats"
```

(Without Git Bash, use the PowerShell `Invoke-RestMethod` lines in the same
README section; plain `curl` in Windows PowerShell breaks the JSON quoting.)

**SAY (pointing at the `/stats` output):**

> Health is ok, the prediction came back from the live service, and live
> `/stats` counts that question.

---

## 7. Focus and next steps (5:10-6:00, required)

**SHOW:** the README **Project Focus** section and the **Post-M1 problems and
improvements** table. If recording on the machine that holds the team's MLflow
data, show the MLflow UI (`http://127.0.0.1:5000`) with the eight-profile
comparison instead.

**SAY:**

> For our focus area, MLflow and LLM-as-a-Judge evaluation, we compared eight
> retrieval profiles and a prompt, model, and temperature ablation with
> versioned judges. That gave us two operating choices: `fixed-hybrid-top5` as
> the quality champion and `fixed-semantic-top3` as the faster challenger.
>
> Next steps for the final presentation: first, grow the three-question
> evaluation set into a larger held-out set, and add representative real
> questions. Second, turn the judges into an automatic evaluation gate in CI,
> so a retrieval or prompt change that lowers fact coverage or groundedness
> cannot be deployed. Third, use the champion and challenger results to choose
> the deployed profile by measured quality and latency. Thank you.

Stop the recording.

---

## After recording

1. Stop the keep-alive window (**Ctrl+C**) and the local container:

   ```powershell
   docker rm -f legal-aid-rag-api
   ```

2. Remove the CI demo branch locally and on GitHub:

   ```powershell
   git switch 14-m2-submission
   git branch -D ci-demo
   git push karnage4 --delete ci-demo
   ```

3. Check the video length (4-6 minutes, never above 7) and that no key,
   `.env` value, or secret is visible anywhere.

## Final recording checklist

- [ ] Introduction covers the project focus, overview, and tech stack.
- [ ] The GitHub repo is shown and the README `docker build` and `docker run`
      commands are run on camera.
- [ ] Local `/docs`, `/health`, and `/predict` are called.
- [ ] The Dockerfile is shown and `docker ps` shows `(healthy)`.
- [ ] Local `/stats` shows the request count and average latency rising after
      `/predict` calls.
- [ ] Actions shows lint, test, and build green; a push visibly starts a run;
      the failing test's red run is shown.
- [ ] The live URL is opened and live `/health`, `/predict`, and `/stats` are
      called with the README cURL examples.
- [ ] The last 0.5-1 minute covers next steps for the focus area.
- [ ] Length is 4-6 minutes (never above 7) and it shows a real run, not just
      slides.
- [ ] No credential, secret, or `.env` value appears on screen.

Instructor's submission checklist: the README has the live URL and cURL
examples, and `CONTRIBUTION.md` and `AI_USAGE.md` are updated.
