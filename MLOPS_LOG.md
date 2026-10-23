# MLOps Implementation Log

This file is the running record for introducing MLOps features one at a time. Update it after every implementation, manual test, and branch transition.

## Working rules

- `Legal-Aid-RAG-main/` is the read-only baseline and must not be edited.
- Preserve the original RAG stack and variable names unless a later phase explicitly changes them.
- Add one MLOps capability per branch and manually verify it before starting the next capability.
- Never commit `.env`, API tokens, `mlflow.db`, `mlruns/`, downloaded models, or generated corpus output.
- Record real run IDs, trace IDs, results, and failures; do not substitute estimated metrics.

## Phase overview

| Phase | Feature | Branch | Implementation | Manual verification | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | MLflow integration | `1-mlflow-integration` | Complete | 2 of 4 feature tests passed | In testing |
| 2 | To be selected after Phase 1 | Not created | Not started | Not started | Blocked by Phase 1 acceptance |

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
| 2026-10-22 | Original environment variable names | Passed | `.env.example` still contains only the original three RAG variables. |
| 2026-10-22 | Credential-free MLflow smoke test | Passed | MLflow 3.17.1 recorded run `3b2ce216871d42278f1c2542ff5c71d4` and trace `tr-733a80cdd7ad53af15152a1874324bea`; readback confirmed the CHAIN, RETRIEVER, and LLM spans. |
| 2026-10-22 | MLflow UI and tracking API | Passed | Local UI returned HTTP 200; its API returned experiment `legal-aid-rag` and the finished smoke run with `integration_ok = 1.0`. Team browser inspection confirmed the session and evaluation run are visible. |
| 2026-10-23 | Corpus regeneration and validation | Passed | Loaded 46 of 50 supplied files and generated 3,593 unique recursive chunks. `bm25_corpus.json` matches `chunks_recursive.json`, with non-empty texts of 30-600 characters. Four PDFs were skipped because they contained fewer than 50 extractable characters. |
| 2026-10-23 | Python 3.12 environment | Passed | Created an ignored project-local Python 3.12.13 environment; PyTorch 2.14.1 CPU and Sentence Transformers 6.2.0 import successfully. |
| 2026-10-23 | Pinecone embedding and upsert | Passed | Created the approved 384-dimensional cosine index `legal-inheritance-rag`, uploaded all 3,593 vectors, and passed a three-result semantic sanity query. The pre-existing `rag-fixed-chunks` index was not changed. |
| 2026-10-23 | Hugging Face to Groq fallback unit tests | Passed | Four tests verify provider order, no unnecessary Groq call, secret exclusion from results, MLflow provider/model/latency attributes, and compact root-chain output. |
| 2026-10-23 | Full traced query with Groq fallback | Blocked | Hybrid retrieval and both Hugging Face attempts ran, then Groq returned HTTP 401. The value currently stored as `GROQ_API_KEY` is not an accepted `gsk_...` Groq key. Test 3 must be rerun after replacing it. |
| 2026-10-22 | Groq evaluation runs in MLflow | Pending | Requires the complete RAG setup and keys. |

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

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
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
- network access to Pinecone and Hugging Face.

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
| Pending | Pending | Pending | Pending | Pending | Pending |

## Phase 1 definition of done

Phase 1 is complete only when Tests A–E pass, their tables above contain real evidence, and the updated log is committed and pushed to `1-mlflow-integration`. Do not begin Phase 2 until the team accepts those results.

## Phase 2 placeholder

Feature, scope, branch name, and acceptance criteria will be added only after Phase 1 is verified. When Phase 2 begins:

1. update the Phase overview table;
2. add a dated Phase 1 completion entry;
3. create a separate branch for the selected feature;
4. document its test plan before implementing it.

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
