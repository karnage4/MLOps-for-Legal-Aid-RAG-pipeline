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
| 1 | MLflow integration | `1-mlflow-integration` | Complete | 1 of 4 feature tests passed | In testing |
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
- Added offline evaluation-run logging for configuration parameters, faithfulness, relevancy, and latency.
- Updated the Streamlit application to use the traced pipeline.
- Ignored local MLflow state (`mlflow.db`, its SQLite journal files, and `mlruns/`) in Git.

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
| 2026-10-22 | MLflow UI and tracking API | Partial | Local UI returned HTTP 200; its API returned experiment `legal-aid-rag` and the finished smoke run with `integration_ok = 1.0`. Browser inspection is awaiting team confirmation. |
| 2026-10-22 | Full Pinecone + Hugging Face traced query | Pending | Requires the original corpus artifacts, Pinecone index, and keys. |
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
| 2026-10-22 22:00 PKT | Codex (automated) | Passed | Pending team confirmation | UI returned HTTP 200. Server API returned experiment ID `1`, finished run `3b2ce216871d42278f1c2542ff5c71d4`, and `integration_ok = 1.0`. |

### Test D — run a real RAG query

Required before this test:

- a `.env` file containing valid `PINECONE_API_KEY`, `HF_API_TOKEN`, and `GROQ_API_KEY` values;
- the populated Pinecone index named `legal-inheritance-rag`;
- the same embedding model used for that index (`all-MiniLM-L6-v2`);
- `output/bm25_corpus.json` produced by `python src/preprocess.py`;
- network access to Pinecone and Hugging Face.

Start the app:

```bash
streamlit run app.py
```

Submit one legal question, then inspect the newest MLflow trace.

Pass conditions:

- the Streamlit response still works as it did in the original project;
- the trace root is `legal_aid_rag_query`;
- `hybrid_legal_retrieval` contains the retrieved chunks and scores;
- `huggingface_legal_generation` contains the prompt inputs and generated result;
- no API token appears in span inputs, outputs, attributes, or tags.

Record the result:

| Date/time | Tester | Query | Result | Trace ID | Retrieval mode | Model used | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

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
