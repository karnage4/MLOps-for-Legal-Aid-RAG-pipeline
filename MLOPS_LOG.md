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
| 1 | MLflow integration | `1-mlflow-integration` | Complete | Smoke, UI, and full RAG trace passed | Accepted |
| 2 | RAG evaluation experiments | `2-rag-evaluation-experiments` | Complete | Three-run suite and MLflow readback passed | Ready for team review |

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

### Planned configurations

| Label | Retrieval mode | Top K | Purpose |
| --- | --- | ---: | --- |
| A | Semantic only | 5 | Dense-retrieval baseline |
| B | Hybrid + reranking | 3 | Test smaller context selection |
| C | Hybrid + reranking | 5 | Current production configuration |

All configurations use the same recursive prepared corpus and Pinecone index so retrieval mode and `top_k` are the only changed variables.

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
- Ran semantic top-5, hybrid/reranked top-3, and hybrid/reranked top-5 against the same three held-out questions.
- Logged configuration parameters, judge settings, faithfulness, relevancy, retrieval/generation/total latency, success counts, and status tags to MLflow.
- Selected and tagged the best successful run using mean faithfulness, mean relevancy, then mean total latency.
- Routed evaluation queries through the application-level `run_rag_query` boundary so each query, retrieval operation, and generated answer form one nested MLflow trace.
- Added the tracked summary artifact `artifacts/rag_evaluation_summary.json` and kept per-query responses in ignored `output/` data.
- Added focused unit coverage for result aggregation, batched verdict parsing, incomplete judge output, deterministic best-run selection, and the three-configuration suite.

### Verified results

| Configuration | Retrieval mode | Top K | Faithfulness | Relevancy | Avg. total latency | Successful | MLflow run ID |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `A-semantic-top5` | Semantic only | 5 | 0.9667 | 0.7435 | 13,778.3 ms | 3/3 | `fad0d83034144ec6b225b75d5317ebaa` |
| `B-hybrid-top3` | Hybrid + reranking | 3 | 0.9000 | 0.7481 | 7,893.3 ms | 3/3 | `15fd491ef5d54a16af4794a1067bc100` |
| `C-hybrid-top5` | Hybrid + reranking | 5 | **0.9762** | **0.7831** | 8,226.3 ms | 3/3 | `f9787d3e95a646baa707770c2754a17a` |

`C-hybrid-top5` is the selected best configuration because it has the highest mean faithfulness and mean relevancy. Direct MLflow readback confirmed all three runs are `FINISHED`, each has `success_rate=1.0` and `evaluation.status=passed`, and only the selected run has `evaluation.best_run=true`.

Two earlier diagnostic attempts were not accepted as milestone evidence: one exposed rate limits from using the generation model for judging, and another exposed GPT-OSS output-budget and Windows console-encoding issues. The final suite separated the judge model, used low reasoning effort, batched verdicts, and made console output Unicode-safe.

Nested-trace correction was verified live on 2026-10-23 with evaluation run `428483309e0148e08bc4eccb19975a9b` and trace `tr-83738135bb07f35ce078e9be0c71ae88`. MLflow readback showed root span `legal_aid_rag_query` (`b2b62a19aa2bd799`) and both `hybrid_legal_retrieval` and `legal_generation` with that root span ID as their `parent_id`. The query completed successfully using the normal Hugging Face-to-Groq fallback and produced faithfulness `1.0`, relevancy `0.7768`, and `success_rate=1.0`.

Final local verification passed on 2026-10-23 after the nesting correction: `12 passed in 116.52s` across `test_evaluation_experiments.py`, `test_generation_fallback.py`, and `test_pipeline_tracing.py`. The pre-push audit also passed `git diff --check`, confirmed `.env`, `mlflow.db`, and detailed `output/` results are ignored, and found no credential-shaped values outside excluded local files.

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
