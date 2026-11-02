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
| 2 | RAG evaluation experiments | `2-rag-evaluation-experiments` | Complete | All six steps and final eight-run suite passed | Accepted |
| 3 | GenAI datasets and judges | `3-mlflow-genai-datasets-judges` | Managed dataset implemented | SDK and UI verification passed | Active |

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
| 4 | Compare additional profiles with the same dataset and judges | In progress (7/8 valid) | Seven completed profiles are checkpointed; only `fixed-hybrid-top5` remains after partial quota recovery |

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

Verified completed profiles on 2026-11-01:

| Profile | Run ID | Correctness | Retrieval sufficiency | Legal quality pass rate | Retrieval groundedness | Retrieval relevance | Query relevance | Mean total latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed-hybrid-top3` | `9fcc1af514b046078f04bc2202c5f60a` | 0.0000 | 0.3333 | 0.3333 | 1.0000 | 0.7778 | 1.0000 | 13,511.7 ms |
| `recursive-hybrid-top3` | `98e7838d135f43fcb830c1d824e9581a` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.8889 | 1.0000 | 9,424.3 ms |
| `recursive-semantic-top5` | `a130d951e9ce446d9238848c24dd8986` | 0.0000 | 0.0000 | 0.3333 | 1.0000 | 0.4667 | 0.3333 | 4,557.7 ms |
| `recursive-semantic-top3` | `5e585a134b844a0a8e50ae4693cee229` | 0.0000 | 0.0000 | 0.3333 | 0.3333 | 0.4444 | 0.3333 | 3,933.7 ms |
| `recursive-hybrid-top5` | `63fe861d46e740a0893c7d13041152ba` | 0.0000 | 0.0000 | 0.3333 | 1.0000 | 0.8667 | 1.0000 | 7,532.3 ms |
| `fixed-semantic-top3` | `91c3ff85269a47ad8a1131b5e28815dc` | 0.0000 | 0.3333 | 0.6667 | 1.0000 | 0.4444 | 0.6667 | 4,589.7 ms |
| `fixed-semantic-top5` | `68e216b95fc3455bad64fd8f516c2902` | 0.0000 | 0.3333 | 0.6667 | 1.0000 | 0.4667 | 0.6667 | 3,807.7 ms |

Still required: `fixed-hybrid-top5`. Partial attempts are excluded from reuse
and ranking. The original live continuation was stopped after Groq repeatedly
rate-limited the registered `openai/gpt-oss-20b` judges and a scorer exceeded
the bounded timeout; one row took more than 90 minutes and could no longer
produce a complete profile. This is an external judge-quota blocker, not a RAG
retrieval or generation failure. Resume without repeating valid profiles with:

```bash
python -m src.genai_comparison
```

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
Only that profile needs another quota-window retry.

Implementation verification passes all 11 focused comparison and
controlled-evaluation tests. The final full local regression after the
quota-safe resume and cleanup changes passed all 70 tests in 223.25 seconds.

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
- Kept `CONTRIBUTION.md` pending because neither repository contains the team
  names and ERP IDs; the report still has placeholder author fields.

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
