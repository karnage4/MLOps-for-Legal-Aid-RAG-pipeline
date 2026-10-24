# Legal Inheritance RAG Architecture

This repository contains the source code, offline evaluation scripts, and deployment resources for a specialized Retrieval-Augmented Generation (RAG) pipeline. The architecture is engineered to query complex legal inheritance laws using a Hybrid Retrieval framework, subsequently validating generated responses through an automated LLM-as-a-Judge system.

## Project Structure

The codebase is logically split into modular stages corresponding to the data ingestion pipeline and final deployment.

- `src/preprocess.py`: Extracts and recursively chunks raw legal documents from PDF formats.
- `src/prepare_bm25.py`: Builds and validates strategy-specific BM25 artifacts from existing fixed or recursive chunks without contacting Pinecone.
- `src/embed_upsert.py`: Processes continuous text chunks into dense mathematical vectors locally using Sentence Transformers and upserts them to a remote Pinecone Serverless matrix.
- `src/retrieval.py`: Houses the mathematical logic for querying. It executes parallel sparse (BM25) and dense (Pinecone) queries, merging their outputs systematically via Reciprocal Rank Fusion (RRF). 
- `src/generation.py`: Sends retrieved context to Hugging Face first and cleanly falls back to the configured Groq Qwen model when Hugging Face is unavailable.
- `src/evaluation.py`: Runs reproducible RAG experiments and measures faithfulness and relevancy with a configurable Groq judge.
- `app.py`: The interactive graphical user interface built entirely natively within Streamlit.

## Setup and Dependencies

To execute the data injection and compilation stages locally, you must provide explicit environmental variables.

1. Clone the repository and navigate into the root directory.
2. Create a `.env` file containing the following access keys:
   - `HF_API_TOKEN=your_huggingface_token`
   - `PINECONE_API_KEY=your_pinecone_key`
   - `GROQ_API_KEY=your_groq_key`

3. Install the fundamental Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Note: The `requirements.txt` includes advanced dependencies natively required for both the Streamlit UI and local offline dense embedding (`sentence-transformers`, `pinecone[grpc]`, `rank-bm25`). 

## Execution Sequence

The pipeline must be engaged consecutively to ensure structural integrity across the Vector DB and local sparse matrices before opening the application loop.

1. **Preprocessing Phase**:
   Combines and validates raw input documents, then creates fixed and recursive chunk/BM25 artifacts. The existing `bm25_corpus.json` remains the recursive production default.
   ```bash
   python3 src/preprocess.py
   ```

   If the chunk files already exist, regenerate only the fixed BM25 artifact with:

   ```bash
   python -m src.prepare_bm25 --strategy fixed
   ```

   This produces `output/bm25_corpus_fixed.json` and verifies required fields, strategy labels, chunk-index ranges, unique IDs, non-empty text, and exact BM25 ID/text alignment. It does not access or modify Pinecone.

2. **Embedding and Upsert Phase**:
   Transforms and uploads one validated chunking strategy at a time. The
   no-argument/default profile remains recursive and writes to Pinecone's
   default namespace. Fixed chunks use their own cache and the isolated
   `fixed` namespace, so their overlapping IDs cannot replace recursive
   vectors.

   ```bash
   python -m src.embed_upsert --strategy recursive
   python -m src.embed_upsert --strategy fixed
   ```

   The fixed command reads `output/chunks_fixed.json`, caches embeddings in
   ignored `output/embeddings_fixed.json`, validates the 384-dimensional
   cosine index, requires exactly 883 vectors in `fixed`, self-queries three
   representative vectors, and confirms the recursive namespace count did
   not change. It is safe to rerun because Pinecone upserts are idempotent.

3. **Application Interface**:
   Boots the interactive frontend layer locally.
   ```bash
   streamlit run app.py
   ```

### Configurable retrieval profiles

`RetrievalProfile` keeps the three experiment dimensions together and validates
them before a query runs:

| Chunking strategy | BM25 artifact | Pinecone namespace |
| --- | --- | --- |
| `recursive` | `output/bm25_corpus.json` | default (`__default__`) |
| `fixed` | `output/bm25_corpus_fixed.json` | `fixed` |

Retrieval mode is either `semantic` (Pinecone dense ranking only) or `hybrid`
(BM25 + Pinecone + RRF + cross-encoder reranking), and `top_k` must be at least
one. With no arguments, the application continues to use
`recursive` + `hybrid` + top-5.

```python
from src.pipeline import run_rag_query
from src.retrieval import RetrievalProfile, build_retriever

profile = RetrievalProfile("fixed", "semantic", 3)
components = build_retriever(profile.chunking_strategy)
result = run_rag_query(
    "Who inherits property when there is no will?",
    components,
    temperature=0.2,
    profile=profile,
)
```

The profile verifies that its chunking strategy matches both the loaded BM25
artifact and Pinecone namespace. The selected strategy, mode, namespace, and
top-K are also recorded on the nested MLflow trace.

### Fixed-profile retrieval smoke test

Run all four fixed retrieval combinations without calling a generation model
or evaluation judge:

```bash
python -m src.retrieval_smoke_test
```

The command builds the fixed BM25 index once, queries only Pinecone namespace
`fixed`, requires exact top-3/top-5 result counts and fixed-strategy metadata,
and reads each MLflow trace back to confirm that `hybrid_legal_retrieval` is a
child of `fixed_retrieval_smoke`. Its detailed local report is written to the
ignored file `output/fixed_retrieval_smoke.json`.

Verified on 2026-10-24:

| Profile | BM25 candidates | Semantic candidates | Returned | Latency | MLflow trace |
| --- | ---: | ---: | ---: | ---: | --- |
| Fixed semantic top-3 | 0 | 20 | 3 | 8,917.83 ms | `tr-16bab80442459f2254969c872c1a390c` |
| Fixed semantic top-5 | 0 | 20 | 5 | 590.65 ms | `tr-884b12d2036c6a7cf570125ef276dd6d` |
| Fixed hybrid top-3 | 20 | 20 | 3 | 11,540.70 ms | `tr-94b7ab775813e06c3e57a2a011297335` |
| Fixed hybrid top-5 | 20 | 20 | 5 | 10,989.34 ms | `tr-477c416ffed26dff909042d956c8735a` |

## Evaluation and Validation

`src/evaluation.py` defines the complete eight-configuration Cartesian matrix
over the same held-out query set. It aggregates faithfulness, relevancy,
success rate, and latency, logs every configuration to MLflow, and selects the
best complete run by mean faithfulness, then mean relevancy, then lower total
latency.

| Chunking | Retrieval | Top K | Pinecone namespace |
| --- | --- | ---: | --- |
| Recursive | Semantic | 3 | default (`__default__`) |
| Recursive | Semantic | 5 | default (`__default__`) |
| Recursive | Hybrid + reranking | 3 | default (`__default__`) |
| Recursive | Hybrid + reranking | 5 | default (`__default__`) |
| Fixed | Semantic | 3 | `fixed` |
| Fixed | Semantic | 5 | `fixed` |
| Fixed | Hybrid + reranking | 3 | `fixed` |
| Fixed | Hybrid + reranking | 5 | `fixed` |

The two BM25 indexes remain separate, while the Pinecone client, embedding
model, and reranker are shared in memory. Step 5 configures and tests this
matrix; the real eight-run execution and comparison are reserved for Step 6.

Every evaluation query is executed through the same traced `run_rag_query` boundary as the application. MLflow therefore stores one `legal_aid_rag_query` parent trace containing the `hybrid_legal_retrieval` and `legal_generation` child spans, rather than separate root traces for retrieval and generation.

The Groq judge model is configured through `GROQ_JUDGE_MODEL` and defaults to the currently verified `openai/gpt-oss-20b`, keeping judge quota separate from the Qwen generation fallback. `EVALUATION_QUERY_LIMIT` controls the common evaluation-set size and defaults to three queries; `EVALUATION_COOLDOWN_SECONDS` defaults to five seconds between queries.

### Historical three-run baseline (verified 2026-10-23)

These three recursive-only configurations evaluated the same three held-out
questions with no failed queries. The current
[`artifacts/rag_evaluation_summary.json`](artifacts/rag_evaluation_summary.json)
retains this verified baseline until Step 6 successfully completes and writes
the new eight-run result.

| Configuration | Retrieval | Top K | Mean faithfulness | Mean relevancy | Mean total latency | MLflow run ID |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | Semantic only | 5 | 0.9667 | 0.7435 | 13,778.3 ms | `fad0d83034144ec6b225b75d5317ebaa` |
| B | Hybrid + reranking | 3 | 0.9000 | 0.7481 | 7,893.3 ms | `15fd491ef5d54a16af4794a1067bc100` |
| C | Hybrid + reranking | 5 | **0.9762** | **0.7831** | 8,226.3 ms | `f9787d3e95a646baa707770c2754a17a` |

Configuration C is the selected best run because it has the highest mean faithfulness; it also has the highest mean relevancy. The deterministic selection rule compares mean faithfulness first, then mean relevancy, then lower mean total latency. MLflow tags this run with `evaluation.best_run=true`.

### Training exemption for this RAG project

This project does not perform task-specific supervised model training. It uses pretrained embedding, reranking, and generation models, so the traditional training requirement is exempted. Its reproducible modeling workflow consists of corpus preparation, embedding/index construction, retrieval configuration, held-out RAG evaluation, experiment tracking, and clearly named evaluation artifacts.

## Deployment Notes

The repository retains its Docker configuration, but Hugging Face Space deployment is intentionally deferred until the milestone's local pipeline, API, and tests are complete.

## MLflow Integration — Phase 1

This working copy keeps the original retrieval architecture and required environment variable names unchanged. Hugging Face remains the primary generation provider; if both Hugging Face model attempts fail, generation falls back to Groq using the existing `GROQ_API_KEY`. The optional `GROQ_MODEL` variable selects the fallback model and currently defaults to `qwen/qwen3.8-27b`; transient Groq failures are retried up to `GROQ_MAX_ATTEMPTS` (default: 3). If no MLflow environment variables are supplied, runs and traces are stored locally in `mlflow.db` under this project folder.

Testing status, evidence, and the handoff rules for later features are maintained in [`MLOPS_LOG.md`](MLOPS_LOG.md). Update that file immediately after each manual test and before opening the next feature branch.

The integration records:

- a root `legal_aid_rag_query` CHAIN span for each Streamlit question;
- a nested `hybrid_legal_retrieval` RETRIEVER span with the actual retrieved chunks;
- a nested `legal_generation` LLM span recording the selected provider, model, latency, fallback status, and sanitized attempt results without recording either provider token;
- a `real-rag-smoke-test` run recording the Test 3 result, provider/model tags, retrieval and generation latency, fallback usage, and trace ID;
- one conventional MLflow run per offline evaluation configuration, including faithfulness, relevancy, and latency metrics.

### Manual test 1: MLflow only — no cloud keys needed

From this directory, install the unchanged RAG environment plus MLflow and run the smoke test:

```bash
python -m pip install -r requirements.txt
python -m src.mlflow_smoke_test
```

A successful run prints JSON containing `"status": "ok"`, a run ID, a trace ID, and these spans:

```text
smoke_rag_chain
smoke_retriever
smoke_llm
```

This proves both standard experiment tracking and GenAI trace storage before Pinecone or Hugging Face are involved.
The first launch can take longer while MLflow creates and migrates the local SQLite schema; subsequent launches reuse that database.

### Manual test 2: inspect the MLflow UI

In a separate terminal, from this directory:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://127.0.0.1:5000`, select the `legal-aid-rag` experiment, and inspect both the Runs and Traces tabs.

### Manual test 3: run the original RAG app with tracing

Use the same `.env` values as the original project:

```dotenv
PINECONE_API_KEY=your_pinecone_key
HF_API_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_key
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_MAX_ATTEMPTS=3
GROQ_JUDGE_MODEL=openai/gpt-oss-20b
GROQ_JUDGE_REASONING_EFFORT=low
EVALUATION_QUERY_LIMIT=3
EVALUATION_COOLDOWN_SECONDS=5
```

Keep the original `output/bm25_corpus.json` and Pinecone index available, then start:

```bash
streamlit run app.py
```

Or run the repeatable Test 3 check directly:

```bash
python -m src.mlflow_rag_smoke_test
```

Submit one question and refresh the MLflow Traces tab. The query should appear as one chain containing retrieval and generation spans. Hugging Face is attempted first and Groq is used only after Hugging Face fails. Retrieval content, provider/model selection, latency, and the generated response are stored for debugging; secret values are not logged.

### Manual test 4: evaluation-run logging

After both BM25 artifacts, both Pinecone namespaces, and provider keys are
available:

```bash
python -m src.evaluation
```

The command logs all eight configurations shown above. A passing suite reports
`3/3` successful queries for each run, writes
`artifacts/rag_evaluation_summary.json`, tags the selected run with
`evaluation.best_run=true`, and creates one nested three-span trace per
evaluated query.

### Optional shared tracking server

No additional variable is required for local testing. To send data to a shared MLflow server later, set the standard `MLFLOW_TRACKING_URI` or `MLFLOW_EXPERIMENT_ID` variables before starting the app; the original three RAG credential names remain unchanged.
