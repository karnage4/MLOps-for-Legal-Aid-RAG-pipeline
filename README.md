# Legal Inheritance RAG Architecture

This repository contains the source code, offline evaluation scripts, and deployment resources for a specialized Retrieval-Augmented Generation (RAG) pipeline. The architecture is engineered to query complex legal inheritance laws using a Hybrid Retrieval framework, subsequently validating generated responses through an automated LLM-as-a-Judge system.

## Project Structure

The codebase is logically split into modular stages corresponding to the data ingestion pipeline and final deployment.

- `src/preprocess.py`: Extracts and recursively chunks raw legal documents from PDF formats.
- `src/embed_upsert.py`: Processes continuous text chunks into dense mathematical vectors locally using Sentence Transformers and upserts them to a remote Pinecone Serverless matrix.
- `src/retrieval.py`: Houses the mathematical logic for querying. It executes parallel sparse (BM25) and dense (Pinecone) queries, merging their outputs systematically via Reciprocal Rank Fusion (RRF). 
- `src/generation.py`: Dispatches retrieved context arrays securely to the Hugging Face Inference API for strict logical generation using Qwen models.
- `src/evaluation.py`: An autonomous testing suite that verifies Faithfulness and Relevancy algorithmically via the unmetered Groq Llama-3.1 API.
- `app.py`: The interactive graphical user interface built entirely natively within Streamlit.

## Setup and Dependencies

To execute the data injection and compilation stages locally, you must provide explicit environmental variables.

1. Clone the repository and navigate into the root directory.
2. Create a `.env` file containing the following access keys:
   - `HF_TOKEN=your_huggingface_token`
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
   Combiles raw input structures.
   ```bash
   python3 src/preprocess.py
   ```

2. **Embedding and Upsert Phase**:
   Transforms and uploads processed textual boundaries.
   ```bash
   python3 src/embed_upsert.py
   ```

3. **Application Interface**:
   Boots the interactive frontend layer locally.
   ```bash
   streamlit run app.py
   ```

## Evaluation and Validation

The implementation was systematically benchmarked using `src/evaluation.py`. The evaluation isolated varying configurations (fixed vs. recursive chunking, semantic-only vs. hybrid retrieval) verifying their latency properties and synthetic metrics natively. 

The evaluation framework automatically aggregates these properties into statistical tables mapping Faithfulness (claims supported by source documents) and Relevancy (cosine stability against generated inverse questions) specifically using the Llama-3.1-8B model natively on Groq processors.

## Deployment Notes

The resultant interface is designed to migrate beyond local boundaries and is actively configured sequentially for Hugging Face Spaces. Using the provided `Dockerfile`, the Streamlit application containerizes efficiently, circumventing rigid SDK requirements. The interface automatically pulls variables continuously, operating directly under the mapped 7860 network port.

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

After the original corpus, Pinecone index, and provider keys are available:

```bash
python src/evaluation.py
```

Each evaluated chunking/retrieval configuration is logged as a separate MLflow run with its parameters and aggregate faithfulness, relevancy, and latency metrics.

### Optional shared tracking server

No additional variable is required for local testing. To send data to a shared MLflow server later, set the standard `MLFLOW_TRACKING_URI` or `MLFLOW_EXPERIMENT_ID` variables before starting the app; the original three RAG credential names remain unchanged.
