# Branch 9 Handout: Post-Milestone-1 Fixes

## Purpose

This file is the restart and rebuild guide for branch `9-post-m1-fixes`. The branch responds to two instructor observations after the first Milestone 1 submission:

1. end-to-end RAG requests sometimes took 20–30 seconds or longer; and
2. the original correctness evaluator returned mostly `0/3`, even when an answer contained some of the approved expected facts.

This branch is the stable post-M1 evaluation baseline from which Milestone 2 begins. It does not contain the later Docker, CI, or deployment work.

## Repository and Branch

- Repository: `https://github.com/karnage4/MLOps-for-Legal-Aid-RAG-pipeline.git`
- Working directory: `Legal-Aid-RAG-MLflow`
- Branch: `9-post-m1-fixes`
- Branch tip when this handout was written: `acc24ac`
- Parent milestone branch: `8-m1-submission`
- Never commit `.env`, `mlflow.db`, `mlruns/`, `.venv/`, or raw corpus files.
- `Legal-Aid-RAG-main/` is the read-only legacy reference project and must not be edited.

## What Changed

### Generation path

- Removed the retired Hugging Face model/provider from the request path.
- Groq is now the only generation provider, so every request no longer performs a guaranteed-to-fail Hugging Face attempt first.
- The default answer model is `qwen/qwen3.8-27b`.
- The judge model used during evaluation is `openai/gpt-oss-20b`.
- Groq retry/failure information and latency are logged for diagnosis.

### Evaluation correctness

- Retained strict correctness for compatibility.
- Added fractional expected-fact coverage so partially correct answers are no longer treated as identical to completely wrong answers.
- Expected facts are evaluated independently and the score is the fraction supported by the generated answer.
- MLflow traces preserve the parent request span with nested retrieval and generation work.

### Latency diagnosis

- Added separate retrieval, generation, retry, and total timing fields.
- Added controlled repeat benchmarks to distinguish genuine configuration cost from temporary provider/network stalls.
- Confirmed that hybrid retrieval and reranking are the principal local latency cost; sporadic 50–100 second generation measurements were transient retry/network behavior.

## Core Eight-Configuration Matrix

All combinations below were evaluated: two chunking strategies, two retrieval strategies, and two `top_k` values.

| ID | Chunking | Retrieval | top_k | Status |
|---:|---|---|---:|---|
| 1 | Recursive | Semantic | 3 | Completed |
| 2 | Recursive | Semantic | 5 | Completed |
| 3 | Recursive | Hybrid + reranking | 3 | Completed |
| 4 | Recursive | Hybrid + reranking | 5 | Completed |
| 5 | Fixed | Semantic | 3 | Completed |
| 6 | Fixed | Semantic | 5 | Completed |
| 7 | Fixed | Hybrid + reranking | 3 | Completed |
| 8 | Fixed | Hybrid + reranking | 5 | Completed |

The canonical comparison is `artifacts/genai_evaluation_comparison_core_v2.json`. Individual core-v2 result files are under `artifacts/`.

## Champion and Challenger

### Quality champion

- Configuration: `fixed-hybrid-top5`
- Expected-fact coverage: `0.7222`
- Reason: best observed coverage in the completed core-v2 matrix.

### Speed challenger

- Configuration: `fixed-semantic-top3`
- Expected-fact coverage: `0.5833`
- Earlier core-v2 measured latency: about `2492 ms`.
- Reason: substantially faster retrieval while preserving useful answer coverage.

### Controlled warm benchmark

Both configurations were benchmarked with 15 questions and 5 repetitions per question.

| Configuration | MLflow run | Mean | Median | p95 |
|---|---|---:|---:|---:|
| `fixed-hybrid-top5` | `ae428d443faa4487a2c01659b2923281` | 6975 ms | 6754 ms | 12211 ms |
| `fixed-semantic-top3` | `c672c620f2b242a9a7a3baf6d4614b1a` | 528 ms | 394 ms | 1341 ms |

The challenger was approximately 92.4% faster by mean and 94.2% faster by median, with a fact-coverage reduction of `0.1389`. This makes the choice explicit: use the champion when answer coverage is the priority and the challenger when interactive latency is the priority.

A manually observed Gradio request using fixed/hybrid/top-5 took `10699 ms`: `9171 ms` retrieval plus `1528 ms` generation, with five chunks, three sources, Groq generation, and zero retries.

## Prompt, Model, Parameter, and Temperature Evaluation

The four inference ablations required by the instructor were rerun under the repaired evaluation pipeline. The controlled rerun made 24 successful calls with no retries or failures.

- MLflow run: `0c4c66a00d1c41b2add2d4ba6791d7b9`
- Baseline prompt: `strict_grounded_v1`
- Alternative prompt: `structured_legal_v2`
- Baseline model: `qwen/qwen3.8-27b`
- Model challenger: Groq-hosted GPT-OSS 120B configuration
- Temperature comparison: low/default versus `0.7`

| Ablation | Median generation | Median total |
|---|---:|---:|
| Baseline, low temperature | 759 ms | 5330 ms |
| Structured prompt | 1377 ms | 5746 ms |
| GPT-OSS 120B model | 1590 ms | 6091 ms |
| Temperature 0.7 | 937 ms | 5370 ms |

Earlier averages of roughly 52.7–106.6 seconds were not reproduced; they were provider/network stalls rather than stable configuration costs. See `artifacts/inference_ablation/core_v2/` and `artifacts/inference_latency_benchmark_core_v2.json`.

## Important Files

- `src/`: retrieval, generation, evaluation, API, and MLflow integration.
- `gradio_app.py`: local manual interface.
- `artifacts/genai_evaluation_comparison_core_v2.json`: canonical 8-config comparison.
- `artifacts/inference_ablation/core_v2/comparison.json`: prompt/model/temperature comparison.
- `artifacts/latency_benchmark_fixed-hybrid-top5.json`: champion repeat benchmark.
- `artifacts/latency_benchmark_fixed-semantic-top3.json`: challenger repeat benchmark.
- `MLOPS_LOG.md`: chronological implementation and manual-testing record.
- `README.md`: user-facing setup, MLflow, API, Gradio, and evaluation instructions.

## Required Local Inputs

Create `.env` locally; never commit it:

```dotenv
PINECONE_API_KEY=...
GROQ_API_KEY=...
```

The prepared recursive and fixed BM25 files are required for complete local retrieval. At this branch they may be local/ignored artifacts rather than versioned Git files. If absent, use the repository's preparation commands and the corpus source documented in `README.md`. Pinecone must contain the matching namespaces and embeddings.

## Rebuild and Verify

From the `Legal-Aid-RAG-MLflow` directory in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m pytest -q
```

Start MLflow with the same local tracking database used by the project:

```powershell
.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Start the FastAPI service in another terminal:

```powershell
.venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` and test `/health` and `/predict`. Start the manual interface with:

```powershell
.venv\Scripts\python.exe gradio_app.py
```

Do not rerun paid or quota-consuming evaluation commands casually. First confirm `.env`, Pinecone contents, the prepared BM25 files, and the intended MLflow tracking URI.

## Known Constraints

- Results depend on live Pinecone and Groq services, network quality, and provider quota.
- The MLflow SQLite database and traces are local and intentionally not versioned; JSON summaries in `artifacts/` are the portable evidence.
- Strict correctness can still be zero when any required condition is missed; use expected-fact coverage for partial correctness analysis.
- Hybrid reranking improves coverage but adds material latency.
- No Docker or CI/CD work belongs to this branch. Continue from branch 10 for containerization.

## Git Recovery

```powershell
git fetch origin
git switch 9-post-m1-fixes
git pull --ff-only origin 9-post-m1-fixes
git status
```

Before any new work, confirm the active branch, keep secrets untracked, run tests, and read the newest entries in `MLOPS_LOG.md`.
