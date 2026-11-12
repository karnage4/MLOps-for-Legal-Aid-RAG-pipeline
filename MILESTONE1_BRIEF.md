# Legal-Aid RAG MLOps — Milestone 1 Brief

## Project outcome

The notebook-era Legal-Aid RAG prototype is now a reproducible, MLflow-tracked pipeline with validated data preparation, fixed and recursive chunking, semantic and hybrid retrieval, direct Groq generation, a FastAPI prediction service, a local Gradio interface, and automated tests.

Training is exempt because the system uses pretrained embedding, reranking, and generation models. The modeling evidence therefore comes from held-out inference evaluation rather than supervised training cycles.

## Architecture

```text
Legal PDF corpus
    ↓
Preparation and validation
    ↓
Fixed / recursive chunks → BM25 + Pinecone indexes
    ↓
Semantic / hybrid retrieval → reranking
    ↓
Direct Groq generation with bounded retries
    ↓
MLflow evaluation and traces → FastAPI /predict + Gradio UI
```

The verified corpus run accepted 46 of 50 PDFs and produced 3,593 recursive chunks plus 883 fixed chunks. Raw corpus access is documented in the README.

## Milestone requirement status

| Requirement | Status | Evidence |
| --- | --- | --- |
| README, architecture, quick start, dataset note | Complete | Reproducible setup, commands, corpus link, and RAG training exemption |
| Separate preparation and validation | Complete | Strategy-aware preparation validates fields, ranges, duplicates, and artifact alignment |
| Model evaluation and saved artifacts | Complete under RAG exemption | Versioned chunk, BM25, and evaluation artifacts replace an unrelated training artifact |
| At least three tracked experiments | Complete | Eight retrieval runs, eight managed GenAI runs, and four controlled generation runs |
| MLflow GenAI integration | Complete | Managed dataset, expected facts, six versioned judges, nested traces, parameters, metrics, and run IDs |
| FastAPI service | Complete | Validated `/predict`, `/health`, and generated `/docs` |
| Unit tests | Complete | Preparation, loading, retrieval, direct generation and retries, tracing, evaluation, API, and Gradio coverage |
| Contribution document | Complete | Names, ERP IDs, task division, and detailed deliverables |

## Retrieval configuration analysis

All eight combinations used the same three held-out questions and completed 24 of 24 queries successfully.

| Configuration | Faithfulness | Relevancy | Avg. latency | Legacy result |
| --- | ---: | ---: | ---: | --- |
| Recursive semantic top-3 | 0.8889 | 0.7717 | 17,208 ms | Strong semantic baseline |
| Recursive semantic top-5 | 0.7556 | 0.4809 | 6,851 ms | Fastest, but weakest relevancy |
| Recursive hybrid top-3 | 0.9667 | **0.7860** | 11,799 ms | Highest relevancy |
| Recursive hybrid top-5 | 0.7778 | 0.7723 | 9,375 ms | Balanced challenger |
| Fixed semantic top-3 | **1.0000** | 0.7228 | 10,590 ms | Faithfulness co-leader |
| Fixed semantic top-5 | 0.7963 | 0.7501 | 16,014 ms | Mid-table |
| **Fixed hybrid top-3** | **1.0000** | 0.7700 | 17,530 ms | **Legacy champion** |
| Fixed hybrid top-5 | 0.6667 | 0.7402 | 15,933 ms | Legacy challenger |

The legacy evaluator selects **fixed-hybrid-top3** because faithfulness has first priority. It ties at 1.0000 and beats fixed-semantic-top3 on relevancy, 0.7700 versus 0.7228. Recursive-hybrid-top3 remains the main challenger when relevancy and latency matter more.

## Managed GenAI champion and challengers

The expanded MLflow comparison applies six pinned judges and reviewed expected facts. Its ranking gives correctness and retrieval sufficiency priority over the legacy faithfulness score.

| Profile | Correctness | Sufficiency | Legal quality | Groundedness | Retrieval relevance | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Fixed hybrid top-5** | **0.333** | **0.333** | **1.000** | 0.667 | 0.667 | 29,517 ms |
| Fixed semantic top-5 | 0.000 | **0.333** | 0.667 | **1.000** | 0.467 | **3,808 ms** |
| Fixed hybrid top-3 | 0.000 | **0.333** | 0.333 | **1.000** | 0.778 | 13,512 ms |
| Recursive hybrid top-3 | 0.000 | 0.000 | **1.000** | **1.000** | **0.889** | 9,424 ms |

**Managed GenAI champion:** fixed-hybrid-top5. It is the only profile with non-zero correctness and also reaches 1.000 legal-answer quality. Its cost is the highest latency and only moderate groundedness.

**Primary challengers:** recursive-hybrid-top3 for retrieval relevance and speed, and fixed-semantic-top5 for low latency with non-zero sufficiency. This difference from the legacy champion is expected because the two evaluators optimize different rubrics.

## Prompt, model, and temperature analysis

Retrieval was frozen to fixed-hybrid-top5 so each run changed one generation variable.

| Run | Variable | Correctness | Legal quality | Retrieval relevance | Sufficiency | Avg. latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline low temperature | Control | 0.000 | 0.667 | 0.667 | 0.667 | 28,756 ms |
| **Structured prompt** | Prompt | **0.333** | **1.000** | **0.733** | 0.667 | 21,272 ms |
| GPT-OSS-120B | Model | 0.000 | 0.333 | 0.600 | 0.667 | 20,145 ms |
| High temperature | Temperature 0.7 | 0.000 | 0.667 | 0.600 | 0.333 | **16,698 ms** |

The **structured prompt is the generation champion**. Compared with the baseline, it improves correctness by 0.333, legal quality by 0.333, retrieval relevance by 0.067, and reduces mean latency by 7,484 ms. GPT-OSS-120B is faster but lowers legal quality. Higher temperature is fastest but halves retrieval sufficiency, from 0.667 to 0.333.

## Serving, observability, and testing

- FastAPI exposes `/predict`, `/health`, and `/docs` with Pydantic request validation and cached component loading.
- Gradio exposes chunking strategy, retrieval mode, top-K, sources, and generation metadata for local manual testing.
- MLflow stores one root RAG chain with nested retrieval and generation spans, plus the Groq model, retry attempts, latency, parameters, metrics, dataset lineage, and judge versions.
- Groq is the sole generation provider and loads `GROQ_API_KEY` securely from `.env`; the removed Hugging Face model is no longer attempted.
- The verified full repository regression recorded 91 passing tests and one pre-existing MLflow SQLAlchemy deprecation warning. The final focused evaluation run recorded 14 passing tests.

## Final selection and limitation

For the current evidence, use **fixed-hybrid-top5 retrieval with the `structured_legal_v2` prompt, Qwen 3.8-27B, and temperature 0.1**. This choice follows the richer managed judge rubric and the controlled generation ablation.

The evaluation set contains only three human-reviewed questions. The results demonstrate reproducibility and configuration effects for Milestone 1, but they are not statistically generalizable. The next milestone should expand the reviewed dataset, add automated CI evaluation, and move MLflow to shared durable storage.

## Post-submission improvement baseline

The current pipeline adds per-stage retrieval timing and bounds hybrid
candidate processing to 10 BM25, 10 semantic, and 12 reranking candidates. A
45-request warm benchmark measured 3,730 ms mean, 3,451 ms median, and 5,469 ms
p95 retrieval latency for fixed-hybrid-top5, compared with the historical
26,287 ms mean. A new MLflow `fact_coverage` judge scores expected facts
independently, while the existing binary correctness judge is retained and
documented as an all-facts check. Historical champion statistics above remain
unchanged until all eight profiles are rerun with the new seven-judge lineage.

