# Milestone 1 Contribution Document

## Team Members

| Member | Student name | ERP ID |
| --- | --- | --- |
| Person 1 | Karam Hussain | 27857 |
| Person 2 | Aadesh Panjwani | 29303 |
| Person 3 | Muhammad Rayyan Khan | 28410 |

## Milestone 1 Task Division

| Member | Responsibility area | Deliverables |
| --- | --- | --- |
| Karam Hussain | Modeling and evaluation | Retrieval experiments; held-out evaluation dataset and metrics; MLflow experiment tracking and nested tracing; eight retrieval configurations; controlled prompt, model, and temperature comparisons; management of GenAI judge lineage and evaluation-completeness checks; comparison of quality, relevance, grounding, and latency results; champion selection and justification; documentation of experiment findings and final technical integration |
| Aadesh Panjwani | Pipeline and API | Data preparation and validation; fixed and recursive chunk artifacts; BM25 and Pinecone indexing support; semantic and hybrid retrieval implementation; FastAPI `/health`, `/predict`, and `/docs`; generation provider, model, fallback status, latency, and evaluation-result logging; pipeline and architecture documentation |
| Muhammad Rayyan Khan | Quality and operations | Unit and integration tests; manual execution of focused and full pytest regression suites; recording and reviewing test results and warnings; verification of MLflow GenAI traces, prompts, judges, and artifacts; manual FastAPI and Gradio validation; reproducibility and final quality assurance |

## Detailed Effort Breakdown

### Karam Hussain — Modeling and evaluation

- Designed the MLOps evaluation approach around the existing RAG architecture.
- Added MLflow experiment tracking and nested query, retrieval, and generation
  traces.
- Ran the eight retrieval configurations covering fixed and recursive chunks,
  semantic and hybrid retrieval, and top-K values of 3 and 5.
- Managed the held-out evaluation dataset, metrics, GenAI judges, and
  prompt/model/temperature inference comparisons.
- Selected and justified the best retrieval and generation configurations.
- Preserved Hugging Face as the primary provider and integrated the Groq
  fallback without changing the retrieval architecture.
- Coordinated experiment lineage, artifacts, branch integration, and final
  technical review.

### Aadesh Panjwani — Pipeline and API

- Implemented and validated separate data-preparation steps.
- Prepared fixed and recursive chunking artifacts and BM25 corpora.
- Supported embeddings and isolated Pinecone namespace setup.
- Implemented the semantic and hybrid retrieval paths used by the evaluation
  configurations.
- Implemented request validation, cached component loading, `/health`, and
  `/predict` in the FastAPI service.
- Logged generation provider, model, fallback status, latency, and evaluation
  results for pipeline observability.
- Maintained the generated `/docs`, pipeline documentation, and architecture
  diagram.

### Muhammad Rayyan Khan — Quality and operations

- Added unit tests for data preparation, retriever loading, provider fallback,
  tracing, evaluation logic, FastAPI endpoints, and the Gradio interface.
- Verified nested MLflow GenAI traces, prompt variants, judge results, and
  tracked evaluation metadata produced by the modeling workflow.
- Executed focused and full pytest regression suites, recorded the pass results,
  and reviewed failures or warnings before milestone sign-off.
- Reviewed tracked evaluation artifacts, run IDs, and test evidence for
  completeness and consistency before final documentation.
- Validated FastAPI `/predict`, generated `/docs`, and the local Gradio
  interface.
- Performed reproducibility checks, regression testing, secret-safety checks,
  and final quality assurance.


