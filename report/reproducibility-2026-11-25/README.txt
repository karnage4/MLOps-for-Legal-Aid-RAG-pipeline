Laptop reproduction test — 25 November 2026 (Asia/Karachi)

Source revision: 5f6953b68faf54d041eba579befce0c441d76985.
The starting Git worktree was clean. This test used the existing local clone,
Python 3.14 virtual environment, corpus, cached public models, and cloud index.
The Dockerfile uses Python 3.12, as does the repository's CI workflow.

Required resources

PINECONE_API_KEY: access to legal-inheritance-rag, dimension 384, cosine metric,
with 3,593 vectors in the default namespace and 883 in the fixed namespace.
GROQ_API_KEY: access and generation quota for qwen/qwen3.8-27b.
Both keys were already populated in .env and successfully validated.
Enter keys locally; never put them in the report, Git, or chat.

The complete 50-file source corpus must be placed in corpus/ with its original
five subfolders. The README links its shared Google Drive distribution.
The public models are sentence-transformers/all-MiniLM-L6-v2 and
cross-encoder/ms-marco-MiniLM-L-6-v2. No Hugging Face token was needed for these
public models. Tokenizer data downloads are also needed on first preprocessing.
Use Python, the project requirements, writable local MLflow storage, and internet
access to Pinecone and Groq. Docker Desktop with its Linux engine is needed for
the container path. Azure, Docker Hub publishing, and cloud deployment credentials
are unnecessary for local reproduction.

Verified results

Preprocessing: 50 input PDFs, 46 accepted, four skipped for fewer than 50
extractable characters. Fixed chunking produced 883 chunks and recursive
chunking produced 3,593. All four regenerated chunk/BM25 JSON files were byte
identical to the supplied files. Original runtime artifacts were preserved.

Embeddings: all 4,476 vectors were recomputed locally. Their IDs and all 384
float32 components matched the provided embedding caches exactly.
Pinecone: every vector ID, metadata record, and embedding matched the local
artifacts using atol=1e-6 and rtol=1e-5. Maximum component difference was below
7.5e-9. The shared cloud index received zero writes. The installed Windows
Pinecone 10.3 SDK's fetch returned no records for IDs containing spaces; direct
requests with explicit %20 encoding returned the expected records. The audit
helper uses this verified encoding. Runtime retrieval uses query, not fetch.

Dependencies: pip check passed. Missing Ruff, ReportLab, and pdfplumber
dependencies were installed. requirements-resolved.txt records the resulting
Windows Python 3.14 environment; it is a platform-specific snapshot.

Untouched test baseline: 160 passed, one failed. The failure globally mocked
time.sleep through retrieval.time and captured unrelated MLflow background-thread
calls. The test-only fix substitutes the retrieval module's clock and preserves
the exact retry count and backoff assertions. No production behavior was changed.
After this fix: 161 tests passed and Ruff passed.

Local application integration: /health, /docs, /ui/, and /metrics returned 200.
Three real predictions succeeded for fixed-semantic-top3, fixed-hybrid-top5,
and recursive-hybrid-top5. Each returned source references and a complete
MLflow chain/retriever/generation trace in a separate reproduction database.
These were integration smoke checks, not a quality or latency benchmark.

/stats returned 404. The README already identifies this compatibility route as
pending; /metrics works. This test did not implement the pending route.

The three-question LLM judge evaluation was skipped at the user's explicit
request after automatic approval review blocked the proposed transfer of
retrieved passages and generated answers to Groq for judging. Historical quality
scores, the eight-profile matrix, and benchmark distributions were not rerun.

Docker: a fresh Dockerfile build succeeded with Python 3.12.16, CPU-only PyTorch,
the pinned API dependencies, and both public models downloaded into the image.
Image ID: sha256:3a0487ff72ee462b9df0a3893086aa5d3db6267c0529fdd719ce1c562c96b59e.
Image size: 814,368,381 bytes. The Compose container ran as user app, used about
501 MiB at idle under its 2 GiB limit, and passed pip check. /health, /docs,
/ui/, and /metrics returned 200; an empty /predict input returned 422; /stats
returned the known 404. Docker's healthcheck status is recorded in the evidence.

The initial plain docker run failed authentication because this laptop's .env
uses quoted key values. Docker's --env-file retains those quotes, as the README
already documents. Compose parsed the same .env correctly; no key was changed.
The successful service is left running as legal-aid-rag-api on localhost:8000.
The failed, exited diagnostic container is named legal-aid-rag-repro-5f6953b.
An additional real container prediction was blocked by automatic approval review
over its provider data transfer. Only readiness, routes, and input validation
were exercised inside the container; real predictions were tested in native
Python above. No attempt was made to bypass that restriction.

Rerun commands from the repository root (PowerShell)

  .\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
  .\.venv\Scripts\python.exe -m pip check
  .\.venv\Scripts\python.exe -m ruff check src tests gradio_app.py --no-cache
  .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
  .\.venv\Scripts\python.exe scripts/reproduce_local.py inventory
  $env:TIKTOKEN_CACHE_DIR = Join-Path (Get-Location) '.cache/tiktoken'
  .\.venv\Scripts\python.exe scripts/reproduce_local.py preprocess
  $env:HF_HUB_OFFLINE = '1'
  $env:TRANSFORMERS_OFFLINE = '1'
  .\.venv\Scripts\python.exe scripts/reproduce_local.py embeddings
  .\.venv\Scripts\python.exe scripts/reproduce_local.py trace
  .\.venv\Scripts\python.exe scripts/reproduce_local.py services
  .\.venv\Scripts\python.exe scripts/reproduce_local.py index_audit
  .\.venv\Scripts\python.exe scripts/reproduce_local.py web

The preprocessing helper rebuilds into a separate prepared/ directory, compares
the result with output/, and preserves the original files. The index audit makes
read-only requests. The web stage makes three real Groq generation calls using
the repository's sample question; it consumes provider quota. No judge stage is
included. Raw logs, regenerated files, vectors, and SQLite traces are stored in
output/reproducibility-2026-11-25/. Compact evidence is copied into this report.

For a fresh checkout, create the virtual environment first, obtain the corpus,
and fill .env using .env.example without overwriting existing credentials.
Public models must be downloaded before setting the offline environment flags.

Local Docker commands

  docker desktop start
  docker compose up --build --detach
  docker compose ps
  .\.venv\Scripts\python.exe scripts/check_local_container.py
  docker compose down

The container checker verifies health, the UI, API docs, monitoring, rejection
of an empty prediction request, and the known /stats 404. It makes no generation
or judge calls. Compose supports the quoted values already present in .env.

Reproducibility limits

The existing Pinecone index was reused and fully checked; constructing and
uploading a fresh independent index was not exercised. The index name is
hardcoded in the project. Embedding regeneration itself was exercised fully.
Local model-cache revisions and source-file hashes are recorded in inventory.json
and corpus-manifest.json. Model revisions and transitive dependencies are not
fully pinned by the repository, so future downloads can differ. Generated
answers can also vary between hosted-model calls even at low temperature.
No cloud deployment, CI push, shared-index mutation, or publication was performed.

MLflow interfaces started locally

Project store: http://127.0.0.1:5000, experiment legal-aid-rag (7 runs, 26 traces).
Reproduction store: http://127.0.0.1:5001, experiment
laptop-reproducibility-2026-11-25 (1 run, 4 traces). The reproduction Traces tab
includes the three real prediction traces plus the credential-free smoke trace.
Both UI pages, health endpoints, experiment reads, and run searches returned 200.
Verification details are in mlflow-ui.json. These servers listen on loopback.
The application Docker container still writes to its separate internal database;
new Docker predictions do not automatically appear in either local store.

Restart these interfaces from the repository root in separate terminals after
stopping any existing server on the same port:

  .\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
  .\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///output/reproducibility-2026-11-25/mlflow.db --host 127.0.0.1 --port 5001 --workers 1
