FROM python:3.12-slim

# CPU-only PyTorch. Without this step pip resolves the default Linux torch
# wheel, which drags in several GB of nvidia-*/CUDA wheels that no laptop, CI
# runner, Render, or Cloud Run CPU container can use. The CPU index serves
# both linux/amd64 and linux/arm64 (Apple Silicon) wheels. Keep the version
# equal to what sentence-transformers resolved before this change.
ARG TORCH_VERSION=2.15.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Download the two Hugging Face models at build time so a fresh container
# starts without reaching huggingface.co. Set to 0 only for offline builds;
# the models are then downloaded on first start instead.
ARG PRELOAD_MODELS=1

# GRADIO_ANALYTICS_ENABLED=False stops the interface from sending usage data
# or checking for updates. FORWARDED_ALLOW_IPS lets uvicorn trust the hosting
# platform's HTTPS proxy headers, so the interface builds https:// URLs; the
# container is reachable only through that proxy.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HF_HOME=/opt/hf-cache \
    GRADIO_ANALYTICS_ENABLED=False \
    FORWARDED_ALLOW_IPS=*

WORKDIR /app

# Run the public API as an unprivileged user. The application directory stays
# writable because local MLflow uses /app/mlflow.db when no remote URI is set.
RUN addgroup --system app \
    && adduser --system --ingroup app --home /app app \
    && chown app:app /app \
    && mkdir -p "$HF_HOME" \
    && chown app:app "$HF_HOME"

# Keep dependency installation in a separate cacheable layer.
COPY requirements-api.txt ./requirements-api.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" \
    && python -m pip install -r requirements-api.txt \
    && python -c "import torch; assert not torch.version.cuda, 'CUDA torch installed'"

USER app

# Model names must match EMBEDDING_MODEL and RERANKER_MODEL in
# src/retrieval.py (enforced by tests/test_container_config.py). This layer
# sits before COPY src so code edits do not re-download the models.
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
        python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)"; \
    fi
# With the models baked in, never call huggingface.co at runtime.
ENV HF_HUB_OFFLINE=${PRELOAD_MODELS}

COPY --chown=app:app src ./src
COPY --chown=app:app gradio_app.py ./gradio_app.py
COPY --chown=app:app output/bm25_corpus.json ./output/bm25_corpus.json
COPY --chown=app:app output/bm25_corpus_fixed.json ./output/bm25_corpus_fixed.json
RUN test -s /app/output/bm25_corpus.json \
    && test -s /app/output/bm25_corpus_fixed.json

EXPOSE 8000

# Render/Cloud Run supply PORT; local Docker uses the image default of 8000.
# Startup loads both retrievers before uvicorn answers, hence the grace period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=4).read()"

# src.web serves the API plus the Miras Gradio interface at /ui (/ redirects
# there). The app writes one structured JSON access-log line per request
# (src/monitoring.py), so uvicorn's own access log is turned off.
CMD ["sh", "-c", "exec uvicorn src.web:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log"]
