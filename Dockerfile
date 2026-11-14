FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Keep dependency installation in a separate cacheable layer.
COPY requirements-api.txt ./requirements-api.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-api.txt

# Run the public API as an unprivileged user. The application directory stays
# writable because local MLflow uses /app/mlflow.db when no remote URI is set.
RUN addgroup --system app \
    && adduser --system --ingroup app --home /app app \
    && chown app:app /app

COPY --chown=app:app src ./src
COPY --chown=app:app output/bm25_corpus.json ./output/bm25_corpus.json
RUN test -s /app/output/bm25_corpus.json

USER app

EXPOSE 8000

# Render supplies PORT; local Docker uses the image default of 8000. Model
# loading can take time on a cold start, so the healthcheck has a grace period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=4).read()"

CMD ["sh", "-c", "exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
