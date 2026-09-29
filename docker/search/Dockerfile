# Light search API: GET /api/search over the shared Neo4j, read only.
# No ingestion, no PDF models, no torch. NEO4J_* is required; with a
# GigaChat key the query is also matched by meaning, without it by BM25.
#
#   docker compose -f compose.search.yaml up -d --build
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# Dependencies first, from pyproject alone, so editing src/ does not
# reinstall them.
COPY pyproject.toml ./
RUN mkdir -p src/lctrend && touch src/lctrend/__init__.py \
    && pip install ".[gui]" \
    && pip uninstall -y lctrend && rm -rf src
COPY src ./src
RUN pip install --no-deps .
COPY frontend/server/__init__.py frontend/server/search_api.py \
    ./frontend/server/
# GigaChat endpoints are signed by the Russian Trusted Root CA (Минцифры).
COPY certs ./certs
ENV GIGACHAT_CA_BUNDLE_FILE=/app/certs/gigachat-ca-bundle.pem

RUN useradd --system --uid 10001 search
USER search
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["python", "-m", "frontend.server.search_api", "--host", "0.0.0.0", "--port", "8000"]
