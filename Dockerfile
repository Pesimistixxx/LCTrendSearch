FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels avoid downloading CUDA libraries for this ingestion service.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Dependencies first, from pyproject alone, with an empty package in place of
# the code: editing src/ no longer reinstalls ~100 packages (audit H-5).
COPY pyproject.toml ./
RUN mkdir -p src/lctrend && touch src/lctrend/__init__.py \
    && pip install ".[gui,llm,pdf,report]" \
    && pip uninstall -y lctrend && rm -rf src
COPY src ./src
RUN pip install --no-deps .
COPY frontend/server ./frontend/server
# GigaChat endpoints are signed by the Russian Trusted Root CA (Минцифры),
# which is not in the default trust store.
COPY certs ./certs

# Runs as root: the named volumes (ingestion ledger, model cache) of existing
# installs are root-owned, and a non-root user could no longer write them.
EXPOSE 8000
CMD ["python", "-m", "frontend.server", "--host", "0.0.0.0", "--port", "8000"]
