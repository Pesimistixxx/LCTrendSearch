# Trainer image: forms the sample, trains CatBoost, HGT and their stack,
# writes models, report and scores. No server, no Neo4j access needed:
# it reads the exported history and subgraphs from /app/artifacts.
#
#   docker build -f docker/training/Dockerfile -t lctrend-trainer .
#   docker run --rm -v "$PWD/artifacts:/app/artifacts" lctrend-trainer \
#       --run 2026-09-29 --rebuild
#   docker save lctrend-trainer | gzip > lctrend-trainer.tar.gz
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# CatBoost needs OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels: the models are small, CUDA libraries would only add gigabytes.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# Dependencies first, from pyproject alone, so editing src/ does not
# reinstall them.
COPY pyproject.toml ./
RUN mkdir -p src/lctrend && touch src/lctrend/__init__.py \
    && pip install ".[models]" \
    && pip uninstall -y lctrend && rm -rf src
COPY src ./src
RUN pip install --no-deps .
COPY scripts ./scripts

ENTRYPOINT ["python", "scripts/train_model.py"]
CMD []
