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
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[gui,llm,pdf]"
COPY frontend/server ./frontend/server

EXPOSE 8000
CMD ["python", "-m", "frontend.server", "--host", "0.0.0.0", "--port", "8000"]
