# ---------- Stage 1: builder ----------
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy metadata first (cache optimization)
COPY pyproject.toml requirements.txt /app/

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip wheel setuptools

# Install API deps
RUN pip install --no-cache-dir -r requirements.txt

# Force CPU-only torch + torchaudio (no CUDA wheels)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install your package + its deps from pyproject.toml
RUN pip install --no-cache-dir .
RUN pip check

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/app/hf-model-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Non-root user 
RUN useradd -m -u 10001 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY api_server.py /app/api_server.py
COPY qwen_tts /app/qwen_tts
COPY pyproject.toml /app/pyproject.toml

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
CMD []
