# =============================================================================
# Macro Causal Engine — Multi-stage Production Dockerfile
# =============================================================================
#
# Stage 1 │ builder  — installs all Python dependencies into an isolated venv.
# Stage 2 │ runtime  — copies the venv and application source into a minimal
#                       image; runs as a non-root user.
#
# Design decisions
# ────────────────
# • python:3.11-slim is used for both stages (Debian Bookworm base).
#   The slim variant omits test suites, manpages, and locale data, saving
#   ~100 MB over the full image without sacrificing build toolchain access.
#
# • PyTorch is installed from the official CPU-only wheel index
#   (https://download.pytorch.org/whl/cpu).  The CUDA variant would add
#   ~2.3 GB to the image; the FastAPI serving layer performs no GPU compute.
#
# • Ray is intentionally excluded from the serving image.  It is only
#   required by distributed_batch.py, which runs as a separate cluster job.
#   Use the full requirements.txt on worker nodes.
#
# • A dedicated non-root user (uid 1001) is created at runtime to follow
#   the principle of least privilege and satisfy most container security
#   scanning policies (CIS Benchmark, Dockerfile best practices).
#
# • The /app/data volume mount point is created with correct ownership so
#   read-only DAG JSON files can be bind-mounted at runtime without
#   permission errors.
#
# Build:    docker build -t macro-causal-engine:latest .
# Run:      docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... macro-causal-engine:latest
# =============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

LABEL stage="builder"

WORKDIR /app

# Install OS-level build dependencies required to compile certain Python
# packages (e.g. uvicorn's C extensions, tokenizers Rust wheel).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment so the runtime stage receives a
# single, self-contained directory with no system-package entanglement.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside the venv first to ensure modern dependency resolver.
RUN pip install --no-cache-dir --upgrade pip

# ── PyTorch (CPU-only) ───────────────────────────────────────────────────────
# Installing torch before the rest of requirements-serve.txt ensures pip
# finds it already satisfied and does not attempt to pull the default
# (CUDA-enabled) wheel from PyPI.
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Remaining serving dependencies ──────────────────────────────────────────
# Copy only the serving requirements file at this step so Docker's layer
# cache is only invalidated when dependencies change, not on source edits.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="macro-causal-engine" \
      description="FastAPI serving layer for the Macro Causal Engine" \
      version="0.1.0"

# ── Non-root user ────────────────────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
    && useradd \
        --uid 1001 \
        --gid appgroup \
        --shell /bin/bash \
        --create-home \
        appuser

WORKDIR /app

# ── Copy venv from builder ───────────────────────────────────────────────────
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Copy application source ──────────────────────────────────────────────────
# Only the files the serving layer needs — distributed_batch.py and the raw
# venv are deliberately excluded.
COPY extractor.py sequencifier.py main.py ./

# ── Data volume mount point ───────────────────────────────────────────────────
# Pre-create /app/data and transfer ownership so the non-root user can read
# bind-mounted JSON files without a runtime chown.
RUN mkdir -p /app/data \
    && chown -R appuser:appgroup /app

USER appuser

# ── Health check ─────────────────────────────────────────────────────────────
# Docker marks the container unhealthy if /health returns non-200 three
# times in a row; orchestrators (ECS, Kubernetes) use this for rolling
# deployments and automatic restarts.
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=20s \
    --retries=3 \
    CMD python -c \
        "import urllib.request, sys; \
         r = urllib.request.urlopen('http://localhost:8000/health', timeout=8); \
         sys.exit(0 if r.status == 200 else 1)"

EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Single worker: scale horizontally behind an ALB/Nginx rather than using
# multiple uvicorn workers inside the container (avoids shared-memory
# contention and simplifies per-container observability).
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
