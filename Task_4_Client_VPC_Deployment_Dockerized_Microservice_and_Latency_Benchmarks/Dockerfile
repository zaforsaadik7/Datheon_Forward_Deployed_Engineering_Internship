# syntax=docker/dockerfile:1
# ==============================================================================
# Stage 1: Builder — Isolated virtualenv & optional air-gapped model pre-warming
# For air-gapped Client VPC registries, pin the base image by immutable digest:
#   FROM python:3.12-slim@sha256:<immutable-digest-from-client-harbor-or-ecr> AS builder
# ==============================================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-docker.txt requirements-models.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-docker.txt

ARG PRELOAD_HF_MODELS=false
RUN mkdir -p /opt/models && \
    if [ "$PRELOAD_HF_MODELS" = "true" ]; then \
        pip install -r requirements-models.txt && \
        HF_HOME=/opt/models SENTENCE_TRANSFORMERS_HOME=/opt/models \
        python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; \
    fi

# ==============================================================================
# Stage 2: Hardened Non-Root Runtime (UID/GID 10001)
# For air-gapped Client VPC registries, pin the runtime stage by digest:
#   FROM python:3.12-slim@sha256:<immutable-digest-from-client-harbor-or-ecr> AS runtime
# ==============================================================================
FROM python:3.12-slim AS runtime

RUN groupadd -g 10001 fdegroup && \
    useradd -u 10001 -g fdegroup -d /app -s /sbin/nologin fdeuser && \
    mkdir -p /app /app/var /opt/models && \
    chown -R 10001:10001 /app /opt/models

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FDE_ENV=production \
    FDE_VAR_DIR=/app/var \
    FDE_AIRGAPPED_MODE=true \
    FDE_PLANNER=rule \
    FDE_AUTO_SEED=true \
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv
COPY --from=builder --chown=10001:10001 /opt/models /opt/models
COPY --chown=10001:10001 src ./src
COPY --chown=10001:10001 scripts ./scripts
COPY --chown=10001:10001 data ./data

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

CMD ["uvicorn", "src.vpc_service.app:create_vpc_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "15"]

