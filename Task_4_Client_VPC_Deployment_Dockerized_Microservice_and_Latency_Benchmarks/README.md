# Task 4: Client VPC Deployment — Dockerized Microservice & Latency Benchmarks

A self-contained, air-gapped-ready **Client VPC AI Microservice Gateway** that unifies the **Multi-Tenant RBAC RAG Engine** and the **Autonomous Field Agent (ReAct + HITL Tool Orchestrator)** behind a hardened FastAPI + ASGI gateway with Docker Compose secret management, an 8-metric Prometheus telemetry catalog, structured JSON correlation logging, and a live-Uvicorn subprocess HTTP concurrency benchmark suite.

---

## Key Capabilities

1. **Unified Client VPC FastAPI Gateway (`src/vpc_service/app.py`)**
   - `GET /healthz`: Liveness probe returning `200 OK` (`{"status": "alive"}`).
   - `GET /readyz`: Readiness probe inspecting SQLite WAL connectivity (`SELECT 1`), ChromaDB collection readiness (`> 0` chunks), and secret bundle validity; returns sanitized component checks (`"ok"` / `"failed"`) without leaking filesystem paths or exception tracebacks.
   - `GET /metrics`: Bearer-token-protected Prometheus exposition endpoint refreshing dynamic dependency (`fde_dependency_up`) and HITL queue (`fde_hitl_pending_approvals`) gauges on every scrape.
   - `POST /api/v1/rag/query`: Multi-tenant RBAC-enforced RAG query endpoint with deterministic citation locators, numeric/entity guardrails, and structured `token_usage` accounting.
   - `POST /api/v1/agent/run`, `GET /api/v1/agent/sessions/{session_id}`, `GET /api/v1/agent/approvals`, `POST /api/v1/agent/approvals/{approval_id}/decide`: Multi-tenant ReAct agent orchestration backed by per-tenant SQLite ERP databases (`acme_corp_erp.db`, `globex_inc_erp.db`) and cryptographic SHA-256 Human-in-the-Loop (HITL) supervisor approvals.

2. **Pure ASGI Hardening Middleware (`src/vpc_service/middleware.py`)**
   - **Streaming + Content-Length Body Size Guard (`413 Payload Too Large`):** Enforces `max_body_bytes` both via `Content-Length` pre-check and chunk-by-chunk streaming byte counting on `receive()`.
   - **Bounded LRU Token-Bucket Rate Limiter (`429 Too Many Requests`):** Keys on `key:<sha256[:16]>` for recognized API keys and falls back to `ip:<client_host>` (`max_buckets=1024`) for unrecognized/unauthenticated callers so rotating random fake `X-API-Key` headers cannot bypass rate limits or exhaust memory.
   - **Route-Template Cardinality Normalization:** Normalizes dynamic path parameters (`/api/v1/agent/sessions/{session_id}`, `/api/v1/agent/approvals/{approval_id}/decide`) and collapses unknown 404 paths to `"UNMATCHED"`.
   - **Correlation ID & JSON Structured Logging (`src/vpc_service/logging_utils.py`):** Propagates `X-Request-ID` and emits single-line JSON access logs with automatic secret redaction.

3. **Hardened Multi-Stage Container & Compose Stack (`Dockerfile`, `docker-compose.yml`)**
   - Multi-stage `python:3.12-slim` build running as non-root `UID 10001:10001` (`fdeuser`) with `read_only: true`, `tmpfs: ["/tmp"]`, `cap_drop: ["ALL"]`, and `security_opt: ["no-new-privileges:true"]`.
   - Supports optional build-time HuggingFace model baking (`--build-arg PRELOAD_HF_MODELS=true`) and enforces offline runtime flags (`FDE_AIRGAPPED_MODE=true`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
   - Internal-only `prometheus` service (`UID 65534:65534`, `expose: ["9090"]` on `vpc_internal_net`) scraping `/metrics` using `/run/secrets/fde_metrics_token` and evaluating 4 alert rules (`FDEHighErrorRate`, `FDEHighP95Latency`, `FDEDependencyDown`, `FDEHitlBacklog`).

4. **Live-Uvicorn Subprocess Concurrency & Latency Benchmark Suite (`src/vpc_service/benchmark.py`)**
   - Spawns a real `uvicorn` HTTP server subprocess on an ephemeral loopback port, waits for `/readyz == 200`, runs warmup requests, and executes all 5 workloads across concurrency levels `c = 1, 5, 10, 20` over real TCP sockets.
   - Verifies live Prometheus counter deltas on `/metrics` and outputs `reports/vpc_latency_benchmark.json` with explicit environment metadata (`execution_mode: "local_uvicorn_subprocess"`, `containerized: false`).

---

## Repository Structure

```text
Task_4_Client_VPC_Deployment_Dockerized_Microservice_and_Latency_Benchmarks/
├── Dockerfile                        # Multi-stage non-root (UID 10001) container build
├── docker-compose.yml                # Hardened fde-vpc-api + internal prometheus stack
├── deploy/
│   ├── prometheus.yml                # Authenticated Bearer-token scrape configuration
│   ├── prometheus_rules.yml          # 4 SLO & operational alert rules
│   └── secrets/                      # Example secret templates (real secrets gitignored)
├── docs/
│   └── CLIENT_INTEGRATION_GUIDE.md   # 8-section Client VPC Operations & Security Runbook
├── scripts/
│   ├── make_sample_data.py           # Deterministic multi-format enterprise document generator
│   ├── seed_vpc.py                   # Idempotent ChromaDB + per-tenant SQLite ERP bootstrap
│   ├── smoke_test.ps1                # End-to-end Docker Compose & promtool smoke test (PowerShell)
│   └── smoke_test.sh                 # End-to-end Docker Compose & promtool smoke test (Bash)
├── src/
│   ├── config.py                     # Unified repository paths & RBAC constants
│   ├── etl/                          # Self-contained multi-format document ETL & ChromaDB store
│   ├── rag/                          # Self-contained Multi-Tenant RBAC RAG engine & guardrails
│   ├── erp_crm/                      # Self-contained WAL-enabled SQLite ERP/CRM backend
│   ├── agent/                        # Self-contained ReAct orchestrator & HITL SessionStore
│   └── vpc_service/                  # Unified VPC gateway, auth, secrets, telemetry, CLI, benchmark
├── reports/
│   └── vpc_latency_benchmark.json    # Generated p50/p95/p99 latency & RPS benchmark report
└── tests/                            # 20 automated unit, integration, config & benchmark tests
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Local Secret Files & Seed VPC Stores
```bash
python -m src.vpc_service.cli init-dev-secrets --out-dir deploy/secrets
python -m src.vpc_service.cli check-secrets --env production \
  --api-keys-file deploy/secrets/fde_api_keys.json \
  --metrics-token-file deploy/secrets/fde_metrics_token.txt
python -m src.vpc_service.cli seed --var-dir var
```

### 3. Verify Docker & Prometheus Configurations
```bash
python -m src.vpc_service.cli verify-config
```

### 4. Run the Live-Uvicorn Concurrency Benchmark Suite
```bash
python -m src.vpc_service.cli benchmark --out reports/vpc_latency_benchmark.json
```

### 5. Run the Full Automated Test Suite
```bash
python -m pytest -v
```

### 6. Run the Hardened Docker Compose Stack (on any Docker Host)
```powershell
# Windows PowerShell
.\scripts\smoke_test.ps1

# Linux / macOS Bash
bash scripts/smoke_test.sh
```

See [docs/CLIENT_INTEGRATION_GUIDE.md](docs/CLIENT_INTEGRATION_GUIDE.md) for the complete 8-section Client VPC integration, secret rotation, telemetry, and operations runbook.
