# Datheon — Forward Deployed Engineering Internship

This repository contains all four engineering deliverables for the **Datheon Forward Deployed Engineering (FDE) Internship**.

## Repository Structure

```text
Datheon_Forward_Deployed_Engineering_Internship/
├── README.md
├── Task_1_Client_Legacy_Data_Ingestion_and_Vector_ETL_Pipeline/
│   ├── README.md                     # Detailed Task 1 documentation & quickstart
│   ├── README_TASK1.md               # Operational summary
│   ├── requirements.txt              # Python dependencies
│   ├── pytest.ini                    # Pytest configuration
│   ├── data/
│   │   └── sample_documents/         # Multi-format sample corpus (PDF, CSV, MD, JSON, TXT) + manifest.yaml
│   ├── reports/
│   │   └── benchmark.json            # Synthetic corpus ingestion & RBAC query latency benchmark report
│   ├── src/
│   │   ├── config.py                 # Pipeline & ChromaDB cosine configuration
│   │   └── etl/                      # Extractors, recursive chunker, manifest resolver, vector store, CLI
│   └── tests/                        # 98 automated unit, integration, and real-embedding tests
├── Task_2_Enterprise_Multi_Tenant_RAG_with_RBAC/
│   ├── README.md                     # Architecture, 6-persona RBAC matrix, evaluation metrics & quickstart
│   ├── requirements.txt              # Python dependencies
│   ├── pytest.ini                    # Pytest configuration
│   ├── data/
│   │   └── sample_documents/         # Multi-format enterprise corpus + manifest.yaml
│   ├── reports/
│   │   └── rag_eval.json             # 19-case multi-persona RBAC & ground-truth evaluation report
│   ├── scripts/
│   │   └── make_sample_data.py       # Deterministic sample corpus generator
│   ├── src/
│   │   ├── config.py                 # RAG, ETL & ChromaDB configuration
│   │   ├── api/                      # Authenticated FastAPI service (/api/v1/rag/query, /personas)
│   │   ├── etl/                      # Multi-format extractors, recursive chunker, tenant-aware vector store
│   │   └── rag/                      # RBAC, query rewriter, ONNX/BM25 reranker, synthesizer, guardrails, CLI
│   └── tests/                        # Automated unit, integration, API, CLI & real neural model tests
├── Task_3_Autonomous_Field_Agent_ERP_CRM_Tool_Orchestration/
│   ├── README.md                     # Architecture, role matrix, 10-scenario evaluation & quickstart
│   ├── requirements.txt              # Python dependencies
│   ├── pytest.ini                    # Pytest configuration
│   ├── data/
│   │   └── seed_enterprise_data.json # Seed inventory SKUs, CRM leads & ITSM tickets
│   ├── reports/
│   │   └── agent_eval.json           # 10-scenario autonomous agent evaluation report
│   ├── scripts/
│   │   └── seed_db.py                # Deterministic ERP/CRM SQLite database seeder
│   ├── src/
│   │   ├── config.py                 # Agent, ERP/CRM & SQLite configuration
│   │   ├── api/                      # Authenticated FastAPI Agent Service (/api/v1/agent/*)
│   │   ├── erp_crm/                  # Transactional SQLite ERP/CRM/ITSM REST API & fault injection
│   │   └── agent/                    # ReAct orchestrator, planner, tools, HTTP retry, SQLite HITL store, CLI
│   └── tests/                        # Automated unit, integration, HITL, retry & injection tests
└── Task_4_Client_VPC_Deployment_Dockerized_Microservice_and_Latency_Benchmarks/
    ├── README.md                     # VPC Gateway architecture, Prometheus metrics, Docker & benchmark guide
    ├── Dockerfile                    # Multi-stage non-root (UID 10001) container build
    ├── docker-compose.yml            # Hardened fde-vpc-api + internal prometheus stack
    ├── requirements.txt              # Local development & test dependencies
    ├── requirements-docker.txt       # Stripped production runtime container dependencies
    ├── deploy/
    │   ├── prometheus.yml            # Bearer-token authenticated Prometheus scrape config
    │   ├── prometheus_rules.yml      # 4 SLO & operational alert rules
    │   └── secrets/                  # Secret file templates
    ├── docs/
    │   └── CLIENT_INTEGRATION_GUIDE.md # 8-section Client VPC Operations & Security Runbook
    ├── reports/
    │   └── vpc_latency_benchmark.json  # Live-Uvicorn c=1..20 p50/p95/p99 latency & RPS benchmark report
    ├── scripts/                      # Corpus generator, VPC seeder, PowerShell & Bash smoke tests
    ├── src/
    │   ├── etl/                      # Self-contained multi-format document ETL & ChromaDB store
    │   ├── rag/                      # Self-contained Multi-Tenant RBAC RAG engine & guardrails
    │   ├── erp_crm/                  # Self-contained WAL-enabled SQLite ERP/CRM backend
    │   ├── agent/                    # Self-contained ReAct orchestrator & HITL SessionStore
    │   └── vpc_service/              # Unified VPC gateway, ASGI guards, secrets, 8-metric Prometheus, CLI
    └── tests/                        # 28 automated unit, integration, config & concurrency benchmark tests
```

---

## Tasks Overview

### 1. [Task 1: Client Legacy Data Ingestion & Vector ETL Pipeline](./Task_1_Client_Legacy_Data_Ingestion_and_Vector_ETL_Pipeline/README.md)
An enterprise ETL & vector indexing pipeline that ingests heterogeneous legacy client documents (**PDF, CSV, Markdown, JSON, and TXT**), preserves fine-grained citation locators (`page_number`, spreadsheet `row_index`, `section_title`), enforces fail-closed access-control metadata (`clearance_tier` 1–4 and `department`), and indexes chunks into **ChromaDB** (`hnsw:space="cosine"`) with deterministic SHA-256 chunk IDs for idempotent re-ingestion.

```bash
cd Task_1_Client_Legacy_Data_Ingestion_and_Vector_ETL_Pipeline
pip install -r requirements.txt
python -m pytest -q
python -m src.etl.cli ingest
python -m src.etl.cli query "VPC peering routes" --tier 2 --dept engineering
```

---

### 2. [Task 2: Enterprise Multi-Tenant RAG with Role-Based Access Control (RBAC)](./Task_2_Enterprise_Multi_Tenant_RAG_with_RBAC/README.md)
A multi-tenant Retrieval-Augmented Generation (RAG) engine and authenticated FastAPI service built on ChromaDB (cosine distance), ONNX Cross-Encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) / candidate-scoped BM25 reranking, defense-in-depth RBAC pre/post-retrieval enforcement across 6 personas and 2 tenants (`acme_corp` and `globex_inc`), uniform abstention against side-channel leaks, 4-level canonical citation locators, and numeric/entity hallucination guardrails.

```bash
cd Task_2_Enterprise_Multi_Tenant_RAG_with_RBAC
pip install -r requirements.txt
python -m pytest -q
python -m src.rag.cli ingest
python -m src.rag.cli ask --persona general_counsel --query "What is the Project Titan aggregate purchase price and founder buyout?"
python -m src.rag.cli eval --out reports/rag_eval.json
```

---

### 3. [Task 3: Autonomous Field Agent — Enterprise ERP/CRM Tool Orchestration](./Task_3_Autonomous_Field_Agent_ERP_CRM_Tool_Orchestration/README.md)
An autonomous ReAct field agent that orchestrates multi-step workflows across transactional **ERP Inventory**, **CRM Lead Management**, and **ITSM Ticketing** REST APIs with pre-flight Pydantic v2 schema validation self-healing, exponential backoff with `Idempotency-Key` protection, SQLite-persisted Human-in-the-Loop (HITL) supervisor approval gates (`PENDING -> APPROVED -> CONSUMED`), operator session isolation, and `<tool_output_data>` indirect prompt-injection defense.

```bash
cd Task_3_Autonomous_Field_Agent_ERP_CRM_Tool_Orchestration
pip install -r requirements.txt
python -m pytest -v
python -m src.agent.cli seed
python -m src.agent.cli run --goal "Check inventory for SKU-GPU-H100 and inspect CRM lead LEAD-101"
python -m src.agent.cli eval --out reports/agent_eval.json
```

---

### 4. [Task 4: Client VPC Deployment — Dockerized Microservice & Latency Benchmarks](./Task_4_Client_VPC_Deployment_Dockerized_Microservice_and_Latency_Benchmarks/README.md)
An air-gapped-ready **Client VPC AI Microservice Gateway** unifying the Multi-Tenant RBAC RAG Engine and Autonomous ReAct Field Agent behind a hardened FastAPI + ASGI gateway (`413` streaming payload size guard, `429` LRU token-bucket rate limiter, route-template cardinality normalization, JSON correlation logs), multi-stage non-root `Dockerfile` (`UID 10001`), hardened `docker-compose.yml` with file-based secrets and internal Prometheus alerting (`4` alert rules, `8` `fde_*` metrics), an 8-section [`docs/CLIENT_INTEGRATION_GUIDE.md`](./Task_4_Client_VPC_Deployment_Dockerized_Microservice_and_Latency_Benchmarks/docs/CLIENT_INTEGRATION_GUIDE.md), and a live-Uvicorn subprocess HTTP concurrency benchmark suite (`c = 1, 5, 10, 20`).

```bash
cd Task_4_Client_VPC_Deployment_Dockerized_Microservice_and_Latency_Benchmarks
pip install -r requirements.txt
python -m pytest -v
python -m src.vpc_service.cli verify-config
python -m src.vpc_service.cli benchmark --out reports/vpc_latency_benchmark.json
```
