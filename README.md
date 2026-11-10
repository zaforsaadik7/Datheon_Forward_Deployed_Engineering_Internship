# Datheon — Forward Deployed Engineering Internship

This repository contains the engineering deliverables for the **Datheon Forward Deployed Engineering (FDE) Internship**.

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
└── Task_2_Enterprise_Multi_Tenant_RAG_with_RBAC/
    ├── README.md                     # Architecture, 6-persona RBAC matrix, evaluation metrics & quickstart
    ├── requirements.txt              # Python dependencies
    ├── pytest.ini                    # Pytest configuration
    ├── data/
    │   └── sample_documents/         # Multi-format enterprise corpus + manifest.yaml
    ├── reports/
    │   └── rag_eval.json             # 19-case multi-persona RBAC & ground-truth evaluation report
    ├── scripts/
    │   └── make_sample_data.py       # Deterministic sample corpus generator
    ├── src/
    │   ├── config.py                 # RAG, ETL & ChromaDB configuration
    │   ├── api/                      # Authenticated FastAPI service (/api/v1/rag/query, /personas)
    │   ├── etl/                      # Multi-format extractors, recursive chunker, tenant-aware vector store
    │   └── rag/                      # RBAC, query rewriter, ONNX/BM25 reranker, synthesizer, guardrails, CLI
    └── tests/                        # Automated unit, integration, API, CLI & real neural model tests
```

---

## Tasks

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
