# Datheon — Forward Deployed Engineering Internship

This repository contains the engineering deliverables for the **Datheon Forward Deployed Engineering (FDE) Internship**.

## Repository Structure

```text
Datheon_Forward_Deployed_Engineering_Internship/
├── README.md
└── Task_1_Client_Legacy_Data_Ingestion_and_Vector_ETL_Pipeline/
    ├── README.md                     # Detailed Task 1 documentation & quickstart
    ├── README_TASK1.md               # Operational summary
    ├── requirements.txt              # Python dependencies
    ├── pytest.ini                    # Pytest configuration
    ├── data/
    │   └── sample_documents/         # Multi-format sample corpus (PDF, CSV, MD, JSON, TXT) + manifest.yaml
    ├── reports/
    │   └── benchmark.json            # Synthetic corpus ingestion & RBAC query latency benchmark report
    ├── src/
    │   ├── config.py                 # Pipeline & ChromaDB cosine configuration
    │   └── etl/                      # Extractors, recursive chunker, manifest resolver, vector store, CLI
    └── tests/                        # 98 automated unit, integration, and real-embedding tests
```

---

## Task 1: Client Legacy Data Ingestion & Vector ETL Pipeline

Located in [`Task_1_Client_Legacy_Data_Ingestion_and_Vector_ETL_Pipeline/`](./Task_1_Client_Legacy_Data_Ingestion_and_Vector_ETL_Pipeline/README.md).

### Overview
An enterprise ETL & vector indexing pipeline that ingests heterogeneous legacy client documents (**PDF, CSV, Markdown, JSON, and TXT**), preserves fine-grained citation locators (`page_number`, spreadsheet `row_index`, `section_title`), enforces fail-closed access-control metadata (`clearance_tier` 1–4 and `department`), and indexes chunks into **ChromaDB** (`hnsw:space="cosine"`) with deterministic SHA-256 chunk IDs for idempotent re-ingestion.

### Quickstart

```bash
cd Task_1_Client_Legacy_Data_Ingestion_and_Vector_ETL_Pipeline
pip install -r requirements.txt

# Run unit & integration test suite
python -m pytest -q

# Ingest sample documents into ChromaDB
python -m src.etl.cli ingest

# Query with clearance tier & department filtering
python -m src.etl.cli query "VPC peering routes" --tier 2 --dept engineering

# Run multi-scale ingestion & latency benchmark
python -m src.etl.cli benchmark --scales 50,200,1000
```

See [`Task_1_Client_Legacy_Data_Ingestion_and_Vector_ETL_Pipeline/README.md`](./Task_1_Client_Legacy_Data_Ingestion_and_Vector_ETL_Pipeline/README.md) for full architecture details.
