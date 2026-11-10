# Task 1: Client Legacy Data Ingestion & Vector ETL Pipeline

Multi-format enterprise document ingestion and vector indexing pipeline built for Forward Deployed Engineering (FDE) deployments. Extracts legacy documents across **PDF, CSV, Markdown, JSON, and TXT**, chunks them with fine-grained citation locators (page numbers, spreadsheet row IDs, and section headers), and indexes them into **ChromaDB** (`hnsw:space="cosine"`) with fail-closed RBAC metadata (`clearance_tier` 1–4 and `department`).

---

## Folder Structure

```text
Task_1_Client_Legacy_Data_Ingestion_and_Vector_ETL_Pipeline/
├── README.md                         # Project documentation & quickstart guide
├── README_TASK1.md                   # Concise operational notes
├── requirements.txt                  # Pinned Python dependencies
├── pytest.ini                        # Pytest configuration & markers
├── data/
│   └── sample_documents/             # Multi-format sample corpus & manifest.yaml
│       ├── manifest.yaml
│       ├── security_compliance_tier1.pdf
│       ├── merger_acquisition_audit_tier4.pdf
│       ├── finance_q3_budget_audit_tier3.csv
│       ├── hr_compensation_bands_tier2.csv
│       ├── engineering_cloud_vpc_runbook_tier2.md
│       ├── executive_ai_roadmap_tier3.md
│       ├── operations_supplier_catalog_tier1.json
│       └── legal_ip_license_agreement_tier4.txt
├── reports/
│   └── benchmark.json                # Ingestion throughput & query latency benchmark report
├── src/
│   ├── __init__.py
│   ├── config.py                     # ETL & ChromaDB configuration (cosine space enforced)
│   └── etl/
│       ├── __init__.py
│       ├── models.py                 # Pydantic schemas for DocumentMetadata, ExtractedSegment, DocumentChunk
│       ├── manifest.py               # Manifest loader with fail-closed clearance & filename fallback
│       ├── chunker.py                # Recursive text chunker preserving page/row/section locators
│       ├── vector_store.py           # ChromaDB cosine store, idempotent upsert & RBAC-filtered query
│       ├── pipeline.py               # End-to-end ingestion pipeline with per-document fault isolation
│       ├── synthetic.py              # Seeded synthetic multi-format corpus generator for benchmarking
│       ├── benchmark.py              # Ingestion & filtered-query P50/P95/P99 latency benchmark harness
│       ├── cli.py                    # Typer + Rich CLI (ingest, query, benchmark, reset)
│       └── extractors/
│           ├── __init__.py           # Extractor registry & file extension dispatcher
│           ├── base.py               # Abstract BaseExtractor interface
│           ├── pdf_extractor.py      # PyMuPDF page-aware PDF extractor
│           ├── csv_extractor.py      # CSV row extractor with spreadsheet row numbers (header=1, data>=2)
│           ├── markdown_extractor.py # Heading-hierarchy Markdown extractor
│           ├── json_extractor.py     # Recursive JSON record/key-path extractor
│           └── text_extractor.py     # Encoding-resilient plain-text extractor
└── tests/                            # Comprehensive unit, integration, and real-embedding test suite
    ├── conftest.py
    ├── test_benchmark.py
    ├── test_chunker.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_extractors.py
    ├── test_manifest.py
    ├── test_models.py
    ├── test_pipeline.py
    ├── test_real_embeddings.py
    ├── test_sample_data.py
    └── test_vector_store.py
```

---

## Key Features

- **Multi-Format Extraction with Source Locators**:
  - **PDF (`PyMuPDF`)**: Extracts text per page with 1-indexed `page_number` metadata and skips image-only pages with warnings.
  - **CSV**: Preserves column context on every row and records exact spreadsheet row numbers (`row_index >= 2`).
  - **Markdown**: Tracks nested heading hierarchies (`section_title`) across chunks.
  - **JSON & TXT**: Flattens structured objects/arrays with key paths and handles UTF-8 / UTF-8-BOM / Latin-1 fallbacks.
- **Fail-Closed Access Control Metadata**:
  - Resolves `department`, `clearance_tier` (`1` Public, `2` Internal, `3` Confidential, `4` Secret), and `document_date` from `manifest.yaml` (matched on relative path) with filename convention fallback (`<department>_*_tierN.ext`).
  - Missing clearance tier fails closed to Tier `4` (Secret); unknown department rejects the document.
- **Deterministic & Idempotent Re-Ingestion**:
  - Chunk IDs are derived deterministically via `sha256(relative_path|chunk_index)`.
  - Re-ingesting a modified document deletes its previous chunks only after extraction succeeds, preventing both stale orphan chunks and data loss on failed extractions.
- **Pre-Ranking RBAC Query Filtering**:
  - `VectorStore.query_by_tier_and_department()` pushes `$and` metadata predicates (`clearance_tier <= user_tier` and `department in user_departments`) into ChromaDB prior to cosine ranking.

---

## Quickstart

```bash
cd Task_1_Client_Legacy_Data_Ingestion_and_Vector_ETL_Pipeline
pip install -r requirements.txt

# 1. Run the automated test suite
python -m pytest -q

# 2. Ingest the sample multi-format corpus into ChromaDB
python -m src.etl.cli ingest

# 3. Run an access-controlled semantic query
python -m src.etl.cli query "VPC peering routes" --tier 2 --dept engineering

# 4. Run the synthetic scaling & latency benchmark (writes reports/benchmark.json)
python -m src.etl.cli benchmark --scales 50,200,1000

# 5. Reset the vector collection
python -m src.etl.cli reset

# 6. Run known-answer retrieval tests with real embeddings (all-MiniLM-L6-v2)
python -m pytest -m real_embeddings -v
```
