# Task 1: Client Legacy Data Ingestion & Vector ETL Pipeline

Extracts PDF, CSV, Markdown, JSON and TXT files, chunks them with citation metadata
(page, row, section), and indexes them in ChromaDB with cosine similarity. Queries are
filtered by clearance tier and department before ranking.

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q                                   # unit + integration tests (fake embedder)
python -m src.etl.cli ingest                          # indexes data/sample_documents
python -m src.etl.cli query "VPC peering routes" --tier 2 --dept engineering
python -m src.etl.cli benchmark --scales 50,200,1000  # writes reports/benchmark.json
python -m src.etl.cli reset                           # drop and recreate the collection
python -m pytest -m real_embeddings -v                # known-answer retrieval, needs the model
```

`ingest` exits 1 if any document failed; the failures are listed with file, stage and error.
`--embedder fake` uses a deterministic hash embedder (no model download) for smoke tests.

## Access-control metadata

Each document's department, clearance tier (1 Public, 2 Internal, 3 Confidential, 4 Secret)
and date come from `manifest.yaml` in the data directory, matched on the full relative path.
Files not listed fall back to `<department>_*_tierN.ext`; a missing tier fails closed to 4 and
an unknown department rejects the file. Departments: engineering, finance, legal, hr,
executive, operations.

`VectorStore.query_by_tier_and_department(text, user_tier, user_departments, top_k)`:
returns chunks with `clearance_tier <= user_tier`; `user_departments=None` filters by tier
only, an empty list returns nothing, and a tier outside 1-4 raises `ValueError`.

## Re-ingesting

Chunk ids are `sha256(relative_path|chunk_index)`. Re-ingesting a document deletes its old
chunks first, so edits that shrink a document leave no stale chunks. A document that fails
extraction keeps its previously indexed chunks.

## Benchmark report

`reports/benchmark.json` holds `ingestion` rows (per scale and batch size: stage timings,
docs/s, chunks/s, MB/s, peak RSS) and `queries` rows (P50/P95/P99 ms for four filter
shapes, plus `filter_violations`, which must be 0). Corpora are synthetic and seeded.

## Offline machines

Chroma's default embedder downloads an ONNX copy of all-MiniLM-L6-v2 on first use. On a
connected machine run one ingest, find Chroma's cache folder (typically under
`~/.cache/chroma/onnx_models/`; confirm for the pinned version) and copy it to the same
location on the offline machine.

## Known limits

No OCR (image-only PDF pages are skipped with a warning; an all-image PDF fails), access
control is enforced at the query layer only, and only ChromaDB is implemented (no Qdrant).
