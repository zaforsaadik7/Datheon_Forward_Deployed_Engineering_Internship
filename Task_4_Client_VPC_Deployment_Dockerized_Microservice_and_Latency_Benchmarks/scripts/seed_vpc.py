"""Idempotent VPC data bootstrap for ChromaDB knowledge store and per-tenant SQLite ERP/CRM databases."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from scripts.make_sample_data import generate_sample_corpus
from src.config import BASE_DIR
from src.erp_crm.database import ERPCRMDatabase
from src.etl.pipeline import ingest_directory
from src.etl.vector_store import VectorStore
from src.rag.cli import DeterministicHashEmbedding

SUPPORTED_TENANTS = ("acme_corp", "globex_inc")


def ensure_vpc_seeded(
    *,
    var_dir: Path,
    env: str = "development",
    auto_seed: bool = True,
    allow_sample_seed: bool = False,
    force_reset: bool = False,
) -> Dict[str, Any]:
    """
    Idempotently seed the VPC persistent volume (`var_dir`) on startup:
    - Refuses to auto-seed sample data in `production` unless `allow_sample_seed=True` or `force_reset=True`.
    - If ChromaDB collection `enterprise_docs` is empty (or `force_reset=True`), generates/ingests the
      8 sample documents into `var_dir / "chroma_db"`.
    - If `var_dir / "acme_corp_erp.db"` has 0 inventory items (or `force_reset=True`), seeds the
      ERP/CRM tables (including `SKU-BENCH-NODE` with 100,000 units) for each tenant.
    - On subsequent restarts, preserves existing records without overwriting.
    """
    var_path = Path(var_dir)
    var_path.mkdir(parents=True, exist_ok=True)

    is_prod = env.strip().lower() == "production"
    can_seed = force_reset or (auto_seed and (not is_prod or allow_sample_seed))

    chroma_dir = var_path / "chroma_db"
    sample_docs_dir = BASE_DIR / "data" / "sample_documents"
    seed_json_path = BASE_DIR / "data" / "seed_enterprise_data.json"

    seeded_now = False
    chroma_chunks = 0
    erp_skus = 0

    if not can_seed:
        # Inspect existing state without seeding
        if chroma_dir.exists():
            store = VectorStore(
                path=chroma_dir,
                collection_name="enterprise_docs",
                embedding_function=DeterministicHashEmbedding(),
            )
            chroma_chunks = store.count()
        acme_db_file = var_path / "acme_corp_erp.db"
        if acme_db_file.exists():
            erp_skus = len(ERPCRMDatabase(acme_db_file).search_inventory())
        return {
            "seeded_now": False,
            "chroma_chunks": chroma_chunks,
            "erp_skus": erp_skus,
        }

    # 1. Ensure ChromaDB knowledge base is populated
    store = VectorStore(
        path=chroma_dir,
        collection_name="enterprise_docs",
        embedding_function=DeterministicHashEmbedding(),
    )
    if force_reset or store.count() == 0:
        if not (sample_docs_dir / "manifest.yaml").exists():
            generate_sample_corpus(sample_docs_dir)
        ingest_directory(sample_docs_dir, store)
        seeded_now = True
    chroma_chunks = store.count()

    # 2. Ensure per-tenant ERP/CRM SQLite databases are populated
    seed_payload = json.loads(seed_json_path.read_text(encoding="utf-8"))
    for tenant_id in SUPPORTED_TENANTS:
        db_file = var_path / f"{tenant_id}_erp.db"
        erp_db = ERPCRMDatabase(db_file)
        existing_items = erp_db.search_inventory()
        if force_reset or len(existing_items) == 0:
            erp_db.reset_and_seed(seed_payload)
            seeded_now = True
        if tenant_id == "acme_corp":
            erp_skus = len(erp_db.search_inventory())

    return {
        "seeded_now": seeded_now,
        "chroma_chunks": chroma_chunks,
        "erp_skus": erp_skus,
    }

