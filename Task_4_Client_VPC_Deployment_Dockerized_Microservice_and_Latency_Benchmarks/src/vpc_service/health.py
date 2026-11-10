"""Liveness (/healthz) and sanitized Deep Readiness (/readyz) probes for the Client VPC service."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional, Tuple

from src.etl.vector_store import VectorStore
from src.rag.cli import DeterministicHashEmbedding
from src.vpc_service.secrets import SecretBundle


def check_liveness() -> Dict[str, str]:
    """Lightweight container liveness probe (/healthz). Never fails on downstream data state."""
    return {
        "status": "alive",
        "service": "datheon-fde-vpc-gateway",
        "version": "1.0.0",
    }


def evaluate_readiness(
    *,
    var_dir: Path,
    secret_bundle: Optional[SecretBundle],
    vector_store: Optional[VectorStore] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Deep dependency readiness check (/readyz):
    1. sqlite_db: verifies `acme_corp_erp.db` exists, executes `SELECT count(*) FROM inventory`, and has > 0 SKUs.
    2. chroma_vector_store: verifies collection `enterprise_docs` is reachable and has > 0 indexed chunks.
    3. secrets_manager: verifies SecretBundle is loaded with >= 1 active principal key.
    Returns (http_status_code, sanitized_response_dict) with zero file paths, DSNs, or stack traces.
    """
    checks: Dict[str, str] = {
        "sqlite_db": "degraded",
        "chroma_vector_store": "degraded",
        "secrets_manager": "degraded",
    }

    # 1. Check SQLite ERP database
    try:
        db_file = Path(var_dir) / "acme_corp_erp.db"
        if db_file.exists():
            conn = sqlite3.connect(str(db_file), timeout=2.0)
            try:
                row = conn.execute("SELECT count(*) FROM inventory").fetchone()
                if row and int(row[0]) > 0:
                    checks["sqlite_db"] = "ok"
            finally:
                conn.close()
    except Exception:
        checks["sqlite_db"] = "degraded"

    # 2. Check ChromaDB VectorStore
    try:
        if vector_store is not None:
            if vector_store.count() > 0:
                checks["chroma_vector_store"] = "ok"
        else:
            chroma_dir = Path(var_dir) / "chroma_db"
            if chroma_dir.exists():
                store = VectorStore(
                    path=chroma_dir,
                    collection_name="enterprise_docs",
                    embedding_function=DeterministicHashEmbedding(),
                )
                if store.count() > 0:
                    checks["chroma_vector_store"] = "ok"
    except Exception:
        checks["chroma_vector_store"] = "degraded"

    # 3. Check Secrets Manager
    try:
        if secret_bundle is not None and len(secret_bundle.api_key_map) > 0:
            checks["secrets_manager"] = "ok"
    except Exception:
        checks["secrets_manager"] = "degraded"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return status_code, {
        "status": "ready" if all_ok else "degraded",
        "checks": checks,
    }

