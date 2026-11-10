"""Unified superset configuration for ETL, Multi-Tenant RAG, ERP/CRM, Autonomous Agent, and VPC Service."""
from dataclasses import dataclass
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TIERS = (1, 2, 3, 4)
DEFAULT_TIER = 4
DEFAULT_TENANT = "acme_corp"

DEPARTMENTS = frozenset(
    {
        "engineering",
        "finance",
        "legal",
        "hr",
        "executive",
        "operations",
    }
)

COSINE_COLLECTION_KWARGS = {"metadata": {"hnsw:space": "cosine"}}


@dataclass(frozen=True)
class Settings:
    """Immutable configuration settings across RAG, ERP/CRM, Autonomous Agent, and VPC Gateway."""
    # Task 1 & 2: ETL + Multi-Tenant RBAC RAG
    data_dir: Path = BASE_DIR / "data" / "sample_documents"
    chroma_path: Path = BASE_DIR / "var" / "chroma_db"
    collection_name: str = "enterprise_docs"
    chunk_size: int = 900
    chunk_overlap: int = 120
    allow_dev_persona_header: bool = False

    # Task 3: ERP/CRM + Autonomous Field Agent
    seed_json_path: Path = BASE_DIR / "data" / "seed_enterprise_data.json"
    erp_db_path: Path = BASE_DIR / "var" / "enterprise_erp_crm.db"
    session_db_path: Path = BASE_DIR / "var" / "agent_sessions.db"
    max_steps: int = 8
    max_retries: int = 3
    base_backoff_s: float = 0.02
    approval_ttl_seconds: int = 900
    allow_dev_role_header: bool = False
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")


def get_settings() -> Settings:
    """Return default application settings."""
    return Settings()
