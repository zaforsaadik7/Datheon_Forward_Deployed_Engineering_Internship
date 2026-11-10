"""Project-wide configuration, security tiers, tenants, and collection constants."""
from dataclasses import dataclass
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
    """Immutable configuration settings for ETL and Multi-Tenant RAG."""
    data_dir: Path = BASE_DIR / "data" / "sample_documents"
    chroma_path: Path = BASE_DIR / "chroma_db"
    collection_name: str = "enterprise_docs"
    chunk_size: int = 900
    chunk_overlap: int = 120
    allow_dev_persona_header: bool = False


def get_settings() -> Settings:
    """Return default application settings."""
    return Settings()
