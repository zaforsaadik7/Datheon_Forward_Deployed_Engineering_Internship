"""Project configuration and global constants for Vector ETL Pipeline."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

# Root paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Clearance Tiers: 1 (Public), 2 (Internal), 3 (Confidential), 4 (Secret)
TIERS = (1, 2, 3, 4)
DEFAULT_TIER = 4

# Fixed corporate departments, strictly lowercase
DEPARTMENTS: frozenset[str] = frozenset({
    "engineering",
    "finance",
    "legal",
    "hr",
    "executive",
    "operations",
})

# ChromaDB cosine similarity configuration
# Verified for ChromaDB 1.5.9: metadata={"hnsw:space": "cosine"}
COSINE_COLLECTION_KWARGS: Dict[str, Any] = {
    "metadata": {"hnsw:space": "cosine"}
}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = BASE_DIR / "data"
    chroma_path: Path = BASE_DIR / "data" / "chroma_db"
    collection_name: str = "enterprise_docs"
    chunk_size: int = 900
    chunk_overlap: int = 120


_DEFAULT_SETTINGS = Settings()


def get_settings() -> Settings:
    """Return the singleton frozen Settings instance."""
    return _DEFAULT_SETTINGS
