"""Application configuration, retry policies, approval TTL, and LLM settings."""
from dataclasses import dataclass
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Immutable configuration for ERP/CRM database and Autonomous Field Agent."""
    seed_json_path: Path = BASE_DIR / "data" / "seed_enterprise_data.json"
    erp_db_path: Path = BASE_DIR / "data" / "enterprise_erp_crm.db"
    session_db_path: Path = BASE_DIR / "data" / "agent_sessions.db"
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

