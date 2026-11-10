"""Seed the SQLite Enterprise ERP/CRM/ITSM database from data/seed_enterprise_data.json."""
import json
from pathlib import Path
from typing import Optional

from src.config import get_settings
from src.erp_crm.database import ERPCRMDatabase


def seed_database(
    db_path: Optional[Path] = None,
    seed_json_path: Optional[Path] = None,
) -> ERPCRMDatabase:
    """Initialize and populate the SQLite ERP/CRM database with deterministic seed records."""
    settings = get_settings()
    target_db = Path(db_path) if db_path is not None else settings.erp_db_path
    source_json = Path(seed_json_path) if seed_json_path is not None else settings.seed_json_path

    with open(source_json, "r", encoding="utf-8") as f:
        seed_data = json.load(f)

    db = ERPCRMDatabase(target_db)
    db.reset_and_seed(seed_data)
    return db


if __name__ == "__main__":
    seeded = seed_database()
    print(f"Seeded ERP/CRM SQLite database at {seeded.db_path}")
