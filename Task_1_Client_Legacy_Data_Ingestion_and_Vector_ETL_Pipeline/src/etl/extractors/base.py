"""Abstract base extractor and heuristic metadata resolvers."""
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from ..models import RawDocument, DocumentMetadata


class BaseExtractor(ABC):
    """Abstract interface for all format-specific extractors."""

    @abstractmethod
    def extract(self, file_path: Path) -> List[RawDocument]:
        """Extract multi-part or single-part documents from the given file."""
        pass

    @staticmethod
    def infer_metadata(file_path: Path, sample_text: str = "") -> Tuple[str, int, str]:
        """
        Infer (department, clearance_tier, document_date) using robust heuristics
        combining filename conventions and in-text classification markers.
        """
        combined = f"{file_path.stem} {sample_text[:1000]}".lower()

        # 1. Infer Clearance Tier (Tiers 1 to 4)
        tier = 2  # Default to Tier 2 (Internal)
        if "tier 4" in combined or "tier4" in combined or "secret" in combined or "restricted" in combined:
            tier = 4
        elif "tier 3" in combined or "tier3" in combined or "confidential" in combined or "executive" in combined:
            tier = 3
        elif "tier 1" in combined or "tier1" in combined or "public" in combined:
            tier = 1
        elif "tier 2" in combined or "tier2" in combined or "internal" in combined:
            tier = 2

        # 2. Infer Department
        department = "General"
        if "eng" in combined or "vpc" in combined or "cloud" in combined or "software" in combined:
            department = "Engineering"
        elif "fin" in combined or "budget" in combined or "audit" in combined or "spend" in combined:
            department = "Finance"
        elif "leg" in combined or "m&a" in combined or "nda" in combined or "covenant" in combined or "license" in combined:
            department = "Legal"
        elif "hr" in combined or "salary" in combined or "compensation" in combined or "talent" in combined:
            department = "HR"
        elif "exec" in combined or "roadmap" in combined or "board" in combined or "strategic" in combined:
            department = "Executive"
        elif "oper" in combined or "supplier" in combined or "catalog" in combined:
            department = "Operations"
        elif "sec" in combined or "compliance" in combined or "soc" in combined:
            department = "IT/Security"

        # 3. Infer Document Date
        date_match = re.search(r"\b(202[0-9]-[0-1][0-9]-[0-3][0-9])\b", combined)
        if date_match:
            document_date = date_match.group(1)
        else:
            try:
                mtime = file_path.stat().st_mtime
                document_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            except Exception:
                document_date = datetime.now().strftime("%Y-%m-%d")

        return department, tier, document_date

