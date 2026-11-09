"""Tabular CSV extractor converting structured corporate rows into semantic units."""
import csv
from pathlib import Path
from typing import List, Dict, Any
from .base import BaseExtractor
from ..models import RawDocument, DocumentMetadata


class CSVExtractor(BaseExtractor):
    """
    Extracts tabular enterprise CSVs and converts each row into a semantically
    rich narrative document, preserving column headers and row indices.
    """

    def extract(self, file_path: Path) -> List[RawDocument]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        extracted_docs: List[RawDocument] = []
        base_dept, base_tier, base_date = self.infer_metadata(file_path)

        with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            for row_idx, row in enumerate(reader, start=1):
                # Clean row values
                cleaned_row = {k.strip(): (v.strip() if v else "") for k, v in row.items() if k}
                if not any(cleaned_row.values()):
                    continue

                # Formulate semantic statement
                field_strs = [f"{k}: {v}" for k, v in cleaned_row.items() if v]
                narrative = (
                    f"Corporate Record [Row {row_idx}] from '{file_path.name}':\n"
                    + " | ".join(field_strs)
                )

                # Check if row has row-level department or clearance overrides
                row_dept = base_dept
                row_tier = base_tier
                row_date = base_date

                for k, v in cleaned_row.items():
                    k_lower = k.lower()
                    v_lower = v.lower()
                    if "department" in k_lower or "dept" in k_lower:
                        if v.strip():
                            row_dept = v.strip()
                    elif "tier" in k_lower or "clearance" in k_lower:
                        if "4" in v_lower or "secret" in v_lower:
                            row_tier = 4
                        elif "3" in v_lower or "confidential" in v_lower:
                            row_tier = 3
                        elif "1" in v_lower or "public" in v_lower:
                            row_tier = 1
                        elif "2" in v_lower or "internal" in v_lower:
                            row_tier = 2
                    elif "date" in k_lower and len(v.strip()) == 10 and "-" in v:
                        row_date = v.strip()

                metadata = DocumentMetadata(
                    department=row_dept,
                    clearance_tier=row_tier,
                    document_date=row_date,
                    source_file=file_path.name,
                    file_type="csv",
                    row_id=row_idx,
                    extra={k: v for k, v in cleaned_row.items() if len(str(v)) < 100}
                )

                doc_id = f"{file_path.stem}_row{row_idx}"
                extracted_docs.append(
                    RawDocument(
                        document_id=doc_id,
                        content=narrative,
                        metadata=metadata
                    )
                )

        return extracted_docs
