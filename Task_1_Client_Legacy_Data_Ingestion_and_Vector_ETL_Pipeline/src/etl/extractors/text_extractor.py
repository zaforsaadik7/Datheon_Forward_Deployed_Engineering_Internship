"""Plain text and JSON extractors for legacy enterprise records."""
import json
from pathlib import Path
from typing import List, Any
from .base import BaseExtractor
from ..models import RawDocument, DocumentMetadata


class TextExtractor(BaseExtractor):
    """Extracts raw text files (.txt, .log) preserving paragraph structure."""

    def extract(self, file_path: Path) -> List[RawDocument]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        dept, tier, doc_date = self.infer_metadata(file_path, content)

        metadata = DocumentMetadata(
            department=dept,
            clearance_tier=tier,
            document_date=doc_date,
            source_file=file_path.name,
            file_type="txt"
        )

        return [
            RawDocument(
                document_id=file_path.stem,
                content=content.strip(),
                metadata=metadata
            )
        ]


class JSONExtractor(BaseExtractor):
    """
    Extracts structured JSON records, flattening lists of enterprise entities
    into individual searchable document entries.
    """

    def extract(self, file_path: Path) -> List[RawDocument]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {file_path}: {e}")

        dept, tier, doc_date = self.infer_metadata(file_path, str(data))

        # Check top-level metadata if present
        if isinstance(data, dict) and "metadata" in data and isinstance(data["metadata"], dict):
            m = data["metadata"]
            dept = m.get("department", dept)
            tier = int(m.get("clearance_tier", tier))
            doc_date = m.get("effective_date", m.get("date", doc_date))

        extracted_docs: List[RawDocument] = []

        # If data is a list of objects or contains a list of items
        items_to_process = []
        if isinstance(data, list):
            items_to_process = data
        elif isinstance(data, dict):
            # Find list keys (e.g. "approved_suppliers", "items", "records")
            list_keys = [k for k, v in data.items() if isinstance(v, list)]
            if list_keys:
                for k in list_keys:
                    for item in data[k]:
                        items_to_process.append((k, item))
            else:
                items_to_process = [("root", data)]

        for idx, entry in enumerate(items_to_process, start=1):
            category = "item"
            item_data = entry
            if isinstance(entry, tuple):
                category, item_data = entry

            if isinstance(item_data, dict):
                item_str = f"Enterprise JSON Record [{category} #{idx}] from '{file_path.name}':\n" + "\n".join(
                    f"- {k}: {v}" for k, v in item_data.items()
                )
            else:
                item_str = f"Record #{idx} from '{file_path.name}': {str(item_data)}"

            metadata = DocumentMetadata(
                department=dept,
                clearance_tier=tier,
                document_date=doc_date,
                source_file=file_path.name,
                file_type="json",
                row_id=idx,
                section_title=category,
                extra={"json_category": category}
            )

            extracted_docs.append(
                RawDocument(
                    document_id=f"{file_path.stem}_{category}_{idx}",
                    content=item_str,
                    metadata=metadata
                )
            )

        return extracted_docs
