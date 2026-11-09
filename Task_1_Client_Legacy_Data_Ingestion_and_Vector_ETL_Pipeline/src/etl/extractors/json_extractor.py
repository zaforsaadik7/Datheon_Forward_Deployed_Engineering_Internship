"""JSON extractor flattening records into citation-ready segments."""
import json
from pathlib import Path
from typing import ClassVar, List, Any, Dict
from .base import BaseExtractor, ExtractionError
from .text_extractor import read_text_file
from ..models import Segment


def _flatten_item(data: Any, prefix: str = "") -> List[str]:
    """Recursively flatten nested dicts/lists to dotted key paths with scalar values."""
    lines: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            lines.extend(_flatten_item(v, new_prefix))
    elif isinstance(data, list):
        if all(isinstance(x, (str, int, float, bool)) for x in data):
            # Join list of scalars with ", "
            joined = ", ".join(str(x) for x in data)
            lines.append(f"{prefix}: {joined}")
        else:
            for idx, x in enumerate(data):
                new_prefix = f"{prefix}[{idx}]"
                lines.extend(_flatten_item(x, new_prefix))
    else:
        lines.append(f"{prefix}: {data}")
    return lines


class JsonExtractor(BaseExtractor):
    """Extracts segments from JSON records."""
    file_type: ClassVar[str] = "json"

    def extract(self, path: Path) -> List[Segment]:
        raw_text = read_text_file(path)
        if not raw_text:
            raise ExtractionError(f"File '{path.name}' contains no extractable text.")

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ExtractionError(f"Invalid JSON in '{path.name}' at line {e.lineno}, column {e.colno}: {e}")

        segments: List[Segment] = []

        if isinstance(data, list):
            for row_idx, item in enumerate(data, start=1):
                lines = _flatten_item(item)
                text = "\n".join(lines).strip()
                if text:
                    segments.append(Segment(text=text, row_id=row_idx))

        elif isinstance(data, dict):
            # Separate scalar headers from lists of records
            scalar_headers: List[str] = []
            list_fields: Dict[str, list] = {}

            for k, v in data.items():
                if isinstance(v, list) and all(isinstance(elem, dict) for elem in v):
                    list_fields[k] = v
                else:
                    scalar_headers.extend(_flatten_item(v, prefix=str(k)))

            row_counter = 1
            if scalar_headers:
                segments.append(Segment(text="\n".join(scalar_headers).strip()))

            for list_name, records in list_fields.items():
                for idx, record in enumerate(records):
                    record_lines = _flatten_item(record, prefix=f"{list_name}[{idx}]")
                    record_text = "\n".join(record_lines).strip()
                    if record_text:
                        segments.append(Segment(text=record_text, row_id=row_counter))
                        row_counter += 1
        else:
            text = str(data).strip()
            if text:
                segments.append(Segment(text=text))

        if not segments:
            raise ExtractionError(f"File '{path.name}' contains no extractable text.")

        return segments
