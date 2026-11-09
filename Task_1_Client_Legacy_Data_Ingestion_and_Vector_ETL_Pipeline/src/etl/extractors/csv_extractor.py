"""CSV extractor converting rows into citation-preserving segments."""
import csv
import io
from pathlib import Path
from typing import ClassVar, List
from .base import BaseExtractor, ExtractionError
from .text_extractor import read_text_file
from ..models import Segment


class CsvExtractor(BaseExtractor):
    """
    Extracts tabular enterprise CSVs. Each non-blank row becomes a Segment
    with exact spreadsheet row_id (1-indexed, header is row 1).
    """
    file_type: ClassVar[str] = "csv"

    def extract(self, path: Path) -> List[Segment]:
        raw_text = read_text_file(path)
        if not raw_text:
            raise ExtractionError(f"File '{path.name}' contains no extractable text.")

        reader = csv.reader(io.StringIO(raw_text))
        header: List[str] = []
        segments: List[Segment] = []

        for row_idx, row in enumerate(reader, start=1):
            stripped_cells = [cell.strip() for cell in row]

            # Row 1 is header
            if row_idx == 1:
                header = stripped_cells
                continue

            # Skip empty rows (blank line)
            if not any(stripped_cells):
                continue

            # Skip repeated header row
            if stripped_cells == header:
                continue

            # Format fields
            fields: List[str] = []
            for col_idx, cell_val in enumerate(stripped_cells):
                if not cell_val:
                    continue
                if col_idx < len(header) and header[col_idx]:
                    col_name = header[col_idx]
                else:
                    extra_num = col_idx - len(header) + 1
                    col_name = f"extra_{extra_num}"
                fields.append(f"{col_name}={cell_val}")

            if not fields:
                continue

            segment_text = f"Record [Row {row_idx}]: " + ", ".join(fields)
            segments.append(Segment(text=segment_text, row_id=row_idx))

        if not segments:
            raise ExtractionError(f"File '{path.name}' contains no extractable data rows.")

        return segments
