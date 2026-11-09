"""PDF extractor with boilerplate stripping and hyphenation repair."""
import logging
import math
import re
from pathlib import Path
from typing import ClassVar, List, Set, Tuple
import pymupdf

from .base import BaseExtractor, ExtractionError
from ..models import Segment

logger = logging.getLogger(__name__)


def _normalize_digits(line: str) -> str:
    """Replace all sequences of digits with '#'."""
    return re.sub(r"\d+", "#", line.strip())


class PdfExtractor(BaseExtractor):
    """
    Extracts text from multi-page PDFs using PyMuPDF.
    Removes repeated headers/footers, repairs hyphenated words, and retains page numbers.
    """
    file_type: ClassVar[str] = "pdf"

    def extract(self, path: Path) -> List[Segment]:
        path = Path(path)
        if not path.exists():
            raise ExtractionError(f"PDF file not found: {path}")

        try:
            doc = pymupdf.open(str(path))
        except Exception as e:
            raise ExtractionError(f"Failed to open PDF '{path.name}': {e}")

        page_count = len(doc)
        if page_count == 0:
            doc.close()
            raise ExtractionError(f"PDF '{path.name}' is empty or corrupt.")

        # Collect raw lines per page (preserving page index 1-based)
        raw_pages: List[Tuple[int, List[str]]] = []
        for idx in range(page_count):
            try:
                page = doc[idx]
                text = page.get_text()
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                raw_pages.append((idx + 1, lines))
            except Exception as e:
                logger.warning(f"Failed to read page {idx + 1} of '{path.name}': {e}")
                raw_pages.append((idx + 1, []))
        doc.close()

        # Boilerplate detection (only for multi-page documents)
        boilerplate_patterns: Set[str] = set()
        if page_count > 1:
            threshold = max(2, math.ceil(0.6 * page_count))
            candidate_counts: dict[str, int] = {}

            for _, lines in raw_pages:
                if not lines:
                    continue
                # First two and last two non-empty lines
                candidates = set()
                if len(lines) <= 4:
                    candidates.update(_normalize_digits(l) for l in lines)
                else:
                    top_lines = lines[:2]
                    bottom_lines = lines[-2:]
                    candidates.update(_normalize_digits(l) for l in top_lines)
                    candidates.update(_normalize_digits(l) for l in bottom_lines)

                for pattern in candidates:
                    candidate_counts[pattern] = candidate_counts.get(pattern, 0) + 1

            for pattern, count in candidate_counts.items():
                if count >= threshold:
                    boilerplate_patterns.add(pattern)

        # Process each page
        segments: List[Segment] = []
        all_pages_were_blank = True

        for page_num, lines in raw_pages:
            # Filter out boilerplate
            filtered_lines: List[str] = []
            for line in lines:
                norm = _normalize_digits(line)
                if norm not in boilerplate_patterns:
                    filtered_lines.append(line)

            page_text = "\n".join(filtered_lines).strip()
            if not page_text:
                logger.warning(f"Page {page_num} of '{path.name}' has no text (needs_ocr).")
                continue

            all_pages_were_blank = False

            # Hyphenation repair: (\w)-\n([a-z]) -> \1\2
            repaired = re.sub(r"(\w)-\n([a-z])", r"\1\2", page_text)

            # Join single newlines into spaces while keeping blank lines as paragraph breaks
            paragraphs = repaired.split("\n\n")
            cleaned_paras = []
            for p in paragraphs:
                single_spaced = re.sub(r"(?<!\n)\n(?!\n)", " ", p.strip())
                if single_spaced:
                    cleaned_paras.append(single_spaced)

            final_text = "\n\n".join(cleaned_paras).strip()
            if final_text:
                segments.append(Segment(text=final_text, page_number=page_num))

        if all_pages_were_blank or not segments:
            raise ExtractionError(
                f"PDF '{path.name}' contains no extractable text layer on any page (OCR required)."
            )

        return segments
