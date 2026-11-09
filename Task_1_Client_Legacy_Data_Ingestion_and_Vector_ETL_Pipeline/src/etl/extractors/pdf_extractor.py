"""High-performance PDF extractor with page tracking and metadata preservation."""
import re
from pathlib import Path
from typing import List
from .base import BaseExtractor
from ..models import RawDocument, DocumentMetadata


class PDFExtractor(BaseExtractor):
    """
    Extracts structured text from multi-page enterprise PDFs.
    Preserves exact page numbers for downstream verifiable citations.
    """

    def __init__(self, clean_headers: bool = True):
        self.clean_headers = clean_headers

    def extract(self, file_path: Path) -> List[RawDocument]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        extracted_docs: List[RawDocument] = []

        # Try pymupdf (fitz) first, fallback to pypdf
        try:
            import pymupdf  # type: ignore
            doc = pymupdf.open(str(file_path))
            full_text_sample = ""
            pages_data = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text()
                if text.strip():
                    pages_data.append((page_idx + 1, text))
                    if len(full_text_sample) < 2000:
                        full_text_sample += f" {text}"
            doc.close()

        except Exception:
            # Fallback to pypdf
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(str(file_path))
            full_text_sample = ""
            pages_data = []

            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_data.append((page_idx + 1, text))
                    if len(full_text_sample) < 2000:
                        full_text_sample += f" {text}"

        dept, tier, doc_date = self.infer_metadata(file_path, full_text_sample)

        for page_num, raw_text in pages_data:
            clean_text = self._clean_text(raw_text)
            if not clean_text:
                continue

            metadata = DocumentMetadata(
                department=dept,
                clearance_tier=tier,
                document_date=doc_date,
                source_file=file_path.name,
                file_type="pdf",
                page_number=page_num,
                extra={"total_pages": len(pages_data)}
            )

            doc_id = f"{file_path.stem}_p{page_num}"
            extracted_docs.append(
                RawDocument(
                    document_id=doc_id,
                    content=clean_text,
                    metadata=metadata
                )
            )

        return extracted_docs

    def _clean_text(self, text: str) -> str:
        """Normalize line breaks, remove redundant carriage returns, and clean whitespace."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multi-spaces within lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        # Remove consecutive blank lines
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            if not line:
                if not prev_blank:
                    cleaned_lines.append("")
                prev_blank = True
            else:
                cleaned_lines.append(line)
                prev_blank = False
        return "\n".join(cleaned_lines).strip()
