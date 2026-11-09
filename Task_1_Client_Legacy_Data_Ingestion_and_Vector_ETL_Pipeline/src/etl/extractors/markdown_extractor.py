"""Markdown extractor that tracks section headers and document hierarchy."""
import re
from pathlib import Path
from typing import List, Tuple
from .base import BaseExtractor
from ..models import RawDocument, DocumentMetadata


class MarkdownExtractor(BaseExtractor):
    """
    Extracts Markdown documents while maintaining header hierarchies (#, ##, ###).
    Captures section titles in metadata to provide structural context during vector retrieval.
    """

    def extract(self, file_path: Path) -> List[RawDocument]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        dept, tier, doc_date = self.infer_metadata(file_path, content)
        sections = self._split_by_headers(content)

        extracted_docs: List[RawDocument] = []
        doc_title = sections[0][0] if sections else file_path.stem

        for idx, (section_title, section_text) in enumerate(sections, start=1):
            if not section_text.strip():
                continue

            # Prefix chunk with document context if it's a subsection
            full_context = f"# Document: {doc_title}\n## Section: {section_title}\n\n{section_text.strip()}"

            metadata = DocumentMetadata(
                department=dept,
                clearance_tier=tier,
                document_date=doc_date,
                source_file=file_path.name,
                file_type="markdown",
                section_title=section_title,
                extra={"section_index": idx, "root_document": doc_title}
            )

            doc_id = f"{file_path.stem}_sec{idx}"
            extracted_docs.append(
                RawDocument(
                    document_id=doc_id,
                    content=full_context,
                    metadata=metadata
                )
            )

        return extracted_docs

    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        """Split markdown content by top-level and second-level headers."""
        lines = text.split("\n")
        sections: List[Tuple[str, str]] = []
        current_title = "Introduction"
        current_lines: List[str] = []

        header_pattern = re.compile(r"^(#{1,3})\s+(.+)$")

        for line in lines:
            match = header_pattern.match(line.strip())
            if match:
                # Flush previous section if it has text
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines)))
                    current_lines = []
                current_title = match.group(2).strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

        return sections
