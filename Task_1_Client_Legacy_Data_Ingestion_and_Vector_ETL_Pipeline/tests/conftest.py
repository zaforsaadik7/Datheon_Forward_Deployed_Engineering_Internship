from pathlib import Path
from typing import List, Optional
import pytest
import pymupdf


@pytest.fixture
def make_pdf():
    """Helper fixture to create PDF files with headers and footers."""
    def _make_pdf(
        path: Path,
        pages: List[str],
        header: Optional[str] = None,
        footer_template: Optional[str] = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open()
        total_pages = len(pages)

        for page_idx, page_text in enumerate(pages, start=1):
            page = doc.new_page(width=595, height=842)  # A4

            # Draw header
            if header:
                page.insert_text((50, 40), header, fontsize=10)

            # Draw body text with insert_textbox
            page.insert_textbox(pymupdf.Rect(50, 60, 545, 780), page_text, fontsize=11)

            # Draw footer
            if footer_template:
                footer_str = footer_template.format(n=page_idx, total=total_pages)
                page.insert_text((50, 810), footer_str, fontsize=10)

        doc.save(str(path))
        doc.close()
        return path

    return _make_pdf
