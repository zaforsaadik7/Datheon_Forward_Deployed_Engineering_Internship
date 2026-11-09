import json
from pathlib import Path
import pytest
from src.etl.extractors.base import ExtractionError
from src.etl.extractors.text_extractor import TextExtractor, read_text_file
from src.etl.extractors.json_extractor import JsonExtractor
from src.etl.extractors.markdown_extractor import MarkdownExtractor


# --- TXT Extractor Tests ---

def test_txt_crlf_bom_trailing_space_normalized(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"\xef\xbb\xbfLine one\r\nLine two  \r\n")

    extractor = TextExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 1
    assert segments[0].text == "Line one\nLine two"


def test_txt_cp1252_fallback(tmp_path: Path):
    file_path = tmp_path / "resume.txt"
    file_path.write_bytes("Café – résumé".encode("cp1252"))

    extractor = TextExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 1
    assert "Café – résumé" in segments[0].text


def test_empty_file_raises(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    extractor = TextExtractor()
    with pytest.raises(ExtractionError, match="no extractable text"):
        extractor.extract(file_path)


# --- JSON Extractor Tests ---

def test_json_list_of_records(tmp_path: Path):
    data = [
        {"name": "Acme Cloud", "tier": "enterprise"},
        {"name": "Beta Services", "tier": "standard"},
    ]
    file_path = tmp_path / "suppliers.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    extractor = JsonExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 2
    assert segments[0].row_id == 1
    assert segments[1].row_id == 2
    assert "name: Acme Cloud" in segments[0].text


def test_json_nested_keys_flattened(tmp_path: Path):
    data = [{"a": {"b": [1, 2]}}]
    file_path = tmp_path / "nested.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    extractor = JsonExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 1
    assert "a.b: 1, 2" in segments[0].text


def test_json_dict_with_record_list(tmp_path: Path):
    data = {
        "catalog_version": "2026.1",
        "author": "IT Ops",
        "suppliers": [
            {"name": "Supplier A", "sla": "99.9%"},
            {"name": "Supplier B", "sla": "99.5%"},
        ]
    }
    file_path = tmp_path / "catalog.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    extractor = JsonExtractor()
    segments = extractor.extract(file_path)

    # 1 segment for header scalars, plus 2 segments for the suppliers list
    assert len(segments) == 3
    assert "catalog_version: 2026.1" in segments[0].text
    assert "suppliers[0]" in segments[1].text
    assert "suppliers[1]" in segments[2].text


def test_invalid_json_error_has_position(tmp_path: Path):
    file_path = tmp_path / "bad.json"
    file_path.write_text('{"name": "Broken", ', encoding="utf-8")

    extractor = JsonExtractor()
    with pytest.raises(ExtractionError) as exc_info:
        extractor.extract(file_path)

    assert "line" in str(exc_info.value).lower()


# --- Markdown Extractor Tests ---

def test_markdown_breadcrumb(tmp_path: Path):
    content = "# Network\n## VPC Peering Architecture\nbody text here"
    file_path = tmp_path / "runbook.md"
    file_path.write_text(content, encoding="utf-8")

    extractor = MarkdownExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 1
    assert segments[0].section_title == "Network > VPC Peering Architecture"


def test_markdown_ignores_hash_in_code_fence(tmp_path: Path):
    content = """# Setup Guide
Run the following bash command:
```bash
# This is a comment inside code block
echo "Hello"
```
Done with setup.
"""
    file_path = tmp_path / "guide.md"
    file_path.write_text(content, encoding="utf-8")

    extractor = MarkdownExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 1
    assert segments[0].section_title == "Setup Guide"


def test_markdown_preamble_has_no_title(tmp_path: Path):
    content = """This is preamble text with no header.

# Section One
Content of section one.
"""
    file_path = tmp_path / "preamble.md"
    file_path.write_text(content, encoding="utf-8")

    extractor = MarkdownExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 2
    assert segments[0].section_title is None
    assert segments[1].section_title == "Section One"


def test_markdown_sibling_resets_stack(tmp_path: Path):
    content = """# Architecture
## Compute
Compute body
## Storage
Storage body
# Security
Security body
"""
    file_path = tmp_path / "arch.md"
    file_path.write_text(content, encoding="utf-8")

    extractor = MarkdownExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 3
    assert segments[0].section_title == "Architecture > Compute"
    assert segments[1].section_title == "Architecture > Storage"
    assert segments[2].section_title == "Security"


def test_markdown_segment_text_starts_with_title(tmp_path: Path):
    content = "# Overview\nWelcome to Datheon."
    file_path = tmp_path / "doc.md"
    file_path.write_text(content, encoding="utf-8")

    extractor = MarkdownExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 1
    assert segments[0].text.startswith("Overview\n\nWelcome to Datheon.")


# --- CSV Extractor Tests ---
from src.etl.extractors.csv_extractor import CsvExtractor


def test_csv_row_serialization(tmp_path: Path):
    content = 'Department,Q3 Spend,Vendor\nFinance,"$450,000",Acme Cloud\n'
    file_path = tmp_path / "budget.csv"
    file_path.write_text(content, encoding="utf-8")

    extractor = CsvExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 1
    assert segments[0].row_id == 2
    assert segments[0].text == 'Record [Row 2]: Department=Finance, Q3 Spend=$450,000, Vendor=Acme Cloud'


def test_csv_bom_and_cp1252(tmp_path: Path):
    # UTF-8 BOM
    p1 = tmp_path / "bom.csv"
    p1.write_bytes(b'\xef\xbb\xbfName,Salary\nAlice,1000\n')
    # CP1252 with euro symbol
    p2 = tmp_path / "cp.csv"
    p2.write_bytes("Name,Bonus\nRené,500 €\n".encode("cp1252"))

    extractor = CsvExtractor()
    s1 = extractor.extract(p1)
    s2 = extractor.extract(p2)

    assert len(s1) == 1
    assert "Alice" in s1[0].text
    assert len(s2) == 1
    assert "René" in s2[0].text
    assert "500 €" in s2[0].text


def test_csv_blank_rows_skipped_row_ids_keep_gaps(tmp_path: Path):
    content = "A,B\nVal1,Val2\n\nVal3,Val4\n"
    file_path = tmp_path / "blanks.csv"
    file_path.write_text(content, encoding="utf-8")

    extractor = CsvExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 2
    assert segments[0].row_id == 2
    assert segments[1].row_id == 4


def test_csv_ragged_rows_do_not_crash(tmp_path: Path):
    content = "H1,H2,H3\nA\nB,C,D,E,F\n"
    file_path = tmp_path / "ragged.csv"
    file_path.write_text(content, encoding="utf-8")

    extractor = CsvExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 2
    assert segments[0].text == "Record [Row 2]: H1=A"
    assert "H1=B" in segments[1].text
    assert "extra_1=E" in segments[1].text
    assert "extra_2=F" in segments[1].text


def test_csv_repeated_header_row_skipped(tmp_path: Path):
    content = "H1,H2\nA,B\nH1,H2\nC,D\n"
    file_path = tmp_path / "repeat.csv"
    file_path.write_text(content, encoding="utf-8")

    extractor = CsvExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 2
    assert segments[0].row_id == 2
    assert segments[1].row_id == 4


def test_csv_multiline_quoted_cell(tmp_path: Path):
    content = 'ID,Note\n1,"Line one\nLine two"\n'
    file_path = tmp_path / "multiline.csv"
    file_path.write_text(content, encoding="utf-8")

    extractor = CsvExtractor()
    segments = extractor.extract(file_path)

    assert len(segments) == 1
    assert "Line one\nLine two" in segments[0].text


def test_csv_header_only_raises(tmp_path: Path):
    content = "H1,H2,H3\n"
    file_path = tmp_path / "header_only.csv"
    file_path.write_text(content, encoding="utf-8")

    extractor = CsvExtractor()
    with pytest.raises(ExtractionError):
        extractor.extract(file_path)


# --- PDF Extractor and Registry Tests ---
from src.etl.extractors.pdf_extractor import PdfExtractor
from src.etl.extractors import get_extractor
from src.etl.extractors.base import UnsupportedFileType


def test_pdf_page_numbers_one_based(tmp_path: Path, make_pdf):
    pdf_path = make_pdf(
        tmp_path / "three_pages.pdf",
        pages=["Page one content", "Page two content", "Page three content"]
    )
    extractor = PdfExtractor()
    segments = extractor.extract(pdf_path)

    assert [s.page_number for s in segments] == [1, 2, 3]


def test_pdf_strips_repeated_header_and_page_counter(tmp_path: Path, make_pdf):
    pdf_path = make_pdf(
        tmp_path / "repeats.pdf",
        pages=[
            "First page valuable corporate information.",
            "Second page continuous business updates.",
            "Third page conclusion and sign-off."
        ],
        header="MERIDIAN CONFIDENTIAL",
        footer_template="Page {n} of {total}"
    )
    extractor = PdfExtractor()
    segments = extractor.extract(pdf_path)

    assert len(segments) == 3
    for s in segments:
        assert "MERIDIAN CONFIDENTIAL" not in s.text
        assert "Page " not in s.text
    assert "corporate information" in segments[0].text


def test_pdf_single_page_keeps_title_line(tmp_path: Path, make_pdf):
    pdf_path = make_pdf(
        tmp_path / "single.pdf",
        pages=["Important Document Title\nBody of single page document."]
    )
    extractor = PdfExtractor()
    segments = extractor.extract(pdf_path)

    assert len(segments) == 1
    assert "Important Document Title" in segments[0].text


def test_pdf_hyphenated_break_repaired(tmp_path: Path, make_pdf):
    pdf_path = make_pdf(
        tmp_path / "hyphen.pdf",
        pages=["The inter-\nnational team met yesterday."]
    )
    extractor = PdfExtractor()
    segments = extractor.extract(pdf_path)

    assert len(segments) == 1
    assert "international" in segments[0].text


def test_pdf_blank_page_skipped_others_kept(tmp_path: Path, make_pdf):
    pdf_path = make_pdf(
        tmp_path / "blank_middle.pdf",
        pages=["First page content.", "   \n  \n  ", "Third page content."]
    )
    extractor = PdfExtractor()
    segments = extractor.extract(pdf_path)

    assert len(segments) == 2
    assert [s.page_number for s in segments] == [1, 3]


def test_pdf_all_blank_raises_ocr_message(tmp_path: Path, make_pdf):
    pdf_path = make_pdf(
        tmp_path / "all_blank.pdf",
        pages=["   ", " \n "]
    )
    extractor = PdfExtractor()
    with pytest.raises(ExtractionError) as exc_info:
        extractor.extract(pdf_path)
    assert "ocr" in str(exc_info.value).lower()


def test_zero_byte_pdf_raises_extraction_error(tmp_path: Path):
    pdf_path = tmp_path / "zero.pdf"
    pdf_path.write_bytes(b"")

    extractor = PdfExtractor()
    with pytest.raises(ExtractionError):
        extractor.extract(pdf_path)


def test_get_extractor_by_suffix(tmp_path: Path):
    assert get_extractor(Path("doc.pdf")).file_type == "pdf"
    assert get_extractor(Path("doc.MD")).file_type == "markdown"
    assert get_extractor(Path("doc.csv")).file_type == "csv"
    assert get_extractor(Path("doc.json")).file_type == "json"
    assert get_extractor(Path("doc.txt")).file_type == "txt"

    with pytest.raises(UnsupportedFileType):
        get_extractor(Path("doc.docx"))
