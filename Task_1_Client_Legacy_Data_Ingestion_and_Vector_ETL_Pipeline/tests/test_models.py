import pytest
from pydantic import ValidationError
from src.etl.models import DocumentChunk, Segment, to_chroma_metadata


def test_to_chroma_metadata_omits_none():
    chunk = DocumentChunk(
        chunk_id="chk1",
        document_id="doc1",
        text="Sample content",
        department="finance",
        clearance_tier=2,
        document_date="2026-10-15",
        source_file="doc1.pdf",
        file_type="pdf",
        chunk_index=0,
        page_number=None,
        row_id=None,
        section_title=None,
    )
    meta = to_chroma_metadata(chunk)
    assert "page_number" not in meta
    assert "row_id" not in meta
    assert "section_title" not in meta
    assert "text" not in meta
    assert None not in meta.values()
    assert meta["clearance_tier"] == 2
    assert meta["department"] == "finance"

    chunk_with_page = DocumentChunk(
        chunk_id="chk2",
        document_id="doc1",
        text="Sample content page 3",
        department="finance",
        clearance_tier=2,
        document_date="2026-10-15",
        source_file="doc1.pdf",
        file_type="pdf",
        chunk_index=1,
        page_number=3,
    )
    meta_page = to_chroma_metadata(chunk_with_page)
    assert meta_page["page_number"] == 3
    assert isinstance(meta_page["page_number"], int)


def test_department_is_lowercased():
    chunk = DocumentChunk(
        chunk_id="chk1",
        document_id="doc1",
        text="Sample",
        department="Finance",
        clearance_tier=3,
        document_date="2026-10-15",
        source_file="audit.csv",
        file_type="csv",
        chunk_index=0,
    )
    assert chunk.department == "finance"


def test_invalid_tier_rejected():
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="chk1",
            document_id="doc1",
            text="Sample",
            department="engineering",
            clearance_tier=0,
            document_date="2026-10-15",
            source_file="test.md",
            file_type="md",
            chunk_index=0,
        )

    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="chk2",
            document_id="doc1",
            text="Sample",
            department="engineering",
            clearance_tier=5,
            document_date="2026-10-15",
            source_file="test.md",
            file_type="md",
            chunk_index=0,
        )


def test_invalid_date_rejected():
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="chk1",
            document_id="doc1",
            text="Sample",
            department="legal",
            clearance_tier=4,
            document_date="15/10/2026",
            source_file="test.txt",
            file_type="txt",
            chunk_index=0,
        )


def test_unknown_department_rejected():
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="chk1",
            document_id="doc1",
            text="Sample",
            department="marketing",
            clearance_tier=1,
            document_date="2026-10-15",
            source_file="test.txt",
            file_type="txt",
            chunk_index=0,
        )
