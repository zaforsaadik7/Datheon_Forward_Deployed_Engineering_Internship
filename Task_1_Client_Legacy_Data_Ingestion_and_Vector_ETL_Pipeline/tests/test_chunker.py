import pytest
from src.etl.chunker import chunk_text, estimate_tokens, build_chunks
from src.etl.models import Segment, ManifestEntry


def test_short_text_single_chunk():
    text = "Short text under 900 chars."
    chunks = chunk_text(text, chunk_size=900, overlap=120)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_all_chunks_within_size():
    # 20,000 characters of text separated by paragraphs
    para = "This is a meaningful paragraph with several sentences that describe operations. " * 8
    text = "\n\n".join([para] * 40)
    assert len(text) >= 20000

    chunks = chunk_text(text, chunk_size=900, overlap=120)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 900


def test_prefers_paragraph_boundary():
    # Two 600-character paragraphs
    p1 = "First paragraph about corporate engineering goals. " * 12  # ~600 chars
    p2 = "Second paragraph describing cloud infrastructure architecture. " * 10  # ~600 chars
    text = f"{p1}\n\n{p2}"

    chunks = chunk_text(text, chunk_size=900, overlap=120)
    assert len(chunks) == 2
    assert chunks[0].endswith(p1.strip())


def test_overlap_carries_context():
    p1 = "Alpha " * 120  # ~720 chars
    p2 = "Beta " * 120
    text = f"{p1}\n\n{p2}"

    chunks = chunk_text(text, chunk_size=900, overlap=120)
    assert len(chunks) >= 2
    # Verify that the beginning of the second chunk overlaps with text from the first chunk
    overlap_sample = chunks[1][:40]
    assert overlap_sample in chunks[0]


def test_no_separator_run_terminates():
    text = "x" * 5000
    chunks = chunk_text(text, chunk_size=900, overlap=120)
    assert len(chunks) >= 6
    for c in chunks:
        assert len(c) <= 900


def test_empty_and_whitespace_give_empty_list():
    assert chunk_text("", chunk_size=900, overlap=120) == []
    assert chunk_text("   \n\n\t  ", chunk_size=900, overlap=120) == []


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("test", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("test", chunk_size=500, overlap=500)
    with pytest.raises(ValueError):
        chunk_text("test", chunk_size=500, overlap=600)


def test_build_chunks_inherits_metadata():
    segments = [
        Segment(text="Page 2 text content here.", page_number=2),
    ]
    entry = ManifestEntry(
        department="finance",
        clearance_tier=3,
        document_date="2026-10-15",
        metadata_source="manifest",
    )
    chunks = build_chunks(
        segments=segments,
        document_id="sub/doc.pdf",
        source_file="doc.pdf",
        file_type="pdf",
        entry=entry,
        chunk_size=900,
        overlap=120,
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c.page_number == 2
    assert c.department == "finance"
    assert c.clearance_tier == 3
    assert c.document_date == "2026-10-15"
    assert c.source_file == "doc.pdf"
    assert c.document_id == "sub/doc.pdf"


def test_chunk_ids_deterministic_and_unique():
    segments = [
        Segment(text="Section A content", section_title="Sec A"),
        Segment(text="Section B content", section_title="Sec B"),
    ]
    entry = ManifestEntry(
        department="engineering",
        clearance_tier=2,
        document_date="2026-08-01",
        metadata_source="manifest",
    )
    run1 = build_chunks(
        segments=segments,
        document_id="runbook.md",
        source_file="runbook.md",
        file_type="markdown",
        entry=entry,
    )
    run2 = build_chunks(
        segments=segments,
        document_id="runbook.md",
        source_file="runbook.md",
        file_type="markdown",
        entry=entry,
    )

    ids1 = [c.chunk_id for c in run1]
    ids2 = [c.chunk_id for c in run2]
    assert ids1 == ids2
    assert len(ids1) == len(set(ids1))


def test_segments_are_never_merged():
    segments = [
        Segment(text="Row 2 data", row_id=2),
        Segment(text="Row 3 data", row_id=3),
    ]
    entry = ManifestEntry(
        department="hr",
        clearance_tier=2,
        document_date="2026-05-12",
        metadata_source="manifest",
    )
    chunks = build_chunks(
        segments=segments,
        document_id="bands.csv",
        source_file="bands.csv",
        file_type="csv",
        entry=entry,
    )
    assert len(chunks) == 2
    assert chunks[0].row_id == 2
    assert chunks[1].row_id == 3


def test_hard_cut_keeps_final_characters():
    # Last window is exactly the packing budget (780) with no separators:
    # the old join added a space, hit 901 characters and cut off the last one.
    text = "A" * (780 * 7 - 8) + "ENDMARK1"
    chunks = chunk_text(text, chunk_size=900, overlap=120)
    assert chunks[-1].endswith("ENDMARK1")
    assert all(len(c) <= 900 for c in chunks)


def test_hard_cut_does_not_inject_spaces():
    chunks = chunk_text("x" * 5000, chunk_size=900, overlap=120)
    assert all(" " not in c for c in chunks)


def test_no_content_lost_across_chunks():
    text = "".join(f"[{i:04d}]" for i in range(1200))  # 7,200 chars, no separators
    joined = "".join(chunk_text(text, chunk_size=900, overlap=120))
    assert all(f"[{i:04d}]" in joined for i in range(1200))
