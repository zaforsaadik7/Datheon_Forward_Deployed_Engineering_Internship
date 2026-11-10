from pathlib import Path
import pytest
from src.etl.pipeline import discover_files, ingest_directory
from src.etl.vector_store import VectorStore
from tests.conftest import FakeEmbedding


def test_discover_files_recursive_and_skips_manifest(tmp_path: Path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.txt").write_text("a", encoding="utf-8")
    (d / "manifest.yaml").write_text("manifest", encoding="utf-8")
    (d / ".hidden.txt").write_text("hidden", encoding="utf-8")
    sub = d / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("b", encoding="utf-8")
    (sub / "c.docx").write_text("unsupported", encoding="utf-8")

    files = discover_files(d)
    rel_names = [f.relative_to(d).as_posix() for f in files]
    assert "a.txt" in rel_names
    assert "sub/b.md" in rel_names
    assert "sub/c.docx" in rel_names  # Kept so discover stage reports it!
    assert "manifest.yaml" not in rel_names
    assert ".hidden.txt" not in rel_names


def test_ingests_all_formats(tmp_path: Path, make_pdf):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # 1. TXT
    (data_dir / "engineering_guide_tier1.txt").write_text("Engineering guide content.", encoding="utf-8")
    # 2. MD
    (data_dir / "engineering_runbook_tier2.md").write_text("# Section\nRunbook details.", encoding="utf-8")
    # 3. CSV
    (data_dir / "finance_audit_tier3.csv").write_text("Col1,Col2\nA,B\n", encoding="utf-8")
    # 4. JSON
    (data_dir / "operations_suppliers_tier1.json").write_text('[{"name": "Supplier X"}]', encoding="utf-8")
    # 5. PDF
    make_pdf(data_dir / "operations_policy_tier1.pdf", pages=["Policy text page."])

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_all", embedding_function=FakeEmbedding())
    report = ingest_directory(data_dir, store)

    assert report.documents_ok == 5
    assert len(report.documents_failed) == 0
    assert store.count() == report.chunks_written
    assert report.bytes_read > 0


def test_chunk_metadata_complete(tmp_path: Path, make_pdf):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "finance_audit_tier3.csv").write_text("Spend,Vendor\n100,Acme\n", encoding="utf-8")
    (data_dir / "engineering_arch_tier2.md").write_text("# Header\nArch body.", encoding="utf-8")
    make_pdf(data_dir / "legal_contract_tier4.pdf", pages=["Page 1 contract text."])

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_meta", embedding_function=FakeEmbedding())
    ingest_directory(data_dir, store)

    coll_data = store.collection.get()
    for meta in coll_data["metadatas"]:
        assert "department" in meta
        assert "clearance_tier" in meta
        assert "document_date" in meta
        assert "source_file" in meta
        assert "file_type" in meta
        assert "document_id" in meta

    # Check format-specific keys
    csv_chunks = [m for m in coll_data["metadatas"] if m["file_type"] == "csv"]
    md_chunks = [m for m in coll_data["metadatas"] if m["file_type"] == "markdown"]
    pdf_chunks = [m for m in coll_data["metadatas"] if m["file_type"] == "pdf"]

    assert any("row_id" in m for m in csv_chunks)
    assert any("section_title" in m for m in md_chunks)
    assert any("page_number" in m for m in pdf_chunks)


def test_document_id_is_relative_posix_path(tmp_path: Path):
    data_dir = tmp_path / "data"
    sub = data_dir / "sub"
    sub.mkdir(parents=True)
    (sub / "engineering_a_tier1.txt").write_text("Sub content.", encoding="utf-8")

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_rel", embedding_function=FakeEmbedding())
    ingest_directory(data_dir, store)

    meta = store.collection.get()["metadatas"][0]
    assert meta["document_id"] == "sub/engineering_a_tier1.txt"


def test_reingest_unchanged_is_idempotent(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "engineering_a_tier1.txt").write_text("Hello world.", encoding="utf-8")

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_idem", embedding_function=FakeEmbedding())
    r1 = ingest_directory(data_dir, store)
    count1 = store.count()

    r2 = ingest_directory(data_dir, store)
    count2 = store.count()

    assert count1 == count2
    assert r1.chunks_written == r2.chunks_written


def test_reingest_shrunk_document_removes_stale_chunks(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    doc_path = data_dir / "engineering_long_tier2.txt"

    # Write long document that generates at least 5 chunks
    doc_path.write_text("Paragraph description. " * 300, encoding="utf-8")

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_shrink", embedding_function=FakeEmbedding())
    r1 = ingest_directory(data_dir, store, chunk_size=300, overlap=50)
    assert store.count() >= 5

    # Overwrite with very short text that produces 1 chunk
    doc_path.write_text("Tiny content.", encoding="utf-8")
    r2 = ingest_directory(data_dir, store, chunk_size=300, overlap=50)

    assert store.count() == 1
    coll_data = store.collection.get()
    assert len(coll_data["ids"]) == 1
    assert coll_data["metadatas"][0]["chunk_index"] == 0


def test_corrupt_files_do_not_abort_batch(tmp_path: Path, make_pdf):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Good CSV
    (data_dir / "finance_good_tier3.csv").write_text("A,B\n1,2\n", encoding="utf-8")
    # Truncated JSON
    (data_dir / "operations_bad_tier1.json").write_text('{"unclosed": [', encoding="utf-8")
    # Zero-byte PDF
    (data_dir / "legal_zero_tier4.pdf").write_bytes(b"")

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_corrupt", embedding_function=FakeEmbedding())
    report = ingest_directory(data_dir, store)

    assert report.documents_ok == 1
    assert len(report.documents_failed) == 2
    failed_names = {f.file for f in report.documents_failed}
    assert "operations_bad_tier1.json" in failed_names
    assert "legal_zero_tier4.pdf" in failed_names
    assert store.count() == 1


def test_failed_reingest_keeps_previous_chunks(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    doc_path = data_dir / "operations_doc_tier1.json"
    doc_path.write_text('[{"key": "initial valid"}]', encoding="utf-8")

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_keep", embedding_function=FakeEmbedding())
    ingest_directory(data_dir, store)
    assert store.count() == 1

    # Corrupt the file
    doc_path.write_text('{"bad_json": [', encoding="utf-8")
    report = ingest_directory(data_dir, store)

    assert report.documents_ok == 0
    assert len(report.documents_failed) == 1
    # Previous chunk must remain intact!
    assert store.count() == 1


def test_unsupported_suffix_reported_at_discover_stage(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "document.docx").write_text("Word content", encoding="utf-8")

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_unsupp", embedding_function=FakeEmbedding())
    report = ingest_directory(data_dir, store)

    assert report.documents_ok == 0
    assert len(report.documents_failed) == 1
    failed = report.documents_failed[0]
    assert failed.file == "document.docx"
    assert failed.stage == "discover"


def test_unlisted_file_fails_closed_to_tier_4(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # No tier in filename, but valid department prefix
    (data_dir / "hr_notes.txt").write_text("Confidential salary notes.", encoding="utf-8")

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_failclose", embedding_function=FakeEmbedding())
    ingest_directory(data_dir, store)

    meta = store.collection.get()["metadatas"][0]
    assert meta["clearance_tier"] == 4


def test_restricted_chunks_never_leak(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "engineering_open_tier1.txt").write_text("Open doc.", encoding="utf-8")
    (data_dir / "finance_secret_tier4.txt").write_text("Secret M&A doc.", encoding="utf-8")

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_leak", embedding_function=FakeEmbedding())
    ingest_directory(data_dir, store)

    # User with Tier 2 clearance must NOT see Tier 4
    tier2_hits = store.query_by_tier_and_department(
        query_text="Secret",
        user_tier=2,
        user_departments=None
    )
    for h in tier2_hits:
        assert h.metadata["clearance_tier"] <= 2
        assert "finance_secret_tier4.txt" not in h.metadata["source_file"]

    # User with Tier 4 clearance CAN see Tier 4
    tier4_hits = store.query_by_tier_and_department(
        query_text="Secret",
        user_tier=4,
        user_departments=None
    )
    assert any(h.metadata["clearance_tier"] == 4 for h in tier4_hits)


def test_report_timings_are_consistent(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "engineering_doc_tier1.txt").write_text("Small doc content.", encoding="utf-8")

    store = VectorStore(path=tmp_path / "chroma", collection_name="test_timings", embedding_function=FakeEmbedding())
    report = ingest_directory(data_dir, store)

    assert report.extract_s >= 0.0
    assert report.chunk_s >= 0.0
    assert report.embed_s >= 0.0
    assert report.write_s >= 0.0
    assert report.total_s >= report.extract_s
    assert report.total_s >= report.chunk_s
    assert report.total_s >= (report.embed_s + report.write_s)

