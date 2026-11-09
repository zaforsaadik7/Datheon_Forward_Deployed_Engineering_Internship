from pathlib import Path
import pytest
from src.etl.manifest import load_manifest, resolve_metadata, MetadataError
from src.etl.models import ManifestEntry


def test_manifest_beats_filename(tmp_path: Path):
    manifest_yaml = """
documents:
  - file: legal_x_tier4.txt
    department: legal
    clearance_tier: 2
    document_date: 2026-05-01
"""
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(manifest_yaml, encoding="utf-8")

    manifest = load_manifest(manifest_file)
    entry = resolve_metadata("legal_x_tier4.txt", manifest, mtime=1760000000.0)

    assert entry.department == "legal"
    assert entry.clearance_tier == 2
    assert entry.metadata_source == "manifest"
    assert entry.document_date == "2026-05-01"


def test_filename_fallback():
    manifest = {}
    mtime = 1770000000.0  # Approx 2026-02-02
    entry = resolve_metadata("finance_q3_audit_tier3.csv", manifest, mtime=mtime)

    assert entry.department == "finance"
    assert entry.clearance_tier == 3
    assert entry.metadata_source == "filename"
    assert len(entry.document_date) == 10
    assert "-" in entry.document_date


def test_missing_tier_fails_closed():
    manifest = {}
    entry = resolve_metadata("hr_notes.csv", manifest, mtime=1760000000.0)
    assert entry.department == "hr"
    assert entry.clearance_tier == 4
    assert entry.metadata_source == "filename"


def test_unknown_department_raises_metadata_error():
    manifest = {}
    with pytest.raises(MetadataError):
        resolve_metadata("report_final.csv", manifest, mtime=1760000000.0)


def test_loader_rejects_bad_tier_naming_file(tmp_path: Path):
    manifest_yaml = """
documents:
  - file: bad_tier_file.txt
    department: engineering
    clearance_tier: 9
    document_date: 2026-01-01
"""
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(manifest_yaml, encoding="utf-8")

    with pytest.raises(MetadataError) as exc_info:
        load_manifest(manifest_file)

    assert "bad_tier_file.txt" in str(exc_info.value)
