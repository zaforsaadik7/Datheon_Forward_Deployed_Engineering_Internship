import re
from pathlib import Path

from src.etl.manifest import load_manifest
from src.etl.pipeline import ingest_directory, discover_files

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_documents"


def test_manifest_covers_every_sample_file():
    manifest = load_manifest(SAMPLE_DIR / "manifest.yaml")
    files = {p.name for p in discover_files(SAMPLE_DIR)}
    assert len(files) == 8
    assert files == set(manifest)


def test_sample_set_covers_formats_and_tiers():
    manifest = load_manifest(SAMPLE_DIR / "manifest.yaml")
    suffixes = sorted(Path(name).suffix for name in manifest)
    assert suffixes == [".csv", ".csv", ".json", ".md", ".md", ".pdf", ".pdf", ".txt"]
    assert {e.clearance_tier for e in manifest.values()} == {1, 2, 3, 4}


def test_filename_tier_matches_manifest_tier():
    manifest = load_manifest(SAMPLE_DIR / "manifest.yaml")
    for name, entry in manifest.items():
        found = re.search(r"_tier(\d)\.", name)
        assert found and int(found.group(1)) == entry.clearance_tier, name


def test_sample_set_ingests_cleanly(tmp_store):
    report = ingest_directory(SAMPLE_DIR, tmp_store)
    assert report.documents_ok == 8
    assert report.documents_failed == []
    assert tmp_store.count() == report.chunks_written > 0
    # restricted documents carry their tier into the index
    hits = tmp_store.query_by_tier_and_department("agreement", user_tier=1, user_departments=None, top_k=100)
    assert hits and all(h.metadata["clearance_tier"] == 1 for h in hits)
