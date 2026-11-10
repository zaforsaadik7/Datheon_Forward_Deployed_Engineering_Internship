import json

import pytest

from src.etl.benchmark import percentile, run_benchmark, write_report
from src.etl.synthetic import make_synthetic_corpus
from src.etl.manifest import load_manifest


def test_percentile_known_values():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([10], 99) == 10
    assert abs(percentile(list(range(1, 101)), 95) - 95.05) < 1e-9


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


@pytest.fixture(scope="module")
def tiny_report(tmp_path_factory):
    return run_benchmark(
        tmp_path_factory.mktemp("bench"),
        scales=(8,), batch_sizes=(16, 32), queries_per_scenario=10, embedder="fake",
    )


def test_report_schema(tiny_report):
    assert {"meta", "ingestion", "queries"} <= set(tiny_report)
    assert len(tiny_report["ingestion"]) == 2
    assert [q["scenario"] for q in tiny_report["queries"]] == [
        "no_filter", "tier_only", "tier_one_dept", "tier_three_depts"]
    for row in tiny_report["ingestion"]:
        assert row["documents"] == 8 and row["failed_documents"] == 0
        assert all(row[k] >= 0 for k in ("extract_s", "chunk_s", "embed_s", "write_s", "rss_peak_mb"))
    for q in tiny_report["queries"]:
        assert 0 <= q["p50_ms"] <= q["p95_ms"] <= q["p99_ms"]


def test_no_filter_violations(tiny_report):
    assert all(q["filter_violations"] == 0 for q in tiny_report["queries"])


def test_throughput_is_consistent(tiny_report):
    for r in tiny_report["ingestion"]:
        assert abs(r["docs_per_s"] - r["documents"] / r["total_s"]) <= 0.01 * r["docs_per_s"]


def test_report_json_roundtrip(tiny_report, tmp_path):
    out = tmp_path / "r" / "report.json"
    write_report(tiny_report, out)
    assert json.loads(out.read_text(encoding="utf-8")) == tiny_report


def test_synthetic_corpus_is_seeded(tmp_path):
    a = make_synthetic_corpus(tmp_path / "a", 12, seed=7).parent
    b = make_synthetic_corpus(tmp_path / "b", 12, seed=7).parent
    c = make_synthetic_corpus(tmp_path / "c", 12, seed=8).parent
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    assert all((a / n).read_bytes() == (b / n).read_bytes() for n in names)
    assert any((a / n).read_bytes() != (c / n).read_bytes() for n in names if (c / n).exists())


def test_synthetic_corpus_covers_all_tiers(tmp_path):
    manifest = make_synthetic_corpus(tmp_path, 30)
    entries = load_manifest(manifest)
    assert len(entries) == 30
    assert {e.clearance_tier for e in entries.values()} == {1, 2, 3, 4}
    assert all((tmp_path / name).exists() for name in entries)
