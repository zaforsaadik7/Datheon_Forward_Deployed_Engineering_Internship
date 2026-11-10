import json
from pathlib import Path

from src.etl.cli import main

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_documents"


def _ingest(db, data_dir=SAMPLE):
    return main(["ingest", "--data-dir", str(data_dir), "--db-path", str(db), "--embedder", "fake"])


def test_ingest_then_query_respects_tier(tmp_path, capsys):
    db = tmp_path / "db"
    assert _ingest(db) == 0
    capsys.readouterr()
    code = main(["query", "agreement", "--tier", "1", "--top-k", "50",
                 "--db-path", str(db), "--embedder", "fake"])
    out = capsys.readouterr().out
    assert code == 0
    assert "operations_supplier_catalog_tier1.json" in out
    for restricted in ("legal_ip_license_agreement_tier4.txt", "merger_acquisition_audit_tier4.pdf",
                       "finance_q3_budget_audit_tier3.csv"):
        assert restricted not in out


def test_ingest_exit_code_1_on_failed_document(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "finance_ok_tier2.txt").write_text("Budget approved for the quarter.", encoding="utf-8")
    (data / "finance_broken_tier2.json").write_text('{"a": ', encoding="utf-8")
    assert _ingest(tmp_path / "db", data) == 1


def test_query_invalid_tier_exits_2(tmp_path, capsys):
    db = tmp_path / "db"
    _ingest(db)
    capsys.readouterr()
    assert main(["query", "x", "--tier", "9", "--db-path", str(db), "--embedder", "fake"]) == 2
    assert "tier" in capsys.readouterr().err.lower()


def test_reset_empties_collection(tmp_path):
    from src.etl.benchmark import HashEmbedding
    from src.etl.vector_store import VectorStore
    db = tmp_path / "db"
    _ingest(db)
    assert main(["reset", "--db-path", str(db), "--embedder", "fake"]) == 0
    assert VectorStore(db, embedding_function=HashEmbedding()).count() == 0


def test_benchmark_writes_json(tmp_path):
    out = tmp_path / "bench.json"
    code = main(["benchmark", "--scales", "4", "--batch-sizes", "8", "--queries", "5",
                 "--embedder", "fake", "--out", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["ingestion"]) == 1 and len(data["queries"]) == 4
