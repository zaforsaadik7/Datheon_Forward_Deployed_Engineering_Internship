import json
from pathlib import Path
from src.etl.pipeline import ingest_directory
from src.rag.engine import MultiTenantRAGEngine
from src.rag.eval import compute_allowed_files_from_manifest, run_rbac_evaluation
from src.rag.rbac import PERSONAS, AuditLogger
from src.rag.reranker import LexicalReranker
from src.rag.cli import main as cli_main
from scripts.make_sample_data import generate_sample_corpus


def test_manifest_ground_truth_allowed_files_matrix(tmp_path: Path):
    corpus_dir = tmp_path / "sample_documents"
    generate_sample_corpus(corpus_dir)
    manifest_path = corpus_dir / "manifest.yaml"

    assert compute_allowed_files_from_manifest(manifest_path, PERSONAS["ops_associate"]) == {
        "security_compliance_tier1.pdf",
        "operations_supplier_catalog_tier1.json",
    }
    assert compute_allowed_files_from_manifest(manifest_path, PERSONAS["cloud_engineer"]) == {
        "security_compliance_tier1.pdf",
        "operations_supplier_catalog_tier1.json",
        "engineering_cloud_vpc_runbook_tier2.md",
    }
    assert compute_allowed_files_from_manifest(manifest_path, PERSONAS["hr_specialist"]) == {
        "security_compliance_tier1.pdf",
        "operations_supplier_catalog_tier1.json",
        "hr_compensation_bands_tier2.csv",
    }
    assert compute_allowed_files_from_manifest(manifest_path, PERSONAS["finance_director"]) == {
        "security_compliance_tier1.pdf",
        "operations_supplier_catalog_tier1.json",
        "finance_q3_budget_audit_tier3.csv",
        "executive_ai_roadmap_tier3.md",
    }
    assert len(compute_allowed_files_from_manifest(manifest_path, PERSONAS["general_counsel"])) == 8
    assert compute_allowed_files_from_manifest(manifest_path, PERSONAS["external_tenant"]) == set()


def test_eval_suite_zero_leaks_and_negative_control_catches_leaks_when_rbac_disabled(
    tmp_path: Path, tmp_store, monkeypatch
):
    """
    Fix #9:
    1. With RBAC enabled, rbac_leaks == 0, abstention_accuracy == 1.0, and gold answer/locator
       accuracy == 1.0 judged against manifest.yaml ground truth.
    2. With RBAC disabled via a test-only unfiltered store wrapper, the SAME evaluation suite
       detects rbac_leaks > 0, proving the evaluation harness genuinely catches leaks.
    """
    corpus_dir = tmp_path / "sample_documents"
    generate_sample_corpus(corpus_dir)
    ingest_directory(corpus_dir, tmp_store)
    manifest_path = corpus_dir / "manifest.yaml"

    # 1. Normal RBAC-enforced run
    safe_engine = MultiTenantRAGEngine(
        store=tmp_store,
        reranker=LexicalReranker(),
        audit_logger=AuditLogger(),
    )
    safe_report = run_rbac_evaluation(safe_engine, manifest_path=manifest_path)
    assert safe_report["summary"]["total_cases"] == 19
    assert safe_report["summary"]["rbac_leaks"] == 0
    assert safe_report["summary"]["abstention_accuracy"] == 1.0
    assert safe_report["summary"]["answer_accuracy"] == 1.0
    assert safe_report["summary"]["citation_locator_accuracy"] == 1.0

    # 2. Negative-control run with a test-only unfiltered wrapper -> MUST catch leaks!
    class UnfilteredStoreWrapper:
        def query_by_tier_and_department(
            self,
            query_text: str,
            user_tier: int,
            user_departments=None,
            tenant_id: str = "acme_corp",
            top_k: int = 5,
        ):
            return tmp_store.query_by_tier_and_department(
                query_text=query_text,
                user_tier=4,
                user_departments=None,
                tenant_id="acme_corp",
                top_k=top_k,
            )

    monkeypatch.setattr("src.rag.engine.assert_hits_authorized", lambda hits, p: list(hits))
    unsafe_engine = MultiTenantRAGEngine(
        store=UnfilteredStoreWrapper(),  # type: ignore[arg-type]
        reranker=LexicalReranker(),
        audit_logger=AuditLogger(),
    )
    unsafe_report = run_rbac_evaluation(unsafe_engine, manifest_path=manifest_path)
    assert unsafe_report["summary"]["rbac_leaks"] > 0


def test_cli_personas_ask_and_eval_subcommands(tmp_path: Path, monkeypatch):
    corpus_dir = tmp_path / "sample_documents"
    chroma_dir = tmp_path / "chroma_cli"
    out_json = tmp_path / "eval_out.json"
    generate_sample_corpus(corpus_dir)

    rc_personas = cli_main(["personas"])
    assert rc_personas == 0

    rc_ingest = cli_main(
        [
            "ingest",
            "--data-dir",
            str(corpus_dir),
            "--chroma-path",
            str(chroma_dir),
            "--use-fake-embeddings",
        ]
    )
    assert rc_ingest == 0

    rc_ask = cli_main(
        [
            "ask",
            "--persona",
            "cloud_engineer",
            "--query",
            "What is the CIDR block of vpc-prod-compute?",
            "--chroma-path",
            str(chroma_dir),
            "--use-fake-embeddings",
            "--reranker",
            "lexical",
        ]
    )
    assert rc_ask == 0

    rc_eval = cli_main(
        [
            "eval",
            "--data-dir",
            str(corpus_dir),
            "--chroma-path",
            str(chroma_dir),
            "--out",
            str(out_json),
            "--use-fake-embeddings",
            "--reranker",
            "lexical",
        ]
    )
    assert rc_eval == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["summary"]["rbac_leaks"] == 0

