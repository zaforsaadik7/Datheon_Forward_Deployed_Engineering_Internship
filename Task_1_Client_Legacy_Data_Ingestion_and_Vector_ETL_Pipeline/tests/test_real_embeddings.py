"""Known-answer retrieval with the real embedding model. Run with: pytest -m real_embeddings"""
from pathlib import Path

import pytest

from src.etl.pipeline import ingest_directory
from src.etl.vector_store import VectorStore

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_documents"
pytestmark = pytest.mark.real_embeddings


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = VectorStore(tmp_path_factory.mktemp("real_db"))  # default Chroma embedder
    report = ingest_directory(SAMPLE, s)
    assert report.documents_ok == 8 and not report.documents_failed
    return s


def test_vpc_question_finds_engineering_runbook(store):
    hits = store.query_by_tier_and_department("how do I configure VPC peering route tables", 2, None, top_k=3)
    assert hits[0].metadata["source_file"] == "engineering_cloud_vpc_runbook_tier2.md"


def test_ip_license_hidden_below_tier_4_and_first_at_tier_4(store):
    q = "intellectual property license royalty terms"
    low = store.query_by_tier_and_department(q, 2, None, top_k=20)
    assert all(h.metadata["source_file"] != "legal_ip_license_agreement_tier4.txt" for h in low)
    top = store.query_by_tier_and_department(q, 4, None, top_k=3)
    assert top[0].metadata["source_file"] in {
        "legal_ip_license_agreement_tier4.txt", "merger_acquisition_audit_tier4.pdf"}


def test_vendor_payment_finds_finance_csv_row(store):
    hits = store.query_by_tier_and_department("GPU cluster training reservation payment AWS", 3, ["finance"], top_k=3)
    assert hits[0].metadata["source_file"] == "finance_q3_budget_audit_tier3.csv"
    assert "row_id" in hits[0].metadata


def test_cap_table_question_cites_pdf_page(store):
    hits = store.query_by_tier_and_department("acquisition valuation antitrust covenant", 4, ["legal"], top_k=3)
    assert hits[0].metadata["source_file"].endswith(".pdf")
    assert "page_number" in hits[0].metadata
