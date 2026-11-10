from pathlib import Path
from src.config import get_settings, DEFAULT_TENANT, TIERS, DEPARTMENTS
from src.etl.extractors import get_extractor
from src.etl.pipeline import ingest_directory
from src.etl.vector_store import VectorStore
from scripts.make_sample_data import generate_sample_corpus


def test_settings_defaults():
    s = get_settings()
    assert s.chunk_size == 900
    assert s.chunk_overlap == 120
    assert s.collection_name == "enterprise_docs"
    assert DEFAULT_TENANT == "acme_corp"
    assert TIERS == (1, 2, 3, 4)
    assert DEPARTMENTS == frozenset({
        "engineering", "finance", "legal", "hr", "executive", "operations"
    })


def test_sample_generator_preserves_dollar_amounts_and_backticks(tmp_path: Path):
    out = tmp_path / "sample_documents"
    generate_sample_corpus(out)

    # Check PDF 1: security_compliance_tier1.pdf
    sec_segs = get_extractor(out / "security_compliance_tier1.pdf").extract(
        out / "security_compliance_tier1.pdf"
    )
    assert len(sec_segs) == 3
    assert "FIDO2" in sec_segs[0].text and sec_segs[0].page_number == 1
    assert "16 characters" in sec_segs[1].text and sec_segs[1].page_number == 2
    assert "soc-alerts@enterprise.internal" in sec_segs[2].text and sec_segs[2].page_number == 3

    # Check PDF 2: merger_acquisition_audit_tier4.pdf
    mna_segs = get_extractor(out / "merger_acquisition_audit_tier4.pdf").extract(
        out / "merger_acquisition_audit_tier4.pdf"
    )
    assert len(mna_segs) == 2
    assert "$184,500,000" in mna_segs[0].text and "$92,000,000" in mna_segs[0].text
    assert "$18,450,000" in mna_segs[1].text and "$22,000,000" in mna_segs[1].text

    # Check Markdown & TXT files
    vpc_text = (out / "engineering_cloud_vpc_runbook_tier2.md").read_text(encoding="utf-8")
    assert "`vpc-prod-compute`" in vpc_text
    exec_text = (out / "executive_ai_roadmap_tier3.md").read_text(encoding="utf-8")
    assert "$14.5M" in exec_text and "$7.2M" in exec_text
    ip_text = (out / "legal_ip_license_agreement_tier4.txt").read_text(encoding="utf-8")
    assert "$3,500,000" in ip_text


def test_vector_store_tier1_public_rule_and_tenant_isolation(tmp_store, chunk_factory):
    chunks = [
        # Tier 1 operations in acme_corp -> readable by any department in acme_corp
        chunk_factory(chunk_id="c_t1_ops", department="operations", clearance_tier=1, tenant_id="acme_corp"),
        # Tier 2 engineering in acme_corp -> readable by engineering Tier 2+
        chunk_factory(chunk_id="c_t2_eng", department="engineering", clearance_tier=2, tenant_id="acme_corp"),
        # Tier 2 hr in acme_corp -> NOT readable by engineering-only user
        chunk_factory(chunk_id="c_t2_hr", department="hr", clearance_tier=2, tenant_id="acme_corp"),
        # Tier 3 engineering in acme_corp -> NOT readable by Tier 2 user
        chunk_factory(chunk_id="c_t3_eng", department="engineering", clearance_tier=3, tenant_id="acme_corp"),
        # Tier 1 operations in globex_inc -> NOT readable by acme_corp user
        chunk_factory(chunk_id="c_globex_t1", department="operations", clearance_tier=1, tenant_id="globex_inc"),
    ]
    tmp_store.upsert_chunks(chunks)

    hits = tmp_store.query_by_tier_and_department(
        query_text="Sample chunk content",
        user_tier=2,
        user_departments=["engineering"],
        tenant_id="acme_corp",
        top_k=20,
    )
    hit_ids = {h.chunk_id for h in hits}
    assert hit_ids == {"c_t1_ops", "c_t2_eng"}
