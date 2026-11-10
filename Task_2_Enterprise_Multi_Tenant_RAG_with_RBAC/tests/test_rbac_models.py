import pytest
from pydantic import ValidationError
from src.rag.models import UserPrincipal, Citation, AuditEvent
from src.rag.rbac import (
    PERSONAS,
    API_KEY_REGISTRY,
    authenticate_api_key,
    authorize_principal,
    is_chunk_authorized,
    assert_hits_authorized,
    build_citation_from_metadata,
    AuditLogger,
    AuthenticationError,
    AccessDeniedError,
    RBACViolationError,
)
from src.etl.vector_store import QueryHit


def test_six_persona_matrix_and_api_key_authentication():
    assert set(PERSONAS.keys()) == {
        "ops_associate",
        "cloud_engineer",
        "hr_specialist",
        "finance_director",
        "general_counsel",
        "external_tenant",
    }
    p = authenticate_api_key("key-eng-tier2")
    assert p.user_id == "cloud_engineer"
    assert p.clearance_tier == 2
    assert p.departments == frozenset({"engineering"})
    assert p.tenant_id == "acme_corp"

    with pytest.raises(AuthenticationError):
        authenticate_api_key(None)
    with pytest.raises(AuthenticationError):
        authenticate_api_key("invalid-token")


def test_principal_rejects_empty_departments_with_access_denied():
    p = UserPrincipal(user_id="no_dept", clearance_tier=2, departments=frozenset())
    with pytest.raises(AccessDeniedError):
        authorize_principal(p)


def test_is_chunk_authorized_enforces_tier1_public_and_tenant_boundary():
    eng_p = PERSONAS["cloud_engineer"]  # acme_corp, tier 2, {"engineering"}
    globex_p = PERSONAS["external_tenant"]  # globex_inc, tier 4, all depts

    # Tier 1 operations in acme_corp -> ALLOWED for engineering user via Tier-1 Public Rule
    meta_t1_ops = {"tenant_id": "acme_corp", "clearance_tier": 1, "department": "operations"}
    assert is_chunk_authorized(meta_t1_ops, eng_p) is True

    # Tier 2 HR in acme_corp -> DENIED for engineering user
    meta_t2_hr = {"tenant_id": "acme_corp", "clearance_tier": 2, "department": "hr"}
    assert is_chunk_authorized(meta_t2_hr, eng_p) is False

    # Tier 1 operations in acme_corp -> DENIED for external_tenant (globex_inc)
    assert is_chunk_authorized(meta_t1_ops, globex_p) is False


def test_post_retrieval_guard_raises_rbac_violation_error():
    eng_p = PERSONAS["cloud_engineer"]
    leaked_hit = QueryHit(
        chunk_id="leak_1",
        text="Restricted M&A valuation $184,500,000",
        distance=0.1,
        metadata={"tenant_id": "acme_corp", "clearance_tier": 4, "department": "legal"},
    )
    with pytest.raises(RBACViolationError):
        assert_hits_authorized([leaked_hit], eng_p)


def test_four_level_citation_locator_hierarchy():
    # 1. Page locator (PDF)
    c_pdf = build_citation_from_metadata(
        1,
        "c1",
        {
            "source_file": "security_compliance_tier1.pdf",
            "department": "operations",
            "clearance_tier": 1,
            "document_date": "2026-10-15",
            "page_number": 2,
            "chunk_index": 1,
        },
        0.85,
    )
    assert c_pdf.locator_type == "page"
    assert c_pdf.format_tag() == "[security_compliance_tier1.pdf, p. 2]"

    # 2. Row locator (CSV / JSON record)
    c_row = build_citation_from_metadata(
        2,
        "c2",
        {
            "source_file": "hr_compensation_bands_tier2.csv",
            "department": "hr",
            "clearance_tier": 2,
            "document_date": "2026-10-22",
            "row_id": 3,
            "chunk_index": 2,
        },
        0.80,
    )
    assert c_row.locator_type == "row"
    assert c_row.format_tag() == "[hr_compensation_bands_tier2.csv, row 3]"

    # 3. Section locator (Markdown)
    c_sec = build_citation_from_metadata(
        3,
        "c3",
        {
            "source_file": "engineering_cloud_vpc_runbook_tier2.md",
            "department": "engineering",
            "clearance_tier": 2,
            "document_date": "2026-10-20",
            "section_title": "1. Network Topology & Peering",
            "chunk_index": 1,
        },
        0.78,
    )
    assert c_sec.locator_type == "section"
    assert c_sec.format_tag() == "[engineering_cloud_vpc_runbook_tier2.md, §1. Network Topology & Peering]"

    # 4. Chunk fallback locator (TXT / JSON header / MD preamble)
    c_txt = build_citation_from_metadata(
        4,
        "c4",
        {
            "source_file": "legal_ip_license_agreement_tier4.txt",
            "department": "legal",
            "clearance_tier": 4,
            "document_date": "2026-10-28",
            "chunk_index": 0,
        },
        0.75,
    )
    assert c_txt.locator_type == "chunk"
    assert c_txt.locator_value == "1"
    assert c_txt.format_tag() == "[legal_ip_license_agreement_tier4.txt, chunk 1]"


def test_audit_logger_records_structured_events_without_restricted_existence_leak():
    audit = AuditLogger()
    eng_p = PERSONAS["cloud_engineer"]
    ev = audit.record(
        principal=eng_p,
        query="What is Project Titan valuation?",
        rewritten_query="Project Titan valuation",
        authorized_candidates=0,
        cited_sources=(),
        abstained=True,
        injection_detected=False,
        rbac_violation_blocked=False,
        total_ms=12.4,
    )
    assert isinstance(ev, AuditEvent)
    assert ev.user_id == "cloud_engineer"
    assert ev.tenant_id == "acme_corp"
    assert ev.abstained is True
    assert len(audit.events) == 1

