from pathlib import Path
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.etl.pipeline import ingest_directory
from src.rag.engine import MultiTenantRAGEngine, UNIFORM_ABSTENTION_MESSAGE
from src.rag.models import UserPrincipal
from src.rag.rbac import API_KEY_REGISTRY, AuditLogger
from src.rag.reranker import LexicalReranker
from scripts.make_sample_data import generate_sample_corpus


def _build_client(tmp_path: Path, tmp_store, allow_dev_header: bool = False) -> TestClient:
    corpus_dir = tmp_path / "sample_documents"
    generate_sample_corpus(corpus_dir)
    ingest_directory(corpus_dir, tmp_store)
    engine = MultiTenantRAGEngine(
        store=tmp_store,
        reranker=LexicalReranker(),
        audit_logger=AuditLogger(),
    )
    app = create_app(engine=engine, allow_dev_persona_header=allow_dev_header)
    return TestClient(app)


def test_api_returns_401_for_missing_or_invalid_api_key(tmp_path: Path, tmp_store):
    client = _build_client(tmp_path, tmp_store)

    # Missing X-API-Key
    r_missing = client.post("/api/v1/rag/query", json={"query": "What is the password length?"})
    assert r_missing.status_code == 401

    # Invalid X-API-Key
    r_bad = client.post(
        "/api/v1/rag/query",
        headers={"X-API-Key": "forged-key"},
        json={"query": "What is the password length?"},
    )
    assert r_bad.status_code == 401


def test_api_rejects_client_supplied_clearance_tier_or_departments_with_422(tmp_path: Path, tmp_store):
    """
    Fix #1: Request body uses extra='forbid'. Any caller attempting to pass clearance_tier,
    departments, or tenant_id in JSON body receives HTTP 422.
    """
    client = _build_client(tmp_path, tmp_store)
    r = client.post(
        "/api/v1/rag/query",
        headers={"X-API-Key": "key-ops-tier1"},
        json={
            "query": "What is the Project Titan valuation?",
            "clearance_tier": 4,
            "departments": ["legal"],
        },
    )
    assert r.status_code == 422


def test_api_returns_403_when_authenticated_principal_has_empty_department_scope(
    tmp_path: Path, tmp_store, monkeypatch
):
    client = _build_client(tmp_path, tmp_store)
    monkeypatch.setitem(
        API_KEY_REGISTRY,
        "key-suspended",
        UserPrincipal(user_id="suspended_user", clearance_tier=1, departments=frozenset()),
    )
    r = client.post(
        "/api/v1/rag/query",
        headers={"X-API-Key": "key-suspended"},
        json={"query": "What is the password length?"},
    )
    assert r.status_code == 403


def test_api_authenticated_queries_enforce_rbac_and_return_citations(tmp_path: Path, tmp_store):
    client = _build_client(tmp_path, tmp_store)

    # Tier 1 user querying Tier 4 M&A -> 200 OK with uniform abstention
    r_t1 = client.post(
        "/api/v1/rag/query",
        headers={"X-API-Key": "key-ops-tier1"},
        json={"query": "What is the Project Titan purchase price?"},
    )
    assert r_t1.status_code == 200
    body_t1 = r_t1.json()
    assert body_t1["abstained"] is True
    assert body_t1["answer"] == UNIFORM_ABSTENTION_MESSAGE
    assert body_t1["citations"] == []

    # Tier 4 General Counsel querying Tier 4 M&A -> 200 OK with $184,500,000 and page 1 citation
    r_t4 = client.post(
        "/api/v1/rag/query",
        headers={"X-API-Key": "key-leg-tier4"},
        json={"query": "What is the Project Titan aggregate purchase price and founder buyout?"},
    )
    assert r_t4.status_code == 200
    body_t4 = r_t4.json()
    assert body_t4["abstained"] is False
    assert "$184,500,000" in body_t4["answer"]
    assert any(
        c["source_file"] == "merger_acquisition_audit_tier4.pdf" and c["page_number"] == 1
        for c in body_t4["citations"]
    )

