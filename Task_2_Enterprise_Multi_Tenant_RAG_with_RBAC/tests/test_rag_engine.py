from pathlib import Path
from typing import List, Optional, Sequence
import pytest
from src.etl.pipeline import ingest_directory
from src.etl.vector_store import QueryHit
from src.rag.engine import MultiTenantRAGEngine, UNIFORM_ABSTENTION_MESSAGE
from src.rag.models import DraftAnswer, DraftStatement, RankedChunk
from src.rag.rbac import PERSONAS, AuditLogger
from src.rag.reranker import LexicalReranker
from scripts.make_sample_data import generate_sample_corpus


@pytest.fixture
def populated_engine(tmp_path: Path, tmp_store) -> MultiTenantRAGEngine:
    corpus_dir = tmp_path / "sample_documents"
    generate_sample_corpus(corpus_dir)
    report = ingest_directory(corpus_dir, tmp_store)
    assert report.documents_ok == 8
    return MultiTenantRAGEngine(
        store=tmp_store,
        reranker=LexicalReranker(),
        audit_logger=AuditLogger(),
    )


def test_tier1_public_document_accessible_across_departments_with_page_citation(populated_engine):
    # cloud_engineer is in {"engineering"}, NOT {"operations"}, but Tier 1 is public inside acme_corp
    eng = PERSONAS["cloud_engineer"]
    resp = populated_engine.answer(
        query="What is the minimum required password length?",
        principal=eng,
    )
    assert resp.abstained is False
    assert "16 characters" in resp.answer
    assert any(
        c.source_file == "security_compliance_tier1.pdf" and c.page_number == 2
        for c in resp.citations
    )


def test_unauthorized_query_and_nonexistent_topic_produce_identical_abstention(populated_engine):
    """
    Existence-leak prevention: an unauthorized query for Tier-4 Project Titan and a query
    for a completely nonexistent topic must return the exact same abstention message,
    abstention_reason, and zero citations.
    """
    eng = PERSONAS["cloud_engineer"]
    resp_restricted = populated_engine.answer(
        query="What is the Project Titan M&A purchase price and founder buyout?",
        principal=eng,
    )
    resp_nonexistent = populated_engine.answer(
        query="What is the Martian colony launch schedule for 2099?",
        principal=eng,
    )
    assert resp_restricted.abstained is True
    assert resp_nonexistent.abstained is True
    assert resp_restricted.answer == UNIFORM_ABSTENTION_MESSAGE
    assert resp_nonexistent.answer == UNIFORM_ABSTENTION_MESSAGE
    assert resp_restricted.abstention_reason == "no_authorized_results"
    assert resp_nonexistent.abstention_reason == "no_authorized_results"
    assert resp_restricted.citations == ()
    assert resp_nonexistent.citations == ()


def test_external_tenant_cannot_read_acme_corp_tier1_documents(populated_engine):
    globex = PERSONAS["external_tenant"]
    resp = populated_engine.answer(
        query="What is the minimum required password length?",
        principal=globex,
    )
    assert resp.abstained is True
    assert resp.answer == UNIFORM_ABSTENTION_MESSAGE
    assert resp.citations == ()


def test_post_retrieval_rbac_violation_is_caught_logged_and_uniformly_abstained():
    class FaultyUnfilteredStore:
        def query_by_tier_and_department(
            self,
            query_text: str,
            user_tier: int,
            user_departments: Optional[Sequence[str]] = None,
            tenant_id: str = "acme_corp",
            top_k: int = 5,
        ) -> List[QueryHit]:
            # Buggy store ignores pre-filter and returns a Tier 4 Legal document to Tier 1 user
            return [
                QueryHit(
                    chunk_id="leaked_t4",
                    text="Project Titan aggregate purchase price is $184,500,000.",
                    distance=0.10,
                    metadata={
                        "source_file": "merger_acquisition_audit_tier4.pdf",
                        "department": "legal",
                        "clearance_tier": 4,
                        "document_date": "2026-10-28",
                        "tenant_id": "acme_corp",
                        "page_number": 1,
                    },
                )
            ]

    audit = AuditLogger()
    engine = MultiTenantRAGEngine(
        store=FaultyUnfilteredStore(),  # type: ignore[arg-type]
        reranker=LexicalReranker(),
        audit_logger=audit,
    )
    resp = engine.answer(
        query="What is the Project Titan purchase price?",
        principal=PERSONAS["ops_associate"],
    )
    assert resp.abstained is True
    assert resp.answer == UNIFORM_ABSTENTION_MESSAGE
    assert resp.abstention_reason == "no_authorized_results"
    assert resp.citations == ()
    assert audit.events[-1].rbac_violation_blocked is True


def test_dishonest_generator_is_rejected_by_guardrail_and_abstains_uniformly(populated_engine):
    class DishonestGenerator:
        def generate(self, query: str, ranked_chunks: Sequence[RankedChunk]) -> DraftAnswer:
            return DraftAnswer(
                statements=(
                    DraftStatement(
                        text="Standard user passwords must be at least 99 characters in length.",
                        cited_indices=(1,),
                    ),
                )
            )

    populated_engine.generator = DishonestGenerator()
    resp = populated_engine.answer(
        query="What is the minimum required password length?",
        principal=PERSONAS["ops_associate"],
    )
    assert resp.abstained is True
    assert resp.answer == UNIFORM_ABSTENTION_MESSAGE
    assert resp.abstention_reason == "no_authorized_results"
    assert resp.guardrails.passed is True
    assert resp.guardrails.checked_sentences == 0
    assert resp.citations == ()
    assert populated_engine.audit_logger.events[-1].authorized_candidates > 0

