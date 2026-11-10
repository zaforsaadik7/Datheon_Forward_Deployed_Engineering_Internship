from pathlib import Path
import pytest
from src.etl.pipeline import ingest_directory
from src.etl.vector_store import VectorStore
from src.rag.engine import MultiTenantRAGEngine, UNIFORM_ABSTENTION_MESSAGE
from src.rag.rbac import PERSONAS
from src.rag.reranker import OnnxCrossEncoder
from scripts.make_sample_data import generate_sample_corpus


@pytest.mark.real_models
def test_end_to_end_real_embeddings_and_onnx_cross_encoder(tmp_path: Path):
    corpus_dir = tmp_path / "sample_documents"
    generate_sample_corpus(corpus_dir)

    # Real Chroma default ONNX MiniLM-L6-v2 embeddings
    store = VectorStore(path=tmp_path / "chroma_real", collection_name="real_enterprise_docs")
    report = ingest_directory(corpus_dir, store)
    assert report.documents_ok == 8

    # Real ONNX MS-MARCO MiniLM cross-encoder reranker
    engine = MultiTenantRAGEngine(
        store=store,
        reranker=OnnxCrossEncoder(),
    )

    # 1. Tier 1 password query by cloud_engineer (engineering) -> cites page 2 of security_compliance_tier1.pdf
    r_sec = engine.answer(
        query="What is the minimum required password length for standard user passwords?",
        principal=PERSONAS["cloud_engineer"],
    )
    assert r_sec.abstained is False
    assert "16 characters" in r_sec.answer
    assert any(
        c.source_file == "security_compliance_tier1.pdf" and c.page_number == 2
        for c in r_sec.citations
    )

    # 2. Tier 4 M&A query by cloud_engineer (Tier 2) -> uniform abstention, zero leaks
    r_denied = engine.answer(
        query="What is the Project Titan purchase price and founder buyout?",
        principal=PERSONAS["cloud_engineer"],
    )
    assert r_denied.abstained is True
    assert r_denied.answer == UNIFORM_ABSTENTION_MESSAGE
    assert r_denied.citations == ()

    # 3. Tier 4 M&A query by general_counsel (Tier 4) -> answers $184,500,000 and cites page 1
    r_allowed = engine.answer(
        query="What is the Project Titan aggregate purchase price and founder buyout?",
        principal=PERSONAS["general_counsel"],
    )
    assert r_allowed.abstained is False
    assert "$184,500,000" in r_allowed.answer
    assert any(
        c.source_file == "merger_acquisition_audit_tier4.pdf" and c.page_number == 1
        for c in r_allowed.citations
    )

