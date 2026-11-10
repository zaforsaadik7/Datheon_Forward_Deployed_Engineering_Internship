import pytest
from src.etl.vector_store import QueryHit
from src.rag.reranker import LexicalReranker, OnnxCrossEncoder


def test_lexical_reranker_reorders_overfetched_candidates_above_cosine():
    reranker = LexicalReranker()
    # Hit 1 has slightly smaller vector distance (0.20), but is unrelated to the exact identifier
    h_generic = QueryHit(
        chunk_id="c_generic",
        text="General cloud networking overview and subnet architecture guidelines.",
        distance=0.20,
        metadata={"source_file": "generic.md", "clearance_tier": 2, "department": "engineering"},
    )
    # Hit 2 has slightly larger vector distance (0.28), but exact phrase and identifier match
    h_exact = QueryHit(
        chunk_id="c_exact",
        text="Production workloads run inside `vpc-prod-compute` with CIDR block 10.42.0.0/16.",
        distance=0.28,
        metadata={"source_file": "runbook.md", "clearance_tier": 2, "department": "engineering"},
    )
    ranked = reranker.rerank(
        query="What is the CIDR block of vpc-prod-compute?",
        candidates=[h_generic, h_exact],
        top_k=2,
        min_score=0.10,
    )
    assert len(ranked) == 2
    assert ranked[0].chunk_id == "c_exact"
    assert ranked[0].rerank_score > ranked[1].rerank_score


def test_lexical_reranker_statistics_depend_only_on_authorized_candidates():
    """
    Fix #7: BM25 IDF statistics must be computed strictly over the candidate list
    passed to rerank(), never from a global corpus index that could leak restricted terms.
    """
    r1 = LexicalReranker()
    r2 = LexicalReranker()
    candidates = [
        QueryHit(
            chunk_id="c1",
            text="Standard user passwords must be at least 16 characters in length.",
            distance=0.25,
            metadata={"source_file": "sec.pdf", "clearance_tier": 1, "department": "operations"},
        ),
        QueryHit(
            chunk_id="c2",
            text="Incident reports go to soc-alerts@enterprise.internal within 15 minutes.",
            distance=0.35,
            metadata={"source_file": "sec.pdf", "clearance_tier": 1, "department": "operations"},
        ),
    ]
    out1 = r1.rerank("minimum password length", candidates, top_k=2)
    out2 = r2.rerank("minimum password length", candidates, top_k=2)
    assert [c.rerank_score for c in out1] == [c.rerank_score for c in out2]


@pytest.mark.real_models
def test_onnx_cross_encoder_scores_query_document_pairs():
    reranker = OnnxCrossEncoder()
    candidates = [
        QueryHit(
            chunk_id="c_wrong",
            text="The cafeteria serves lunch from 11:30 AM to 2:00 PM on weekdays.",
            distance=0.30,
            metadata={"source_file": "a.pdf", "clearance_tier": 1, "department": "operations"},
        ),
        QueryHit(
            chunk_id="c_right",
            text="Standard user passwords must be at least 16 characters in length and rotated upon compromise.",
            distance=0.35,
            metadata={"source_file": "b.pdf", "clearance_tier": 1, "department": "operations"},
        ),
    ]
    ranked = reranker.rerank(
        query="What is the minimum required password length?",
        candidates=candidates,
        top_k=2,
        min_score=0.05,
    )
    assert len(ranked) >= 1
    assert ranked[0].chunk_id == "c_right"
    assert ranked[0].rerank_score > 0.5

