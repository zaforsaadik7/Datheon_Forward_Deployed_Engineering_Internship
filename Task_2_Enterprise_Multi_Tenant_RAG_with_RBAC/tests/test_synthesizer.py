from src.rag.models import RankedChunk
from src.rag.synthesizer import ExtractiveSynthesizer, render_cited_answer


def test_synthesizer_deduplicates_overlapping_sentences_and_renders_citations():
    c1 = RankedChunk(
        chunk_id="c1",
        text="Section 2: Data Classification.\nStandard user passwords must be at least 16 characters in length and rotated immediately upon any suspected compromise.",
        vector_distance=0.15,
        vector_similarity=0.85,
        rerank_score=0.88,
        metadata={
            "source_file": "security_compliance_tier1.pdf",
            "department": "operations",
            "clearance_tier": 1,
            "document_date": "2026-10-15",
            "page_number": 2,
            "chunk_index": 1,
        },
    )
    # Overlapping chunk repeating the same sentence
    c2 = RankedChunk(
        chunk_id="c2",
        text="Standard user passwords must be at least 16 characters in length and rotated immediately upon any suspected compromise. Session idle lockout is enforced at 15 minutes.",
        vector_distance=0.20,
        vector_similarity=0.80,
        rerank_score=0.72,
        metadata={
            "source_file": "security_compliance_tier1.pdf",
            "department": "operations",
            "clearance_tier": 1,
            "document_date": "2026-10-15",
            "page_number": 2,
            "chunk_index": 2,
        },
    )
    synth = ExtractiveSynthesizer(max_sentences=2)
    draft = synth.generate("What is the minimum password length?", [c1, c2])
    assert len(draft.statements) >= 1
    # Ensure duplicate sentence is not emitted twice
    texts = [s.text for s in draft.statements]
    assert len(texts) == len(set(texts))

    answer_text, citations = render_cited_answer(draft, [c1, c2])
    assert "16 characters" in answer_text
    assert "[security_compliance_tier1.pdf, p. 2]" in answer_text
    assert len(citations) >= 1
    assert citations[0].page_number == 2


def test_synthesizer_handles_pipe_delimited_csv_records_and_row_citations():
    c_csv = RankedChunk(
        chunk_id="c_hr_3",
        text="level: L5 | role_title: Staff Software Engineer | department: Engineering | base_min_usd: 185000 | base_midpoint_usd: 210000 | base_max_usd: 240000 | target_bonus_pct: 20% | equity_refresh_usd: 95000",
        vector_distance=0.18,
        vector_similarity=0.82,
        rerank_score=0.85,
        metadata={
            "source_file": "hr_compensation_bands_tier2.csv",
            "department": "hr",
            "clearance_tier": 2,
            "document_date": "2026-10-22",
            "row_id": 3,
            "chunk_index": 2,
        },
    )
    synth = ExtractiveSynthesizer(max_sentences=2)
    draft = synth.generate("What is the L5 Staff Software Engineer base salary range?", [c_csv])
    answer_text, citations = render_cited_answer(draft, [c_csv])
    assert "185000" in answer_text and "240000" in answer_text
    assert "[hr_compensation_bands_tier2.csv, row 3]" in answer_text
    assert citations[0].row_id == 3
