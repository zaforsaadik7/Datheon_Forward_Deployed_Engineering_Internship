from src.rag.models import DraftAnswer, DraftStatement, RankedChunk
from src.rag.guardrails import (
    extract_canonical_numbers,
    extract_key_entities,
    verify_draft_answer,
)


def _chunk(idx: int, text: str, source_file: str = "doc_tier1.pdf") -> RankedChunk:
    return RankedChunk(
        chunk_id=f"c{idx}",
        text=text,
        vector_distance=0.2,
        vector_similarity=0.8,
        rerank_score=0.75,
        metadata={"source_file": source_file, "clearance_tier": 1, "department": "operations"},
    )


def test_canonical_numeric_normalization_matches_csv_digits_and_formatted_dollars():
    """
    Fix #4: CSV holds '450000' while prose says '$450,000'. Both must normalize to 450000.0.
    Also '$14.5M' and '$14,500,000' both normalize to 14500000.0.
    """
    assert 450000.0 in extract_canonical_numbers("q3_allocated_usd: 450000")
    assert 450000.0 in extract_canonical_numbers("The allocated budget is $450,000.")
    assert 14500000.0 in extract_canonical_numbers("Approved CapEx of $14.5M.")
    assert 14500000.0 in extract_canonical_numbers("Total budget ($14,500,000).")


def test_guardrail_verifies_statement_against_cited_chunk_without_tag_digit_false_positives():
    """
    Fix #4: Guardrail checks DraftStatement.text BEFORE citation tags like
    '[security_compliance_tier1.pdf, p. 2]' are rendered, so '1' and '2' in the locator
    never trigger false numeric hallucinations.
    """
    chunks = [
        _chunk(
            1,
            "cost_center_id: FIN-Q3-002 | category: Enterprise SaaS & Snowflake | q3_allocated_usd: 450000 | q3_actual_usd: 480000",
            source_file="finance_q3_budget_audit_tier3.csv",
        )
    ]
    draft = DraftAnswer(
        statements=(
            DraftStatement(
                text="Cost center FIN-Q3-002 (Enterprise SaaS & Snowflake) had an allocated budget of $450,000 and actual spend of $480,000.",
                cited_indices=(1,),
            ),
        )
    )
    report = verify_draft_answer(draft, chunks)
    assert report.passed is True
    assert report.hallucinated_numbers == ()
    assert report.hallucinated_entities == ()


def test_guardrail_rejects_fabricated_numbers_entities_and_out_of_range_markers():
    chunks = [
        _chunk(
            1,
            "Standard user passwords must be at least 16 characters in length.",
            source_file="security_compliance_tier1.pdf",
        )
    ]
    # Dishonest generator invents '24 characters', entity 'FIN-Q3-999', and invalid marker 5
    dishonest_draft = DraftAnswer(
        statements=(
            DraftStatement(
                text="Cost center FIN-Q3-999 requires passwords of 24 characters.",
                cited_indices=(1, 5),
            ),
        )
    )
    report = verify_draft_answer(dishonest_draft, chunks)
    assert report.passed is False
    assert 5 in report.invalid_marker_indices
    assert any("24" in n for n in report.hallucinated_numbers)
    assert "FIN-Q3-999" in report.hallucinated_entities

