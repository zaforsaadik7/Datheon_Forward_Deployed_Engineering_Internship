"""Post-generation hallucination, numeric, entity, and citation-marker guardrails."""
import math
import re
from typing import List, Sequence, Set, Tuple
from .models import DraftAnswer, GuardrailReport, RankedChunk
from .reranker import _tokenize

MARKER_RE = re.compile(r"\[\d+\]")

# Mask out code identifiers and CIDR blocks before standalone number extraction
CODE_OR_CIDR_MASK_RE = re.compile(
    r"\b(?:\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?|24/7/365|[A-Za-z]+-[A-Za-z0-9\-]+|[A-Za-z]+\d+[A-Za-z0-9\-]*)\b"
)

STANDALONE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])([+-]?\s*\$?\s*\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)([MmKkBb%])?(?![A-Za-z0-9_])"
)

ENTITY_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:FIN-[A-Z0-9\-]+|SUP-\d+|Aegis-X|tgw-[a-z0-9]+|vpc-[a-z0-9\-]+|us-[a-z]+-\d+[a-z]?|FIDO2)\b"),
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?\b"),
    re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"),
)

MULTIPLIERS = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "b": 1_000_000_000.0,
}


def _extract_number_pairs(text: str) -> List[Tuple[str, float]]:
    cleaned = MARKER_RE.sub(" ", text)
    masked = CODE_OR_CIDR_MASK_RE.sub(" ", cleaned)
    pairs: List[Tuple[str, float]] = []

    for match in STANDALONE_NUMBER_RE.finditer(masked):
        raw_num = match.group(1)
        suffix = match.group(2) or ""
        raw_token = (raw_num + suffix).strip()

        normalized_str = raw_num.replace("$", "").replace(",", "").replace(" ", "")
        try:
            val = float(normalized_str)
        except ValueError:
            continue

        mult = MULTIPLIERS.get(suffix.lower(), 1.0)
        canonical_val = val * mult
        pairs.append((raw_token, canonical_val))

    return pairs


def extract_canonical_numbers(text: str) -> Set[float]:
    """
    Extract standalone numbers from text and normalize currency/commas/unit suffixes
    to canonical floats (e.g., '$450,000' -> 450000.0, '$14.5M' -> 14500000.0).
    """
    pairs = _extract_number_pairs(text)
    out: Set[float] = set()
    for _, val in pairs:
        out.add(val)
        out.add(abs(val))
    return out


def extract_key_entities(text: str) -> Set[str]:
    """
    Extract structured technical identifiers, cost-center codes, supplier IDs,
    CIDR blocks, and email addresses for entity-grounding verification.
    """
    cleaned = MARKER_RE.sub(" ", text)
    entities: Set[str] = set()
    for pat in ENTITY_PATTERNS:
        for m in pat.finditer(cleaned):
            entities.add(m.group(0))
    return entities


def _number_in_set(target: float, allowed: Set[float]) -> bool:
    for candidate in allowed:
        if math.isclose(target, candidate, rel_tol=1e-4, abs_tol=1e-4):
            return True
    return False


def verify_draft_answer(
    draft: DraftAnswer,
    ranked_chunks: Sequence[RankedChunk],
) -> GuardrailReport:
    """
    Verify every DraftStatement in DraftAnswer against its referenced RankedChunks:
    1. Every cited marker index [N] must be within 1..len(ranked_chunks).
    2. Every standalone number in the statement must match a canonical float in its cited chunks.
    3. Every structured entity/identifier in the statement must appear in its cited chunks.
    4. Every statement must share lexical content tokens with its cited chunks.
    """
    if not draft.statements:
        return GuardrailReport(
            passed=False,
            checked_sentences=0,
            verified_sentences=0,
            unverified_claims=(),
            hallucinated_numbers=(),
            hallucinated_entities=(),
            invalid_marker_indices=(),
        )

    num_chunks = len(ranked_chunks)
    checked = len(draft.statements)
    verified = 0

    unverified_claims: List[str] = []
    hallucinated_numbers: List[str] = []
    hallucinated_entities: List[str] = []
    invalid_markers: List[int] = []

    for stmt in draft.statements:
        stmt_ok = True
        valid_indices: List[int] = []

        if not stmt.cited_indices:
            stmt_ok = False
        else:
            for idx in stmt.cited_indices:
                if 1 <= idx <= num_chunks:
                    valid_indices.append(idx)
                else:
                    stmt_ok = False
                    if idx not in invalid_markers:
                        invalid_markers.append(idx)

        if not valid_indices:
            unverified_claims.append(stmt.text)
            continue

        combined_source_text = " \n ".join(ranked_chunks[i - 1].text for i in valid_indices)
        source_numbers = extract_canonical_numbers(combined_source_text)
        source_lower = combined_source_text.lower()

        # Check numbers
        for raw_tok, canonical_val in _extract_number_pairs(stmt.text):
            if not _number_in_set(canonical_val, source_numbers):
                stmt_ok = False
                if raw_tok not in hallucinated_numbers:
                    hallucinated_numbers.append(raw_tok)

        # Check structured entities
        for ent in extract_key_entities(stmt.text):
            if ent.lower() not in source_lower:
                stmt_ok = False
                if ent not in hallucinated_entities:
                    hallucinated_entities.append(ent)

        # Check lexical grounding overlap
        stmt_tokens = set(_tokenize(MARKER_RE.sub(" ", stmt.text)))
        source_tokens = set(_tokenize(combined_source_text))
        if stmt_tokens:
            overlap_ratio = len(stmt_tokens & source_tokens) / float(len(stmt_tokens))
            if overlap_ratio < 0.25:
                stmt_ok = False

        if stmt_ok:
            verified += 1
        else:
            unverified_claims.append(stmt.text)

    passed = (
        verified == checked
        and len(invalid_markers) == 0
        and len(hallucinated_numbers) == 0
        and len(hallucinated_entities) == 0
    )

    return GuardrailReport(
        passed=passed,
        checked_sentences=checked,
        verified_sentences=verified,
        unverified_claims=tuple(unverified_claims),
        hallucinated_numbers=tuple(hallucinated_numbers),
        hallucinated_entities=tuple(hallucinated_entities),
        invalid_marker_indices=tuple(invalid_markers),
    )
