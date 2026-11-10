"""Deterministic query rewriter with acronym expansion, filler stripping, and advisory injection telemetry."""
from dataclasses import dataclass
import re
from typing import Dict, List, Tuple

ACRONYM_MAP: Dict[str, str] = {
    "MFA": "multi-factor authentication",
    "SOC": "Security Operations Center",
    "SSO": "single sign-on",
    "MDM": "Mobile Device Management",
    "VPC": "Virtual Private Cloud",
    "CIDR": "Classless Inter-Domain Routing",
    "BGP": "Border Gateway Protocol",
    "BFD": "bidirectional forwarding detection",
    "SLA": "service level agreement",
    "ARR": "annual recurring revenue",
    "M&A": "Merger & Acquisition",
    "CAPEX": "capital expenditure",
    "EBITDA": "earnings before interest, taxes, depreciation, and amortization",
    "SWE": "Software Engineer",
}

INJECTION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions\b(?:\s+and\b)?", re.IGNORECASE),
    re.compile(r"\bbypass\s+(?:rbac|clearance|security|tier)\b(?:\s+and\b)?", re.IGNORECASE),
    re.compile(r"\boverride\s+(?:clearance|tier|permissions|access)\b(?:\s+and\b)?", re.IGNORECASE),
    re.compile(r"\bgrant\s+(?:me\s+)?tier\s*[1-4](?:\s+access)?\b(?:\s*[;,]|\s+and\b)?", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
)

FILLER_PREFIX_RE = re.compile(
    r"^(?:(?:hey|hi|hello)\b[\s,]*)*(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(?:(?:tell|show|explain(?:\s+to)?|give)\s+me\s+)?",
    re.IGNORECASE,
)

COMPOUND_SPLIT_RE = re.compile(
    r"\s*(?:;|\band\s+(?=(?:what|how|who|where|which|when)\b))\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RewriteResult:
    """Output of query rewriting, sub-query decomposition, and advisory telemetry."""
    original_query: str
    rewritten_query: str
    sub_queries: Tuple[str, ...]
    expanded_terms: Tuple[str, ...]
    injection_detected: bool


def _clean_single_clause(clause: str) -> Tuple[str, List[str]]:
    cleaned = FILLER_PREFIX_RE.sub("", clause.strip()).strip(" ?.,;:")
    expanded_terms: List[str] = []

    for acronym, expansion in ACRONYM_MAP.items():
        pattern = re.compile(rf"(?<!\w){re.escape(acronym)}(?!\w)", re.IGNORECASE)
        if pattern.search(cleaned) and expansion.lower() not in cleaned.lower():
            expanded_terms.append(f"{acronym} ({expansion})")
            cleaned = pattern.sub(f"{acronym} ({expansion})", cleaned, count=1)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, expanded_terms


def rewrite_query(raw_query: str) -> RewriteResult:
    """
    Normalize a raw user query for multi-tenant hybrid retrieval:
    - Raises ValueError on empty or whitespace-only queries
    - Detects prompt-injection phrases as advisory telemetry (without false-positiving on 'as root cause')
    - Strips conversational filler prefixes
    - Expands enterprise acronyms inline
    - Splits compound questions into up to 3 sub-queries
    """
    if raw_query is None or not raw_query.strip():
        raise ValueError("Query must not be empty.")

    working = raw_query.strip()
    injection_detected = False

    for pat in INJECTION_PATTERNS:
        if pat.search(working):
            injection_detected = True
            working = pat.sub(" ", working)

    working = re.sub(r"^[\s;,\-:]+", "", working)
    working = re.sub(r"\s+", " ", working).strip()
    if not working:
        working = raw_query.strip()

    raw_clauses = [c.strip() for c in COMPOUND_SPLIT_RE.split(working) if c.strip()]
    if not raw_clauses:
        raw_clauses = [working]

    sub_queries: List[str] = []
    all_expanded: List[str] = []

    for clause in raw_clauses[:3]:
        rewritten_clause, exp = _clean_single_clause(clause)
        if rewritten_clause:
            sub_queries.append(rewritten_clause)
            for item in exp:
                if item not in all_expanded:
                    all_expanded.append(item)

    if not sub_queries:
        fallback, exp = _clean_single_clause(working)
        sub_queries = [fallback or working]
        all_expanded = exp

    rewritten_query = " ; ".join(sub_queries)
    return RewriteResult(
        original_query=raw_query,
        rewritten_query=rewritten_query,
        sub_queries=tuple(sub_queries),
        expanded_terms=tuple(all_expanded),
        injection_detected=injection_detected,
    )
