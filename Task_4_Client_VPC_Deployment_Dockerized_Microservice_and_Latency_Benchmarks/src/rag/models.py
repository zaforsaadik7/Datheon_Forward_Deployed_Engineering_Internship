"""Immutable Pydantic v2 data models for RBAC principals, citations, guardrails, and RAG responses."""
from typing import Any, Dict, FrozenSet, Iterable, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, field_validator
from ..config import DEPARTMENTS, TIERS, DEFAULT_TENANT


class UserPrincipal(BaseModel):
    """Authenticated employee identity and RBAC clearance scope."""
    model_config = ConfigDict(frozen=True)

    user_id: str
    clearance_tier: int
    departments: FrozenSet[str]
    tenant_id: str = DEFAULT_TENANT
    role_title: str = ""

    @field_validator("user_id", "tenant_id")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Identifier must be a non-empty string.")
        return cleaned

    @field_validator("clearance_tier")
    @classmethod
    def validate_tier(cls, v: int) -> int:
        if isinstance(v, bool) or v not in TIERS:
            raise ValueError(f"Invalid clearance_tier '{v}'. Must be in {TIERS}.")
        return v

    @field_validator("departments", mode="before")
    @classmethod
    def validate_departments(cls, v: Iterable[str]) -> FrozenSet[str]:
        if v is None:
            raise ValueError("departments cannot be None.")
        normalized = frozenset(str(d).strip().lower() for d in v if str(d).strip())
        invalid = normalized - DEPARTMENTS
        if invalid:
            raise ValueError(f"Unknown departments {sorted(invalid)}. Must be in {sorted(DEPARTMENTS)}.")
        return normalized


class Citation(BaseModel):
    """Structured provenance citation with 4-level canonical locator formatting."""
    model_config = ConfigDict(frozen=True)

    citation_index: int
    chunk_id: str
    source_file: str
    department: str
    clearance_tier: int
    document_date: str
    tenant_id: str = DEFAULT_TENANT
    page_number: Optional[int] = None
    row_id: Optional[int] = None
    section_title: Optional[str] = None
    chunk_index: int = 0
    locator_type: Literal["page", "row", "section", "chunk"]
    locator_value: str
    rerank_score: float

    def format_tag(self) -> str:
        if self.locator_type == "page":
            return f"[{self.source_file}, p. {self.locator_value}]"
        if self.locator_type == "row":
            return f"[{self.source_file}, row {self.locator_value}]"
        if self.locator_type == "section":
            return f"[{self.source_file}, §{self.locator_value}]"
        return f"[{self.source_file}, chunk {self.locator_value}]"


class RankedChunk(BaseModel):
    """An authorized retrieved chunk scored and ordered by the cross-encoder reranker."""
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    vector_distance: float
    vector_similarity: float
    rerank_score: float
    metadata: Dict[str, Any]


class DraftStatement(BaseModel):
    """A single synthesized statement paired with 1-based candidate chunk indices."""
    model_config = ConfigDict(frozen=True)

    text: str
    cited_indices: Tuple[int, ...]


class DraftAnswer(BaseModel):
    """Structured generator output prior to citation tag rendering."""
    model_config = ConfigDict(frozen=True)

    statements: Tuple[DraftStatement, ...]


class GuardrailReport(BaseModel):
    """Verification report from the numeric, entity, and marker index guardrail pass."""
    model_config = ConfigDict(frozen=True)

    passed: bool
    checked_sentences: int
    verified_sentences: int
    unverified_claims: Tuple[str, ...] = ()
    hallucinated_numbers: Tuple[str, ...] = ()
    hallucinated_entities: Tuple[str, ...] = ()
    invalid_marker_indices: Tuple[int, ...] = ()


class StageTimings(BaseModel):
    """Per-stage latency breakdown in milliseconds."""
    model_config = ConfigDict(frozen=True)

    rewrite_ms: float
    retrieve_ms: float
    rerank_ms: float
    synthesize_ms: float
    guardrail_ms: float
    total_ms: float


class RAGResponse(BaseModel):
    """Complete, audit-ready response returned by MultiTenantRAGEngine."""
    model_config = ConfigDict(frozen=True)

    answer: str
    abstained: bool
    abstention_reason: Optional[
        Literal["no_authorized_results", "low_relevance", "guardrail_rejected"]
    ] = None
    rewritten_query: str
    sub_queries: Tuple[str, ...] = ()
    citations: Tuple[Citation, ...] = ()
    guardrails: GuardrailReport
    timings: StageTimings
    authorized_PostFilter_count: int = 0


class AuditEvent(BaseModel):
    """Structured security audit record for every RAG query execution."""
    model_config = ConfigDict(frozen=True)

    timestamp_iso: str
    user_id: str
    tenant_id: str
    clearance_tier: int
    departments: Tuple[str, ...]
    query: str
    rewritten_query: str
    authorized_candidates: int
    cited_sources: Tuple[str, ...]
    abstained: bool
    injection_detected: bool
    rbac_violation_blocked: bool
    total_ms: float

