"""End-to-end Multi-Tenant RAG engine with double RBAC enforcement and uniform abstention."""
import logging
import time
from typing import Dict, List, Optional
from ..etl.vector_store import QueryHit, VectorStore
from .guardrails import verify_draft_answer
from .models import GuardrailReport, RAGResponse, StageTimings, UserPrincipal
from .query_rewriter import rewrite_query
from .rbac import (
    AuditLogger,
    RBACViolationError,
     assert_hits_authorized,
    authorize_principal,
)
from .reranker import LexicalReranker, Reranker
from .synthesizer import ExtractiveSynthesizer, Generator, render_cited_answer

logger = logging.getLogger(__name__)

UNIFORM_ABSTENTION_MESSAGE = (
    "I cannot answer this question because no relevant documents were found "
    "within your authorized clearance tier and department scope."
)


class MultiTenantRAGEngine:
    """
    Production Multi-Tenant RAG pipeline enforcing:
    1. Principal authorization check (AccessDeniedError on empty department scope)
    2. Query rewriting, acronym expansion, and advisory injection telemetry
    3. Pre-filtered ChromaDB retrieval (tenant_id + Tier-1 Public Rule + tier/dept filter)
    4. Defense-in-depth post-retrieval assertion (assert_hits_authorized)
    5. Candidate-scoped cross-encoder reranking over over-fetched candidates
    6. Marker-based synthesis and numeric/entity hallucination guardrails
    7. Uniform existence-leak-free abstention and structured audit logging
    """

    def __init__(
        self,
        store: VectorStore,
        reranker: Optional[Reranker] = None,
        generator: Optional[Generator] = None,
        audit_logger: Optional[AuditLogger] = None,
        overfetch_k: int = 24,
        top_k: int = 5,
        min_rerank_score: float = 0.25,
        _unsafe_disable_rbac: bool = False,
    ) -> None:
        self.store = store
        self.reranker: Reranker = reranker if reranker is not None else LexicalReranker()
        self.generator: Generator = generator if generator is not None else ExtractiveSynthesizer()
        self.audit_logger: AuditLogger = audit_logger if audit_logger is not None else AuditLogger()
        self.overfetch_k = overfetch_k
        self.top_k = top_k
        self.min_rerank_score = min_rerank_score
        self._unsafe_disable_rbac = _unsafe_disable_rbac

    def _abstain_response(
        self,
        *,
        principal: UserPrincipal,
        raw_query: str,
        rewritten_query: str,
        sub_queries: tuple[str, ...],
        reason: str,
        guardrails: GuardrailReport,
        rewrite_ms: float,
        retrieve_ms: float,
        rerank_ms: float,
        synthesize_ms: float,
        guardrail_ms: float,
        t_start: float,
        injection_detected: bool,
        rbac_violation_blocked: bool = False,
    ) -> RAGResponse:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        timings = StageTimings(
            rewrite_ms=round(rewrite_ms, 3),
            retrieve_ms=round(retrieve_ms, 3),
            rerank_ms=round(rerank_ms, 3),
            synthesize_ms=round(synthesize_ms, 3),
            guardrail_ms=round(guardrail_ms, 3),
            total_ms=round(total_ms, 3),
        )
        self.audit_logger.record(
            principal=principal,
            query=raw_query,
            rewritten_query=rewritten_query,
            authorized_candidates=0,
            cited_sources=(),
            abstained=True,
            injection_detected=injection_detected,
            rbac_violation_blocked=rbac_violation_blocked,
            total_ms=total_ms,
        )
        return RAGResponse(
            answer=UNIFORM_ABSTENTION_MESSAGE,
            abstained=True,
            abstention_reason=reason,  # type: ignore[arg-type]
            rewritten_query=rewritten_query,
            sub_queries=sub_queries,
            citations=(),
            guardrails=guardrails,
            timings=timings,
            authorized_PostFilter_count=0,
        )

    def answer(
        self,
        query: str,
        principal: UserPrincipal,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        t_start = time.perf_counter()
        authorize_principal(principal)
        effective_top_k = top_k if top_k is not None else self.top_k

        # Stage 1: Rewrite query
        t0 = time.perf_counter()
        rw = rewrite_query(query)
        rewrite_ms = (time.perf_counter() - t0) * 1000.0

        empty_guardrails = GuardrailReport(
            passed=True,
            checked_sentences=0,
            verified_sentences=0,
        )

        # Stage 2: Pre-filtered multi-tenant vector retrieval + post-retrieval RBAC barrier
        t1 = time.perf_counter()
        dedup_hits: Dict[str, QueryHit] = {}
        for sub_q in rw.sub_queries:
            if self._unsafe_disable_rbac:
                hits = self.store.query_by_tier_and_department(
                    query_text=sub_q,
                    user_tier=4,
                    user_departments=None,
                    tenant_id=principal.tenant_id,
                    top_k=self.overfetch_k,
                )
            else:
                hits = self.store.query_by_tier_and_department(
                    query_text=sub_q,
                    user_tier=principal.clearance_tier,
                    user_departments=sorted(principal.departments),
                    tenant_id=principal.tenant_id,
                    top_k=self.overfetch_k,
                )
            for h in hits:
                if h.chunk_id not in dedup_hits or h.distance < dedup_hits[h.chunk_id].distance:
                    dedup_hits[h.chunk_id] = h

        raw_candidates: List[QueryHit] = list(dedup_hits.values())
        try:
            if self._unsafe_disable_rbac:
                verified_hits = raw_candidates
            else:
                verified_hits = assert_hits_authorized(raw_candidates, principal)
        except RBACViolationError as exc:
            retrieve_ms = (time.perf_counter() - t1) * 1000.0
            logger.error("Security barrier blocked unauthorized retrieval: %s", exc)
            return self._abstain_response(
                principal=principal,
                raw_query=query,
                rewritten_query=rw.rewritten_query,
                sub_queries=rw.sub_queries,
                reason="no_authorized_results",
                guardrails=empty_guardrails,
                rewrite_ms=rewrite_ms,
                retrieve_ms=retrieve_ms,
                rerank_ms=0.0,
                synthesize_ms=0.0,
                guardrail_ms=0.0,
                t_start=t_start,
                injection_detected=rw.injection_detected,
                rbac_violation_blocked=True,
            )

        retrieve_ms = (time.perf_counter() - t1) * 1000.0

        if not verified_hits:
            return self._abstain_response(
                principal=principal,
                raw_query=query,
                rewritten_query=rw.rewritten_query,
                sub_queries=rw.sub_queries,
                reason="no_authorized_results",
                guardrails=empty_guardrails,
                rewrite_ms=rewrite_ms,
                retrieve_ms=retrieve_ms,
                rerank_ms=0.0,
                synthesize_ms=0.0,
                guardrail_ms=0.0,
                t_start=t_start,
                injection_detected=rw.injection_detected,
            )

        # Stage 3: Candidate-scoped cross-encoder reranking
        t2 = time.perf_counter()
        ranked_chunks = self.reranker.rerank(
            query=rw.rewritten_query,
            candidates=verified_hits,
            top_k=effective_top_k,
            min_score=self.min_rerank_score,
        )
        rerank_ms = (time.perf_counter() - t2) * 1000.0

        if not ranked_chunks:
            return self._abstain_response(
                principal=principal,
                raw_query=query,
                rewritten_query=rw.rewritten_query,
                sub_queries=rw.sub_queries,
                reason="no_authorized_results",
                guardrails=empty_guardrails,
                rewrite_ms=rewrite_ms,
                retrieve_ms=retrieve_ms,
                rerank_ms=rerank_ms,
                synthesize_ms=0.0,
                guardrail_ms=0.0,
                t_start=t_start,
                injection_detected=rw.injection_detected,
            )

        # Stage 4: Answer synthesis
        t3 = time.perf_counter()
        draft = self.generator.generate(rw.rewritten_query, ranked_chunks)
        synthesize_ms = (time.perf_counter() - t3) * 1000.0

        if not draft.statements:
            return self._abstain_response(
                principal=principal,
                raw_query=query,
                rewritten_query=rw.rewritten_query,
                sub_queries=rw.sub_queries,
                reason="no_authorized_results",
                guardrails=empty_guardrails,
                rewrite_ms=rewrite_ms,
                retrieve_ms=retrieve_ms,
                rerank_ms=rerank_ms,
                synthesize_ms=synthesize_ms,
                guardrail_ms=0.0,
                t_start=t_start,
                injection_detected=rw.injection_detected,
            )

        # Stage 5: Hallucination & entity/numeric guardrails
        t4 = time.perf_counter()
        guardrail_report = verify_draft_answer(draft, ranked_chunks)
        guardrail_ms = (time.perf_counter() - t4) * 1000.0

        if not guardrail_report.passed:
            return self._abstain_response(
                principal=principal,
                raw_query=query,
                rewritten_query=rw.rewritten_query,
                sub_queries=rw.sub_queries,
                reason="guardrail_rejected",
                guardrails=guardrail_report,
                rewrite_ms=rewrite_ms,
                retrieve_ms=retrieve_ms,
                rerank_ms=rerank_ms,
                synthesize_ms=synthesize_ms,
                guardrail_ms=guardrail_ms,
                t_start=t_start,
                injection_detected=rw.injection_detected,
            )

        answer_text, citations = render_cited_answer(draft, ranked_chunks)
        total_ms = (time.perf_counter() - t_start) * 1000.0
        timings = StageTimings(
            rewrite_ms=round(rewrite_ms, 3),
            retrieve_ms=round(retrieve_ms, 3),
            rerank_ms=round(rerank_ms, 3),
            synthesize_ms=round(synthesize_ms, 3),
            guardrail_ms=round(guardrail_ms, 3),
            total_ms=round(total_ms, 3),
        )

        self.audit_logger.record(
            principal=principal,
            query=query,
            rewritten_query=rw.rewritten_query,
            authorized_candidates=len(ranked_chunks),
            cited_sources=tuple(c.format_tag() for c in citations),
            abstained=False,
            injection_detected=rw.injection_detected,
            rbac_violation_blocked=False,
            total_ms=total_ms,
        )

        return RAGResponse(
            answer=answer_text,
            abstained=False,
            abstention_reason=None,
            rewritten_query=rw.rewritten_query,
            sub_queries=rw.sub_queries,
            citations=citations,
            guardrails=guardrail_report,
            timings=timings,
            authorized_PostFilter_count=len(ranked_chunks),
        )

