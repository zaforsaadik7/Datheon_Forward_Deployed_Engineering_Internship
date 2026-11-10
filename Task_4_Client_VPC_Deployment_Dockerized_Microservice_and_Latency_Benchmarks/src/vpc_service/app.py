"""Unified Client VPC FastAPI Microservice Gateway (/healthz, /readyz, /metrics, /api/v1/rag/*, /api/v1/agent/*)."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import httpx
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, ConfigDict, Field, field_validator

from scripts.seed_vpc import ensure_vpc_seeded
from src.agent.auth import AccessDeniedError as AgentAccessDeniedError
from src.agent.http_client import ResilientERPClient
from src.agent.models import AgentRunResult, PendingApproval
from src.agent.orchestrator import AgentOrchestrator
from src.agent.planner import AgentPlanner, OpenAIFunctionPlanner, RuleBasedReActPlanner
from src.agent.session_store import (
    ApprovalExpiredError,
    ApprovalMismatchError,
    ApprovalNotFoundError,
    ApprovalReplayError,
    SessionNotFoundError,
    SessionStore,
)
from src.agent.tools import ToolExecutor
from src.erp_crm.app import create_erp_crm_app
from src.erp_crm.database import ERPCRMDatabase
from src.etl.vector_store import VectorStore
from src.rag.cli import DeterministicHashEmbedding
from src.rag.engine import MultiTenantRAGEngine
from src.rag.rbac import AuditLogger
from src.rag.reranker import LexicalReranker
from src.vpc_service.auth import (
    VPCAuthenticationError,
    VPCAuthorizationError,
    VPCPrincipal,
    authenticate_vpc_principal,
    require_vpc_supervisor,
)
from src.vpc_service.config import load_vpc_settings
from src.vpc_service.health import check_liveness, evaluate_readiness
from src.vpc_service.middleware import BoundedTokenBucketLimiter, VPCGatewayMiddleware
from src.vpc_service.secrets import SecretBundle, load_secret_bundle
from src.vpc_service.telemetry import TokenUsage, VPCTelemetry, estimate_tokens


class VPCRAGQueryRequest(BaseModel):
    """Strict request schema for POST /api/v1/rag/query."""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class VPCAgentRunRequest(BaseModel):
    """Strict request schema for POST /api/v1/agent/run."""
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(..., min_length=1, max_length=2000)
    max_steps: int = Field(default=8, ge=1, le=15)

    @field_validator("goal")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("goal must not be blank")
        return cleaned


class VPCApprovalDecisionRequest(BaseModel):
    """Strict request schema for POST /api/v1/agent/approvals/{approval_id}/decide."""
    model_config = ConfigDict(extra="forbid")

    decision: Literal["APPROVE", "REJECT"]
    reason: str = Field(default="", max_length=500)


def _compute_agent_token_usage(
    result: AgentRunResult,
    planner: AgentPlanner,
) -> TokenUsage:
    if isinstance(planner, OpenAIFunctionPlanner):
        snap = planner.pop_usage_snapshot()
        if snap is not None:
            return TokenUsage(
                prompt_tokens=snap["prompt_tokens"],
                completion_tokens=snap["completion_tokens"],
                total_tokens=snap["total_tokens"],
                accounting_mode="provider_reported",
            )

    prompt_text = result.goal + " " + " ".join(s.observation or "" for s in result.steps)
    completion_text = (result.final_answer or "") + " " + " ".join(s.thought for s in result.steps)
    p_tok = estimate_tokens(prompt_text)
    c_tok = max(1, estimate_tokens(completion_text))
    return TokenUsage(
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        total_tokens=p_tok + c_tok,
        accounting_mode="estimated",
    )


def create_vpc_app(
    *,
    env: Optional[str] = None,
    var_dir: Optional[Path] = None,
    api_keys_file: Optional[Path] = None,
    metrics_token_file: Optional[Path] = None,
    airgapped_mode: Optional[bool] = None,
    planner_mode: Optional[str] = None,
    auto_seed_on_startup: Optional[bool] = None,
    allow_sample_seed_in_prod: Optional[bool] = None,
    max_body_bytes: Optional[int] = None,
    rate_limit_rps: Optional[float] = None,
    rate_limit_burst: Optional[int] = None,
    openai_http_client: Optional[httpx.Client] = None,
    log_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> FastAPI:
    """Create the hardened Client VPC FastAPI microservice."""
    settings = load_vpc_settings(
        env=env,
        var_dir=var_dir,
        airgapped_mode=airgapped_mode,
        planner_mode=planner_mode,
        auto_seed_on_startup=auto_seed_on_startup,
        allow_sample_seed_in_prod=allow_sample_seed_in_prod,
        max_body_bytes=max_body_bytes,
        rate_limit_rps=rate_limit_rps,
        rate_limit_burst=rate_limit_burst,
        api_keys_file=api_keys_file,
        metrics_token_file=metrics_token_file,
    )
    secret_bundle: SecretBundle = load_secret_bundle(
        env=settings.env,
        api_keys_file=settings.api_keys_file,
        metrics_token_file=settings.metrics_token_file,
    )
    telemetry = VPCTelemetry()

    state: Dict[str, Any] = {
        "vector_store": None,
        "rag_engine": None,
        "session_store": None,
        "tenant_executors": {},
        "tenant_dbs": {},
        "lifespan_active": False,
    }

    def _init_stores() -> None:
        settings.var_dir.mkdir(parents=True, exist_ok=True)
        ensure_vpc_seeded(
            var_dir=settings.var_dir,
            env=settings.env,
            auto_seed=settings.auto_seed_on_startup,
            allow_sample_seed=settings.allow_sample_seed_in_prod,
        )
        vec_store = VectorStore(
            path=settings.var_dir / "chroma_db",
            collection_name="enterprise_docs",
            embedding_function=DeterministicHashEmbedding(),
        )
        state["vector_store"] = vec_store
        state["rag_engine"] = MultiTenantRAGEngine(
            store=vec_store,
            reranker=LexicalReranker(),
            audit_logger=AuditLogger(),
        )
        state["session_store"] = SessionStore(db_path=settings.var_dir / "agent_sessions.db")

    def _get_tenant_executor(tenant_id: str) -> ToolExecutor:
        executors: Dict[str, ToolExecutor] = state["tenant_executors"]
        if tenant_id not in executors:
            db_path = settings.var_dir / f"{tenant_id}_erp.db"
            erp_db = ERPCRMDatabase(db_path)
            if state.get("lifespan_active"):
                erp_db.open_keep_alive()
            state["tenant_dbs"][tenant_id] = erp_db
            erp_app = create_erp_crm_app(db=erp_db)
            client = ResilientERPClient(app=erp_app, max_retries=3, base_backoff_s=0.01)
            executors[tenant_id] = ToolExecutor(client)
        return executors[tenant_id]

    def _make_planner() -> AgentPlanner:
        if settings.planner_mode == "openai" and not settings.airgapped_mode:
            return OpenAIFunctionPlanner(
                api_key=secret_bundle.openai_api_key or "vpc-openai-key",
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                http_client=openai_http_client,
            )
        return RuleBasedReActPlanner()

    # Initialize eagerly so stores exist even if a caller doesn't enter lifespan context
    _init_stores()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["vector_store"] is None:
            _init_stores()
        state["lifespan_active"] = True
        if state["session_store"] is not None:
            state["session_store"].open_keep_alive()
        for erp_db in state["tenant_dbs"].values():
            erp_db.open_keep_alive()
        try:
            yield
        finally:
            state["lifespan_active"] = False
            executors: Dict[str, ToolExecutor] = state.get("tenant_executors", {})
            for executor in executors.values():
                executor.client.close()
            for erp_db in state.get("tenant_dbs", {}).values():
                erp_db.close()
            if state.get("session_store") is not None:
                state["session_store"].close()

    app = FastAPI(
        title="Datheon Client VPC AI Microservice Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )

    def _get_principal(
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
    ) -> VPCPrincipal:
        raw_cred = x_api_key if (x_api_key and x_api_key.strip()) else authorization
        try:
            principal = authenticate_vpc_principal(raw_cred, secret_bundle)
            request.scope["fde_principal_id"] = principal.principal_id
            request.scope["fde_tenant_id"] = principal.tenant_id
            return principal
        except VPCAuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return check_liveness()

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        code, body = evaluate_readiness(
            var_dir=settings.var_dir,
            secret_bundle=secret_bundle,
            vector_store=state["vector_store"],
        )
        telemetry.refresh_dynamic_gauges(
            var_dir=settings.var_dir,
            secret_bundle=secret_bundle,
            vector_store=state["vector_store"],
            session_store=state["session_store"],
            precomputed_readiness=body,
        )
        return JSONResponse(status_code=code, content=body)

    @app.get("/metrics")
    def metrics(
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Response:
        if secret_bundle.metrics_token:
            raw = authorization or x_api_key or ""
            token = raw.strip()
            if token.lower().startswith("bearer "):
                token = token[7:].strip()
            if token != secret_bundle.metrics_token and token not in secret_bundle.api_key_map:
                raise HTTPException(
                    status_code=401,
                    detail="Valid metrics bearer token or API key required to scrape /metrics.",
                )

        payload_bytes = telemetry.render_prometheus_text(
            var_dir=settings.var_dir,
            secret_bundle=secret_bundle,
            vector_store=state["vector_store"],
            session_store=state["session_store"],
        )
        return Response(content=payload_bytes, media_type=CONTENT_TYPE_LATEST)

    @app.post("/api/v1/rag/query")
    def rag_query(
        payload: VPCRAGQueryRequest,
        principal: VPCPrincipal = Depends(_get_principal),
    ) -> Dict[str, Any]:
        rag_engine: MultiTenantRAGEngine = state["rag_engine"]
        rag_res = rag_engine.answer(
            query=payload.query,
            principal=principal.to_rag_principal(),
            top_k=payload.top_k,
        )
        prompt_text = (
            payload.query
            + " "
            + rag_res.rewritten_query
            + " "
            + " ".join(c.format_tag() for c in rag_res.citations)
        )
        p_tok = estimate_tokens(prompt_text)
        c_tok = max(1, estimate_tokens(rag_res.answer))
        usage = TokenUsage(
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
            accounting_mode="estimated",
        )
        telemetry.record_tokens("rag_query", usage)
        outcome_label = "abstained" if rag_res.abstained else "answered"
        telemetry.rag_queries_total.labels(outcome=outcome_label, reranker="lexical").inc()

        out = rag_res.model_dump()
        out["token_usage"] = usage.model_dump()
        return out

    @app.post("/api/v1/agent/run")
    def agent_run(
        payload: VPCAgentRunRequest,
        principal: VPCPrincipal = Depends(_get_principal),
    ) -> Dict[str, Any]:
        planner = _make_planner()
        executor = _get_tenant_executor(principal.tenant_id)
        session_store: SessionStore = state["session_store"]
        orchestrator = AgentOrchestrator(
            executor=executor,
            session_store=session_store,
            planner=planner,
            max_steps=payload.max_steps,
        )
        result = orchestrator.run_goal(
            operator=principal.to_agent_operator(),
            goal=payload.goal,
        )
        for step in result.steps:
            if step.action_type == "TOOL_CALL" and step.tool_name:
                telemetry.agent_tool_calls_total.labels(
                    tool_name=step.tool_name,
                    sensitivity=step.sensitivity or "SAFE_READ",
                    status=step.status,
                ).inc()

        usage = _compute_agent_token_usage(result, planner)
        telemetry.record_tokens("agent_run", usage)

        out = result.model_dump()
        out["token_usage"] = usage.model_dump()
        return out

    @app.get("/api/v1/agent/sessions/{session_id}")
    def get_agent_session(
        session_id: str,
        principal: VPCPrincipal = Depends(_get_principal),
    ) -> Dict[str, Any]:
        session_store: SessionStore = state["session_store"]
        try:
            session = session_store.get_session_result(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if session.tenant_id != principal.tenant_id:
            raise HTTPException(
                status_code=403,
                detail=f"Cross-tenant access denied for session '{session.session_id}'.",
            )
        if principal.agent_role != "supervisor" and session.operator_id != principal.principal_id:
            raise HTTPException(
                status_code=403,
                detail=f"Operator '{principal.principal_id}' is not authorized to view session '{session.session_id}'.",
            )
        return session.model_dump()

    @app.get("/api/v1/agent/approvals", response_model=List[PendingApproval])
    def list_approvals(
        session_id: Optional[str] = None,
        status: Optional[str] = None,
        principal: VPCPrincipal = Depends(_get_principal),
    ) -> List[PendingApproval]:
        session_store: SessionStore = state["session_store"]
        if session_id:
            try:
                session = session_store.get_session_result(session_id)
            except SessionNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if session.tenant_id != principal.tenant_id or (
                principal.agent_role != "supervisor" and session.operator_id != principal.principal_id
            ):
                raise HTTPException(
                    status_code=403,
                    detail=f"Not authorized to view approvals for session '{session.session_id}'.",
                )

        req_by = None if principal.agent_role == "supervisor" else principal.principal_id
        return session_store.list_approvals(
            session_id=session_id,
            requested_by=req_by,
            tenant_id=principal.tenant_id,
            status=status,
        )

    @app.post("/api/v1/agent/approvals/{approval_id}/decide")
    def decide_approval(
        approval_id: str,
        payload: VPCApprovalDecisionRequest,
        principal: VPCPrincipal = Depends(_get_principal),
    ) -> Dict[str, Any]:
        try:
            require_vpc_supervisor(principal)
        except VPCAuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

        planner = _make_planner()
        executor = _get_tenant_executor(principal.tenant_id)
        session_store: SessionStore = state["session_store"]
        orchestrator = AgentOrchestrator(
            executor=executor,
            session_store=session_store,
            planner=planner,
        )
        try:
            result = orchestrator.decide_and_resume(
                approval_id=approval_id,
                supervisor=principal.to_agent_operator(),
                decision=payload.decision,
                notes=payload.reason,
            )
        except ApprovalNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AgentAccessDeniedError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (ApprovalReplayError, ApprovalExpiredError, ApprovalMismatchError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        usage = _compute_agent_token_usage(result, planner)
        telemetry.record_tokens("agent_run", usage)

        out = result.model_dump()
        out["token_usage"] = usage.model_dump()
        return out

    limiter = BoundedTokenBucketLimiter(
        rate_rps=settings.rate_limit_rps,
        burst=settings.rate_limit_burst,
    )
    app.add_middleware(
        VPCGatewayMiddleware,
        telemetry=telemetry,
        secret_bundle=secret_bundle,
        max_body_bytes=settings.max_body_bytes,
        rate_limiter=limiter,
        log_sink=log_sink,
    )

    return app
