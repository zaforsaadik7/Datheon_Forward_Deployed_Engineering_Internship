"""Authenticated FastAPI service exposing the Autonomous Field Agent and ERP/CRM endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from scripts.seed_db import seed_database
from src.agent.auth import (
    AccessDeniedError,
    AuthenticationError,
    authenticate_operator,
    require_supervisor,
)
from src.agent.http_client import ResilientERPClient
from src.agent.models import (
    AgentRunResult,
    OperatorPrincipal,
    PendingApproval,
)
from src.agent.orchestrator import AgentOrchestrator
from src.agent.planner import AgentPlanner, RuleBasedReActPlanner
from src.agent.session_store import (
    ApprovalExpiredError,
    ApprovalMismatchError,
    ApprovalNotFoundError,
    ApprovalReplayError,
    SessionNotFoundError,
    SessionStore,
)
from src.agent.tools import ToolExecutor
from src.config import get_settings
from src.erp_crm.app import create_erp_crm_app
from src.erp_crm.database import ERPCRMDatabase


class AgentRunRequest(BaseModel):
    """Strict request schema for POST /api/v1/agent/run."""
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(..., min_length=1, max_length=2000)
    max_steps: int = Field(default=8, ge=1, le=15)

    @field_validator("goal")
    @classmethod
    def _reject_blank_goal(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("goal must not be blank")
        return cleaned


class ApprovalDecisionRequest(BaseModel):
    """Strict request schema for POST /api/v1/agent/approvals/{approval_id}/decide."""
    model_config = ConfigDict(extra="forbid")

    decision: Literal["APPROVE", "REJECT"]
    reason: str = Field(default="", max_length=500)


def create_app(
    erp_db_path: Optional[Path] = None,
    session_db_path: Optional[Path] = None,
    planner: Optional[AgentPlanner] = None,
    allow_dev_header: bool = False,
    seed_on_startup: bool = True,
) -> FastAPI:
    """Create the combined FastAPI service with authenticated `/api/v1/agent/*` routes."""
    settings = get_settings()
    resolved_erp_db = erp_db_path or settings.erp_db_path
    resolved_session_db = session_db_path or settings.session_db_path

    if seed_on_startup and not resolved_erp_db.exists():
        seed_database(resolved_erp_db)

    erp_db = ERPCRMDatabase(resolved_erp_db)
    erp_app = create_erp_crm_app(db=erp_db)
    erp_client = ResilientERPClient(app=erp_app, max_retries=3, base_backoff_s=0.01)
    executor = ToolExecutor(erp_client)
    session_store = SessionStore(db_path=resolved_session_db)
    active_planner = planner or RuleBasedReActPlanner()

    app = FastAPI(
        title="Datheon Autonomous Field Agent & Enterprise ERP/CRM API",
        version="1.0.0",
    )

    # Expose ERP/CRM routes on the unified ASGI application
    app.router.routes.extend(erp_app.router.routes)

    def _get_principal(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        authorization: Optional[str] = Header(default=None, alias="Authorization"),
        x_dev_operator: Optional[str] = Header(default=None, alias="X-Dev-Operator"),
    ) -> OperatorPrincipal:
        token = x_api_key if (x_api_key and x_api_key.strip()) else authorization
        try:
            return authenticate_operator(
                api_key=token,
                dev_operator=x_dev_operator,
                allow_dev_header=allow_dev_header,
            )
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def _build_orchestrator(max_steps: int = 8) -> AgentOrchestrator:
        return AgentOrchestrator(
            executor=executor,
            session_store=session_store,
            planner=active_planner,
            max_steps=max_steps,
        )

    @app.get("/api/v1/agent/health")
    def agent_health() -> dict:
        return {"status": "ok", "service": "autonomous-field-agent"}

    @app.post("/api/v1/agent/run", response_model=AgentRunResult)
    def run_agent_goal(
        payload: AgentRunRequest,
        principal: OperatorPrincipal = Depends(_get_principal),
    ) -> AgentRunResult:
        orchestrator = _build_orchestrator(max_steps=payload.max_steps)
        return orchestrator.run_goal(
            operator=principal,
            goal=payload.goal,
        )

    @app.get("/api/v1/agent/sessions/{session_id}", response_model=AgentRunResult)
    def get_agent_session(
        session_id: str,
        principal: OperatorPrincipal = Depends(_get_principal),
    ) -> AgentRunResult:
        try:
            session = session_store.get_session_result(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if principal.role != "supervisor" and session.operator_id != principal.operator_id:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Operator '{principal.operator_id}' is not authorized to view "
                    f"session '{session.session_id}' owned by '{session.operator_id}'."
                ),
            )
        return session

    @app.get("/api/v1/agent/approvals", response_model=List[PendingApproval])
    def list_pending_approvals(
        session_id: Optional[str] = None,
        status: Optional[str] = None,
        principal: OperatorPrincipal = Depends(_get_principal),
    ) -> List[PendingApproval]:
        if session_id:
            try:
                session = session_store.get_session_result(session_id)
            except SessionNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if principal.role != "supervisor" and session.operator_id != principal.operator_id:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Operator '{principal.operator_id}' is not authorized to view approvals "
                        f"for session '{session.session_id}' owned by '{session.operator_id}'."
                    ),
                )

        requested_by_filter = None if principal.role == "supervisor" else principal.operator_id
        return session_store.list_approvals(
            session_id=session_id,
            requested_by=requested_by_filter,
            status=status,
        )

    @app.post("/api/v1/agent/approvals/{approval_id}/decide", response_model=AgentRunResult)
    def decide_pending_approval(
        approval_id: str,
        payload: ApprovalDecisionRequest,
        principal: OperatorPrincipal = Depends(_get_principal),
    ) -> AgentRunResult:
        try:
            require_supervisor(principal)
        except AccessDeniedError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

        orchestrator = _build_orchestrator()
        try:
            return orchestrator.decide_and_resume(
                approval_id=approval_id,
                supervisor=principal,
                decision=payload.decision,
                notes=payload.reason,
            )
        except ApprovalNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ApprovalReplayError, ApprovalExpiredError, ApprovalMismatchError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app
