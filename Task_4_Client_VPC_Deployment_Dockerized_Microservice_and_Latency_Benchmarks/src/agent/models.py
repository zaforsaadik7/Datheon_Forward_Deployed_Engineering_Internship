"""Immutable Pydantic v2 data models for ReAct traces, tool definitions, sessions, and HITL approvals."""
from typing import Any, Dict, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

ToolSensitivity = Literal["SAFE_READ", "AUTO_WRITE", "HITL_REQUIRED"]
SessionStatus = Literal["RUNNING", "AWAITING_APPROVAL", "COMPLETED", "FAILED"]
ApprovalStatus = Literal["PENDING", "APPROVED", "CONSUMED", "REJECTED", "EXPIRED"]
OperatorRole = Literal["operator", "supervisor"]


class OperatorPrincipal(BaseModel):
    """Authenticated operator or supervisor identity."""
    model_config = ConfigDict(frozen=True)

    operator_id: str
    role: OperatorRole
    tenant_id: str = "acme_corp"
    display_name: str = ""


class ToolSpec(BaseModel):
    """JSON-Schema specification for an agent tool."""
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    sensitivity: ToolSensitivity
    parameters_schema: Dict[str, Any]


class ReActStep(BaseModel):
    """A single Thought -> Action -> Observation cycle in the agent execution trace."""
    model_config = ConfigDict(frozen=True)

    step_index: int
    thought: str
    action_type: Literal["TOOL_CALL", "FINAL_ANSWER"]
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    sensitivity: Optional[ToolSensitivity] = None
    observation: Optional[str] = None
    status: Literal[
        "OK",
        "VALIDATION_ERROR",
        "API_ERROR",
        "AWAITING_APPROVAL",
        "REJECTED_BY_SUPERVISOR",
        "FINAL",
    ] = "OK"
    http_status: Optional[int] = None
    retries_used: int = 0
    latency_ms: float = 0.0


class PendingApproval(BaseModel):
    """Persisted Human-in-the-Loop approval gate bound to canonical tool arguments hash."""
    model_config = ConfigDict(frozen=True)

    approval_id: str
    session_id: str
    step_index: int
    requested_by: str
    tenant_id: str = "acme_corp"
    tool_name: str
    tool_args: Dict[str, Any]
    args_sha256: str
    reason: str
    status: ApprovalStatus
    created_at: str
    expires_at: str
    decided_by: Optional[str] = None
    decision_notes: Optional[str] = None
    decided_at: Optional[str] = None


class ToolCallAction(BaseModel):
    """Planner output requesting a tool invocation."""
    model_config = ConfigDict(frozen=True)

    action_type: Literal["TOOL_CALL"] = "TOOL_CALL"
    thought: str
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)


class FinalAnswerAction(BaseModel):
    """Planner output concluding the ReAct loop with a final answer."""
    model_config = ConfigDict(frozen=True)

    action_type: Literal["FINAL_ANSWER"] = "FINAL_ANSWER"
    thought: str
    final_answer: str


class AgentRunResult(BaseModel):
    """Complete result of starting or resuming an autonomous field agent session."""
    model_config = ConfigDict(frozen=True)

    session_id: str
    operator_id: str
    tenant_id: str = "acme_corp"
    goal: str
    status: SessionStatus
    final_answer: Optional[str] = None
    steps: Tuple[ReActStep, ...] = ()
    pending_approval: Optional[PendingApproval] = None
    total_retries: int = 0
    validation_repairs: int = 0
    total_ms: float = 0.0

