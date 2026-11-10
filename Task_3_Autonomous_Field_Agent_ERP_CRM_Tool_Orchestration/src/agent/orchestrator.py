"""ReAct Thought-Action-Observation orchestrator with validation self-healing and SQLite HITL pause/resume."""
import time
from typing import Literal, Optional

from .models import (
    AgentRunResult,
    FinalAnswerAction,
    OperatorPrincipal,
    ReActStep,
    ToolCallAction,
)
from .planner import AgentPlanner, RuleBasedReActPlanner
from .session_store import SessionStore
from .tools import TOOL_SPECS, ToolExecutor, classify_tool_sensitivity


class AgentOrchestrator:
    """
    Coordinates the ReAct loop across:
    - Pluggable AgentPlanner (RuleBasedReActPlanner or OpenAIFunctionPlanner)
    - Pre-flight Pydantic schema validation and self-healing repair loop
    - Code-enforced sensitivity policy (SAFE_READ, AUTO_WRITE, HITL_REQUIRED)
    - SQLite-persisted sessions and single-use, hash-bound, expiring PendingApproval tickets
    """

    def __init__(
        self,
        executor: ToolExecutor,
        session_store: SessionStore,
        planner: Optional[AgentPlanner] = None,
        max_steps: int = 8,
        approval_ttl_seconds: int = 900,
    ) -> None:
        self.executor = executor
        self.session_store = session_store
        self.planner: AgentPlanner = planner if planner is not None else RuleBasedReActPlanner()
        self.max_steps = max_steps
        self.approval_ttl_seconds = approval_ttl_seconds

    def run_goal(
        self,
        operator: OperatorPrincipal,
        goal: str,
    ) -> AgentRunResult:
        """Create a new SQLite session and run the ReAct loop until completion or HITL pause."""
        if not goal or not goal.strip():
            raise ValueError("Agent goal must not be empty.")
        session_id = self.session_store.create_session(
            operator_id=operator.operator_id,
            goal=goal.strip(),
        )
        return self._continue_loop(session_id=session_id, operator_id=operator.operator_id)

    def decide_and_resume(
        self,
        *,
        approval_id: str,
        supervisor: OperatorPrincipal,
        decision: Literal["APPROVE", "REJECT"],
        notes: str = "",
    ) -> AgentRunResult:
        """
        Record a supervisor's APPROVE/REJECT decision on a PendingApproval ticket,
        consume the ticket if approved (verifying args_sha256 and TTL), update the paused
        step's observation, and resume the ReAct loop from SQLite.
        """
        t0 = time.perf_counter()
        decided = self.session_store.decide_approval(
            approval_id=approval_id,
            supervisor=supervisor,
            decision=decision,
            notes=notes,
        )
        session_res = self.session_store.get_session_result(decided.session_id)

        if decision == "REJECT":
            rejected_step = ReActStep(
                step_index=decided.step_index,
                thought=f"Awaiting supervisor decision on {decided.tool_name}.",
                action_type="TOOL_CALL",
                tool_name=decided.tool_name,
                tool_args=decided.tool_args,
                sensitivity="HITL_REQUIRED",
                observation=f"Supervisor '{supervisor.operator_id}' REJECTED action: {notes or 'Policy restriction'}.",
                status="REJECTED_BY_SUPERVISOR",
                http_status=403,
                retries_used=0,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )
            self.session_store.update_last_step(decided.session_id, rejected_step)
            return self._continue_loop(
                session_id=decided.session_id,
                operator_id=session_res.operator_id,
            )

        # Consume the single-use ticket (verifies status=='APPROVED', TTL, and args_sha256)
        self.session_store.consume_approved_ticket(
            approval_id=decided.approval_id,
            tool_name=decided.tool_name,
            tool_args=decided.tool_args,
        )

        # Execute the approved tool against the ERP/CRM REST service
        exec_out = self.executor.execute_validated(
            session_id=decided.session_id,
            step_index=decided.step_index,
            tool_name=decided.tool_name,
            tool_args=decided.tool_args,
            operator_id=session_res.operator_id,
        )
        completed_step = ReActStep(
            step_index=decided.step_index,
            thought=f"Supervisor '{supervisor.operator_id}' approved {decided.tool_name}; executing transaction.",
            action_type="TOOL_CALL",
            tool_name=decided.tool_name,
            tool_args=decided.tool_args,
            sensitivity="HITL_REQUIRED",
            observation=exec_out.observation_text,
            status=exec_out.status_label,  # type: ignore[arg-type]
            http_status=exec_out.http_status,
            retries_used=exec_out.retries_used,
            latency_ms=exec_out.latency_ms,
        )
        self.session_store.update_last_step(decided.session_id, completed_step)

        return self._continue_loop(
            session_id=decided.session_id,
            operator_id=session_res.operator_id,
        )

    def _continue_loop(
        self,
        *,
        session_id: str,
        operator_id: str,
    ) -> AgentRunResult:
        t_start = time.perf_counter()
        current_session = self.session_store.get_session_result(session_id)
        goal = current_session.goal

        while True:
            steps = self.session_store.get_steps(session_id)
            total_retries = sum(s.retries_used for s in steps)
            validation_repairs = sum(1 for s in steps if s.status == "VALIDATION_ERROR")

            if len(steps) >= self.max_steps:
                elapsed = current_session.total_ms + (time.perf_counter() - t_start) * 1000.0
                self.session_store.update_session_state(
                    session_id,
                    status="FAILED",
                    final_answer="Max reasoning steps exceeded without reaching a final answer.",
                    total_retries=total_retries,
                    validation_repairs=validation_repairs,
                    total_ms=elapsed,
                )
                return self.session_store.get_session_result(session_id)

            next_idx = len(steps) + 1
            action = self.planner.plan_next_step(
                goal=goal,
                tools=TOOL_SPECS,
                history=steps,
            )

            if isinstance(action, FinalAnswerAction):
                final_step = ReActStep(
                    step_index=next_idx,
                    thought=action.thought,
                    action_type="FINAL_ANSWER",
                    observation=action.final_answer,
                    status="FINAL",
                )
                self.session_store.append_step(session_id, final_step)
                elapsed = current_session.total_ms + (time.perf_counter() - t_start) * 1000.0
                self.session_store.update_session_state(
                    session_id,
                    status="COMPLETED",
                    final_answer=action.final_answer,
                    total_retries=total_retries,
                    validation_repairs=validation_repairs,
                    total_ms=elapsed,
                )
                return self.session_store.get_session_result(session_id)

            assert isinstance(action, ToolCallAction)
            tool_name = action.tool_name
            raw_args = action.tool_args

            # 1. Pre-flight Pydantic schema validation
            val_error = self.executor.validate_args(tool_name, raw_args)
            if val_error is not None:
                err_step = ReActStep(
                    step_index=next_idx,
                    thought=action.thought,
                    action_type="TOOL_CALL",
                    tool_name=tool_name,
                    tool_args=raw_args,
                    observation=val_error,
                    status="VALIDATION_ERROR",
                    http_status=422,
                )
                self.session_store.append_step(session_id, err_step)
                continue

            norm_args = self.executor.normalize_args(tool_name, raw_args)
            sensitivity = classify_tool_sensitivity(tool_name, norm_args)

            # 2. Code-enforced HITL barrier for sensitive writes
            if sensitivity == "HITL_REQUIRED":
                pending = self.session_store.create_pending_approval(
                    session_id=session_id,
                    step_index=next_idx,
                    requested_by=operator_id,
                    tool_name=tool_name,
                    tool_args=norm_args,
                    reason=f"Tool '{tool_name}' is classified as HITL_REQUIRED and requires supervisor approval.",
                    ttl_seconds=self.approval_ttl_seconds,
                )
                pause_step = ReActStep(
                    step_index=next_idx,
                    thought=action.thought,
                    action_type="TOOL_CALL",
                    tool_name=tool_name,
                    tool_args=norm_args,
                    sensitivity=sensitivity,
                    observation=(
                        f"Execution paused: awaiting supervisor approval "
                        f"(approval_id={pending.approval_id}, args_sha256={pending.args_sha256[:12]}...)."
                    ),
                    status="AWAITING_APPROVAL",
                )
                self.session_store.append_step(session_id, pause_step)
                elapsed = current_session.total_ms + (time.perf_counter() - t_start) * 1000.0
                self.session_store.update_session_state(
                    session_id,
                    status="AWAITING_APPROVAL",
                    final_answer=None,
                    total_retries=total_retries,
                    validation_repairs=validation_repairs,
                    total_ms=elapsed,
                )
                return self.session_store.get_session_result(session_id)

            # 3. Execute SAFE_READ or AUTO_WRITE tool immediately
            exec_out = self.executor.execute_validated(
                session_id=session_id,
                step_index=next_idx,
                tool_name=tool_name,
                tool_args=norm_args,
                operator_id=operator_id,
            )
            step_record = ReActStep(
                step_index=next_idx,
                thought=action.thought,
                action_type="TOOL_CALL",
                tool_name=tool_name,
                tool_args=norm_args,
                sensitivity=sensitivity,
                observation=exec_out.observation_text,
                status=exec_out.status_label,  # type: ignore[arg-type]
                http_status=exec_out.http_status,
                retries_used=exec_out.retries_used,
                latency_ms=exec_out.latency_ms,
            )
            self.session_store.append_step(session_id, step_record)

