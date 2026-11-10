from pathlib import Path
from typing import Sequence
from scripts.seed_db import seed_database
from src.erp_crm.app import create_erp_crm_app
from src.erp_crm.database import ERPCRMDatabase
from src.agent.auth import OPERATORS
from src.agent.http_client import ResilientERPClient
from src.agent.models import FinalAnswerAction, ReActStep, ToolCallAction, ToolSpec
from src.agent.orchestrator import AgentOrchestrator
from src.agent.planner import RuleBasedReActPlanner
from src.agent.session_store import SessionStore
from src.agent.tools import ToolExecutor


def _setup_orchestrator(
    tmp_path: Path,
    planner=None,
) -> tuple[AgentOrchestrator, ERPCRMDatabase, ResilientERPClient, SessionStore]:
    erp_db = seed_database(db_path=tmp_path / "erp.db")
    erp_app = create_erp_crm_app(db=erp_db)
    erp_client = ResilientERPClient(app=erp_app, max_retries=3, base_backoff_s=0.005)
    session_store = SessionStore(db_path=tmp_path / "sessions.db")
    orch = AgentOrchestrator(
        executor=ToolExecutor(erp_client),
        session_store=session_store,
        planner=planner or RuleBasedReActPlanner(),
    )
    return orch, erp_db, erp_client, session_store


def test_read_only_multi_tool_workflow_completes_without_hitl_pause(tmp_path: Path):
    orch, _, _, _ = _setup_orchestrator(tmp_path)
    res = orch.run_goal(
        operator=OPERATORS["field_operator"],
        goal="Check inventory for SKU-GPU-H100 and inspect CRM lead LEAD-101.",
    )
    assert res.status == "COMPLETED"
    assert res.pending_approval is None
    assert res.final_answer is not None
    assert "SKU-GPU-H100" in res.final_answer and "20" in res.final_answer
    assert "LEAD-101" in res.final_answer and "Apex Biotech" in res.final_answer
    tool_names = [s.tool_name for s in res.steps if s.action_type == "TOOL_CALL"]
    assert tool_names == ["get_inventory", "get_lead"]


def test_sensitive_write_workflow_pauses_persists_in_sqlite_and_resumes_on_supervisor_approval(
    tmp_path: Path,
):
    orch, erp_db, _, _ = _setup_orchestrator(tmp_path)

    # Goal requires checking stock, reserving 5 units of SKU-GPU-H100 for LEAD-101, and updating LEAD-101 to WON
    run1 = orch.run_goal(
        operator=OPERATORS["field_operator"],
        goal="Reserve 5 units of SKU-GPU-H100 for LEAD-101 and update LEAD-101 status to WON.",
    )
    # Pauses before reserve_inventory
    assert run1.status == "AWAITING_APPROVAL"
    assert run1.pending_approval is not None
    assert run1.pending_approval.tool_name == "reserve_inventory"
    assert erp_db.get_inventory("SKU-GPU-H100").available == 20

    # Supervisor approves reservation -> resumes, reserves 5 units (available=15), and pauses before update_lead_status
    run2 = orch.decide_and_resume(
        approval_id=run1.pending_approval.approval_id,
        supervisor=OPERATORS["ops_supervisor"],
        decision="APPROVE",
        notes="Stock reservation approved.",
    )
    assert erp_db.get_inventory("SKU-GPU-H100").available == 15
    assert run2.status == "AWAITING_APPROVAL"
    assert run2.pending_approval is not None
    assert run2.pending_approval.tool_name == "update_lead_status"
    assert erp_db.get_lead("LEAD-101").status == "QUALIFIED"

    # Supervisor approves lead update -> resumes, updates LEAD-101 to WON, and completes
    run3 = orch.decide_and_resume(
        approval_id=run2.pending_approval.approval_id,
        supervisor=OPERATORS["ops_supervisor"],
        decision="APPROVE",
        notes="Deal closure verified.",
    )
    assert run3.status == "COMPLETED"
    assert erp_db.get_lead("LEAD-101").status == "WON"
    assert run3.final_answer is not None
    assert "RES-" in run3.final_answer and "TRN-" in run3.final_answer


def test_supervisor_rejection_aborts_write_and_leaves_database_unchanged(tmp_path: Path):
    orch, erp_db, _, _ = _setup_orchestrator(tmp_path)
    run1 = orch.run_goal(
        operator=OPERATORS["field_operator"],
        goal="Reserve 5 units of SKU-GPU-H100 for LEAD-101.",
    )
    assert run1.status == "AWAITING_APPROVAL"
    assert run1.pending_approval is not None

    run2 = orch.decide_and_resume(
        approval_id=run1.pending_approval.approval_id,
        supervisor=OPERATORS["ops_supervisor"],
        decision="REJECT",
        notes="Hold inventory for higher-priority outage.",
    )
    assert run2.status == "COMPLETED"
    assert erp_db.get_inventory("SKU-GPU-H100").available == 20
    assert run2.final_answer is not None
    assert "rejected" in run2.final_answer.lower()


def test_validation_self_healing_loop_repairs_invalid_arguments(tmp_path: Path):
    """
    Planner initially emits invalid status 'CLOSED_WON', receives VALIDATION_ERROR observation,
    repairs to 'WON' on the next step, and succeeds after supervisor approval.
    """

    class MalformedFirstAttemptPlanner:
        def plan_next_step(
            self,
            goal: str,
            tools: Sequence[ToolSpec],
            history: Sequence[ReActStep],
        ):
            if not history:
                return ToolCallAction(
                    thought="Attempting to close LEAD-101 using status CLOSED_WON.",
                    tool_name="update_lead_status",
                    tool_args={"lead_id": "LEAD-101", "status": "CLOSED_WON"},
                )
            last = history[-1]
            if last.status == "VALIDATION_ERROR":
                return ToolCallAction(
                    thought="Repairing invalid status 'CLOSED_WON' to allowed enum value 'WON'.",
                    tool_name="update_lead_status",
                    tool_args={"lead_id": "LEAD-101", "status": "WON", "notes": "Repaired status"},
                )
            return FinalAnswerAction(
                thought="Lead status updated after schema repair.",
                final_answer="Successfully updated LEAD-101 to WON after self-healing validation error.",
            )

    orch, erp_db, _, _ = _setup_orchestrator(tmp_path, planner=MalformedFirstAttemptPlanner())
    run1 = orch.run_goal(
        operator=OPERATORS["field_operator"],
        goal="Update LEAD-101 to WON.",
    )
    # Step 1 failed validation (recorded in trace), Step 2 repaired args and paused at HITL gate!
    assert run1.status == "AWAITING_APPROVAL"
    assert run1.validation_repairs == 1
    assert run1.steps[0].status == "VALIDATION_ERROR"
    assert run1.pending_approval is not None
    assert run1.pending_approval.tool_args["status"] == "WON"

    run2 = orch.decide_and_resume(
        approval_id=run1.pending_approval.approval_id,
        supervisor=OPERATORS["ops_supervisor"],
        decision="APPROVE",
    )
    assert run2.status == "COMPLETED"
    assert run2.validation_repairs == 1
    assert erp_db.get_lead("LEAD-101").status == "WON"


def test_out_of_stock_conflict_creates_auto_write_escalation_ticket(tmp_path: Path):
    orch, erp_db, _, _ = _setup_orchestrator(tmp_path)
    res = orch.run_goal(
        operator=OPERATORS["field_operator"],
        goal="Reserve 2 units of SKU-SEC-HSM1 for LEAD-103 or create a MEDIUM backorder ticket if out of stock.",
    )
    # RuleBasedReActPlanner checks inventory first, sees available=0 for SKU-SEC-HSM1,
    # and creates a MEDIUM priority backorder ticket (AUTO_WRITE -> no HITL pause needed)
    assert res.status == "COMPLETED"
    assert res.final_answer is not None
    assert "TCK-" in res.final_answer
    tickets = erp_db.list_tickets()
    assert any(t.related_sku == "SKU-SEC-HSM1" for t in tickets)

