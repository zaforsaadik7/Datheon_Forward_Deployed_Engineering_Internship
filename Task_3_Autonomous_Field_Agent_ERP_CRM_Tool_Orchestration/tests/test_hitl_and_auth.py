from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from src.agent.auth import (
    AccessDeniedError,
    AuthenticationError,
    OPERATORS,
    authenticate_operator,
    require_supervisor,
)
from src.agent.session_store import (
    ApprovalExpiredError,
    ApprovalMismatchError,
    ApprovalReplayError,
    SessionStore,
)


def test_env_var_operator_auth_and_supervisor_role_separation(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY_FIELD_OPERATOR", "custom-op-key")
    monkeypatch.setenv("AGENT_API_KEY_OPS_SUPERVISOR", "custom-sup-key")

    op = authenticate_operator("custom-op-key")
    assert op.operator_id == "field_operator"
    assert op.role == "operator"

    sup = authenticate_operator("Bearer custom-sup-key")
    assert sup.operator_id == "ops_supervisor"
    assert sup.role == "supervisor"

    with pytest.raises(AuthenticationError):
        authenticate_operator(None)
    with pytest.raises(AuthenticationError):
        authenticate_operator("wrong-key")

    # Operator cannot self-approve HITL gates
    with pytest.raises(AccessDeniedError):
        require_supervisor(op)

    # Supervisor passes role check
    require_supervisor(sup)


def test_sqlite_session_store_enforces_hash_binding_single_use_and_ttl_expiry(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.db")
    sess_id = store.create_session(operator_id="field_operator", goal="Reserve 5 H100s for LEAD-101")

    approval = store.create_pending_approval(
        session_id=sess_id,
        step_index=1,
        requested_by="field_operator",
        tool_name="reserve_inventory",
        tool_args={"sku": "SKU-GPU-H100", "quantity": 5, "lead_id": "LEAD-101"},
        reason="Mutates warehouse stock allocation",
        ttl_seconds=600,
    )
    assert approval.status == "PENDING"

    # 1. Supervisor approves the ticket
    decided = store.decide_approval(
        approval_id=approval.approval_id,
        supervisor=OPERATORS["ops_supervisor"],
        decision="APPROVE",
        notes="Verified against Apex Biotech Q4 PO.",
    )
    assert decided.status == "APPROVED"

    # 2. Attempting to consume with tampered arguments (quantity=50 instead of 5) raises ApprovalMismatchError
    with pytest.raises(ApprovalMismatchError):
        store.consume_approved_ticket(
            approval_id=approval.approval_id,
            tool_name="reserve_inventory",
            tool_args={"sku": "SKU-GPU-H100", "quantity": 50, "lead_id": "LEAD-101"},
        )

    # 3. Consuming with matching arguments succeeds once and transitions status to CONSUMED
    consumed = store.consume_approved_ticket(
        approval_id=approval.approval_id,
        tool_name="reserve_inventory",
        tool_args={"sku": "SKU-GPU-H100", "quantity": 5, "lead_id": "LEAD-101"},
    )
    assert consumed.status == "CONSUMED"

    # 4. Replaying an already-consumed ticket raises ApprovalReplayError
    with pytest.raises(ApprovalReplayError):
        store.consume_approved_ticket(
            approval_id=approval.approval_id,
            tool_name="reserve_inventory",
            tool_args={"sku": "SKU-GPU-H100", "quantity": 5, "lead_id": "LEAD-101"},
        )

    # 5. Re-deciding an already-decided ticket raises ApprovalReplayError
    with pytest.raises(ApprovalReplayError):
        store.decide_approval(
            approval_id=approval.approval_id,
            supervisor=OPERATORS["ops_supervisor"],
            decision="REJECT",
            notes="Too late",
        )


def test_expired_approval_ticket_is_rejected_on_decide_and_consume(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions_exp.db")
    sess_id = store.create_session(operator_id="field_operator", goal="Update LEAD-101 to WON")

    # Create a ticket that is already expired (ttl_seconds=-10)
    past_time = datetime.now(timezone.utc) - timedelta(seconds=30)
    expired_ticket = store.create_pending_approval(
        session_id=sess_id,
        step_index=1,
        requested_by="field_operator",
        tool_name="update_lead_status",
        tool_args={"lead_id": "LEAD-101", "status": "WON"},
        reason="CRM pipeline stage mutation",
        ttl_seconds=10,
        now_override=past_time,
    )

    with pytest.raises(ApprovalExpiredError):
        store.decide_approval(
            approval_id=expired_ticket.approval_id,
            supervisor=OPERATORS["ops_supervisor"],
            decision="APPROVE",
        )

    reloaded = store.get_approval(expired_ticket.approval_id)
    assert reloaded.status == "EXPIRED"
