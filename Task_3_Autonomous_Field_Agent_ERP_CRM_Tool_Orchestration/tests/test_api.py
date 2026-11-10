from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.agent.auth import _DEFAULT_DEV_KEYS
from src.api.app import create_app


FIELD_KEY = _DEFAULT_DEV_KEYS["field_operator"]
SUPERVISOR_KEY = _DEFAULT_DEV_KEYS["ops_supervisor"]


def _make_client(tmp_path: Path) -> TestClient:
    app = create_app(
        erp_db_path=tmp_path / "erp.db",
        session_db_path=tmp_path / "sessions.db",
    )
    return TestClient(app)


def test_agent_endpoints_require_authentication(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # Missing header -> 401
    assert client.post("/api/v1/agent/run", json={"goal": "Check SKU-GPU-H100"}).status_code == 401
    assert client.get("/api/v1/agent/sessions/sess-1").status_code == 401
    assert client.get("/api/v1/agent/approvals").status_code == 401
    assert (
        client.post(
            "/api/v1/agent/approvals/appr-1/decide",
            json={"decision": "APPROVE", "reason": "ok"},
        ).status_code
        == 401
    )

    # Invalid key -> 401
    bad_headers = {"X-API-Key": "invalid-key"}
    assert (
        client.post("/api/v1/agent/run", json={"goal": "Check SKU-GPU-H100"}, headers=bad_headers).status_code
        == 401
    )


def test_agent_run_rejects_malformed_body_or_privilege_injection(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    headers = {"X-API-Key": FIELD_KEY}

    # Blank goal -> 422
    assert client.post("/api/v1/agent/run", json={"goal": "   "}, headers=headers).status_code == 422

    # Injecting role or operator_id into body -> 422 (extra="forbid")
    assert (
        client.post(
            "/api/v1/agent/run",
            json={"goal": "Check SKU-GPU-H100", "role": "supervisor"},
            headers=headers,
        ).status_code
        == 422
    )


def test_full_http_hitl_lifecycle_and_replay_protection(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    field_headers = {"X-API-Key": FIELD_KEY}
    sup_headers = {"X-API-Key": SUPERVISOR_KEY}

    # 1. Field operator starts a goal that requires HITL approval
    run_resp = client.post(
        "/api/v1/agent/run",
        json={"goal": "Reserve 3 units of SKU-EDGE-NPU1 for LEAD-101 and update LEAD-101 status to WON"},
        headers=field_headers,
    )
    assert run_resp.status_code == 200
    session_data = run_resp.json()
    assert session_data["status"] == "AWAITING_APPROVAL"
    session_id = session_data["session_id"]
    first_approval_id = session_data["pending_approval"]["approval_id"]

    # 2. Pending approvals list includes the ticket
    list_resp = client.get("/api/v1/agent/approvals", headers=sup_headers)
    assert list_resp.status_code == 200
    approvals = list_resp.json()
    assert any(a["approval_id"] == first_approval_id for a in approvals)

    # 3. Field operator CANNOT self-approve (403 Forbidden)
    forbid_resp = client.post(
        f"/api/v1/agent/approvals/{first_approval_id}/decide",
        json={"decision": "APPROVE", "reason": "Self-approving my own request"},
        headers=field_headers,
    )
    assert forbid_resp.status_code == 403

    # 4. Supervisor approves the first sensitive action (reserve_inventory)
    #    Session resumes from SQLite and pauses at the second sensitive action (update_lead_status)
    decide_1 = client.post(
        f"/api/v1/agent/approvals/{first_approval_id}/decide",
        json={"decision": "APPROVE", "reason": "Stock verified for Apex"},
        headers=sup_headers,
    )
    assert decide_1.status_code == 200
    after_first = decide_1.json()
    assert after_first["session_id"] == session_id
    assert after_first["status"] == "AWAITING_APPROVAL"
    second_approval_id = after_first["pending_approval"]["approval_id"]
    assert second_approval_id != first_approval_id

    # 5. Replaying the first approval_id returns 409 Conflict
    replay_resp = client.post(
        f"/api/v1/agent/approvals/{first_approval_id}/decide",
        json={"decision": "APPROVE", "reason": "Replay attempt"},
        headers=sup_headers,
    )
    assert replay_resp.status_code == 409

    # 6. Supervisor approves the second sensitive action (update_lead_status) -> COMPLETED
    decide_2 = client.post(
        f"/api/v1/agent/approvals/{second_approval_id}/decide",
        json={"decision": "APPROVE", "reason": "Deal signed"},
        headers=sup_headers,
    )
    assert decide_2.status_code == 200
    final_data = decide_2.json()
    assert final_data["status"] == "COMPLETED"
    assert "WON" in final_data["final_answer"]

    # 7. GET /api/v1/agent/sessions/{session_id} returns the full persisted trace
    get_sess = client.get(f"/api/v1/agent/sessions/{session_id}", headers=field_headers)
    assert get_sess.status_code == 200
    assert get_sess.json()["status"] == "COMPLETED"
    assert len(get_sess.json()["steps"]) >= 4
