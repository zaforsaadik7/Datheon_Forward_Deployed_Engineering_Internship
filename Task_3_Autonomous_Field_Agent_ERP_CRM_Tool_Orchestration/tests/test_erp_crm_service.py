from pathlib import Path
from fastapi.testclient import TestClient
from scripts.seed_db import seed_database
from src.erp_crm.app import create_erp_crm_app
from src.erp_crm.database import ERPCRMDatabase


def _make_client(tmp_path: Path) -> tuple[TestClient, ERPCRMDatabase]:
    db_path = tmp_path / "test_erp.db"
    db = seed_database(db_path=db_path)
    app = create_erp_crm_app(db=db)
    return TestClient(app), db


def test_inventory_lookup_exact_boundary_reservation_and_idempotency(tmp_path: Path):
    client, _ = _make_client(tmp_path)

    # 1. Lookup SKU-EDGE-NPU1 (available=3)
    r_get = client.get("/erp/v1/inventory/SKU-EDGE-NPU1")
    assert r_get.status_code == 200
    assert r_get.json()["available"] == 3

    # 2. Exact boundary reservation: reserve all 3 units with Idempotency-Key
    headers = {"Idempotency-Key": "idem-edge-reserve-3"}
    payload = {"quantity": 3, "lead_id": "LEAD-101", "requested_by": "field_operator"}
    r_res1 = client.post("/erp/v1/inventory/SKU-EDGE-NPU1/reserve", json=payload, headers=headers)
    assert r_res1.status_code == 200
    body1 = r_res1.json()
    assert body1["quantity_reserved"] == 3
    assert body1["remaining_available"] == 0

    # 3. Replay with the SAME Idempotency-Key -> returns identical reservation_id without double-decrementing!
    r_res2 = client.post("/erp/v1/inventory/SKU-EDGE-NPU1/reserve", json=payload, headers=headers)
    assert r_res2.status_code == 200
    assert r_res2.json()["reservation_id"] == body1["reservation_id"]
    assert r_res2.json()["remaining_available"] == 0

    # 4. Subsequent reservation with a new Idempotency-Key fails with 409 Conflict (0 available)
    r_conflict = client.post(
        "/erp/v1/inventory/SKU-EDGE-NPU1/reserve",
        json={"quantity": 1, "lead_id": "LEAD-101", "requested_by": "field_operator"},
        headers={"Idempotency-Key": "idem-edge-extra-1"},
    )
    assert r_conflict.status_code == 409
    assert "Insufficient available stock" in r_conflict.json()["detail"]


def test_crm_lead_state_machine_enforces_valid_and_illegal_transitions(tmp_path: Path):
    client, _ = _make_client(tmp_path)

    # 1. Invalid enum value ('CLOSED_WON') -> 422 Unprocessable Entity
    r_bad_enum = client.patch(
        "/crm/v1/leads/LEAD-101/status",
        json={"status": "CLOSED_WON", "notes": "Attempting invalid status string"},
    )
    assert r_bad_enum.status_code == 422

    # 2. Illegal state jump: LEAD-102 is CONTACTED -> cannot jump directly to WON (409 Conflict)
    r_illegal_jump = client.patch(
        "/crm/v1/leads/LEAD-102/status",
        json={"status": "WON", "notes": "Skipping qualification"},
    )
    assert r_illegal_jump.status_code == 409
    assert "Illegal lead status transition" in r_illegal_jump.json()["detail"]

    # 3. Terminal state transition: LEAD-105 is LOST -> cannot transition to QUALIFIED (409 Conflict)
    r_terminal = client.patch(
        "/crm/v1/leads/LEAD-105/status",
        json={"status": "QUALIFIED", "notes": "Reopening lost deal"},
    )
    assert r_terminal.status_code == 409

    # 4. Valid transition: LEAD-101 is QUALIFIED -> WON succeeds (200 OK) and is idempotent
    r_ok = client.patch(
        "/crm/v1/leads/LEAD-101/status",
        json={"status": "WON", "deal_value_usd": 680000.0, "notes": "Contract executed"},
        headers={"Idempotency-Key": "idem-lead-101-won"},
    )
    assert r_ok.status_code == 200
    assert r_ok.json()["status"] == "WON"
    assert r_ok.json()["previous_status"] == "QUALIFIED"
    assert r_ok.json()["deal_value_usd"] == 680000.0

    # Replay with same Idempotency-Key returns 200 OK (not 409)
    r_replay = client.patch(
        "/crm/v1/leads/LEAD-101/status",
        json={"status": "WON", "deal_value_usd": 680000.0, "notes": "Contract executed"},
        headers={"Idempotency-Key": "idem-lead-101-won"},
    )
    assert r_replay.status_code == 200
    assert r_replay.json()["transition_id"] == r_ok.json()["transition_id"]


def test_itsm_ticket_creation_and_transient_fault_injection(tmp_path: Path):
    client, _ = _make_client(tmp_path)

    # Inject 2 transient 503 failures on POST /itsm/v1/tickets
    r_fault = client.post(
        "/admin/faults",
        json={"method": "POST", "path": "/itsm/v1/tickets", "remaining_failures": 2, "status_code": 503},
    )
    assert r_fault.status_code == 200

    payload = {
        "title": "Backorder escalation for SKU-SEC-HSM1",
        "description": "Zero stock in eu-central-de warehouse.",
        "priority": "MEDIUM",
        "related_sku": "SKU-SEC-HSM1",
        "related_lead_id": "LEAD-103",
        "created_by": "field_operator",
    }
    headers = {"Idempotency-Key": "idem-ticket-hsm1"}

    # Attempts 1 and 2 fail with 503
    assert client.post("/itsm/v1/tickets", json=payload, headers=headers).status_code == 503
    assert client.post("/itsm/v1/tickets", json=payload, headers=headers).status_code == 503

    # Attempt 3 succeeds with 200
    r_ok = client.post("/itsm/v1/tickets", json=payload, headers=headers)
    assert r_ok.status_code == 200
    assert r_ok.json()["ticket_id"].startswith("TCK-")
