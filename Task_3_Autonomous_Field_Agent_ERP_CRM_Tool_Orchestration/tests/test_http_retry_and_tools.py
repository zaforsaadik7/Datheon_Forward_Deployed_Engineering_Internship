from pathlib import Path
from scripts.seed_db import seed_database
from src.erp_crm.app import create_erp_crm_app
from src.agent.http_client import ResilientERPClient
from src.agent.tools import (
    ToolExecutor,
    canonical_args_sha256,
    classify_tool_sensitivity,
)


def _make_erp_client(tmp_path: Path) -> ResilientERPClient:
    db = seed_database(db_path=tmp_path / "erp_retry.db")
    app = create_erp_crm_app(db=db)
    return ResilientERPClient(app=app, max_retries=3, base_backoff_s=0.005)


def test_resilient_http_client_retries_503_with_idempotency_and_never_retries_422_or_409(
    tmp_path: Path,
):
    client = _make_erp_client(tmp_path)

    # 1. Inject 2 transient 503 errors on POST /erp/v1/inventory/SKU-GPU-H100/reserve
    client.request(
        "POST",
        "/admin/faults",
        json_body={
            "method": "POST",
            "path": "/erp/v1/inventory/SKU-GPU-H100/reserve",
            "remaining_failures": 2,
            "status_code": 503,
        },
    )

    resp = client.request(
        "POST",
        "/erp/v1/inventory/SKU-GPU-H100/reserve",
        json_body={"quantity": 4, "lead_id": "LEAD-101", "requested_by": "field_operator"},
        idempotency_key="idem-gpu-4",
    )
    assert resp.status_code == 200
    assert resp.retries_used == 2
    assert resp.data["quantity_reserved"] == 4
    assert resp.data["remaining_available"] == 16

    # 2. Client error 409 Conflict (insufficient stock on SKU-SEC-HSM1) is NOT retried (retries_used == 0)
    resp_conflict = client.request(
        "POST",
        "/erp/v1/inventory/SKU-SEC-HSM1/reserve",
        json_body={"quantity": 2, "lead_id": "LEAD-101", "requested_by": "field_operator"},
        idempotency_key="idem-hsm-conflict",
    )
    assert resp_conflict.status_code == 409
    assert resp_conflict.retries_used == 0


def test_tool_sensitivity_classification_and_preflight_schema_validation(tmp_path: Path):
    client = _make_erp_client(tmp_path)
    executor = ToolExecutor(client=client)

    # Sensitivity rules enforced by code
    assert classify_tool_sensitivity("get_inventory", {"sku": "SKU-GPU-H100"}) == "SAFE_READ"
    assert classify_tool_sensitivity("get_lead", {"lead_id": "LEAD-101"}) == "SAFE_READ"
    assert (
        classify_tool_sensitivity(
            "create_ticket",
            {"title": "Low issue", "description": "Minor check", "priority": "MEDIUM"},
        )
        == "AUTO_WRITE"
    )
    assert (
        classify_tool_sensitivity(
            "create_ticket",
            {"title": "Outage", "description": "Critical issue", "priority": "HIGH"},
        )
        == "HITL_REQUIRED"
    )
    assert (
        classify_tool_sensitivity(
            "reserve_inventory",
            {"sku": "SKU-GPU-H100", "quantity": 5, "lead_id": "LEAD-101"},
        )
        == "HITL_REQUIRED"
    )
    assert (
        classify_tool_sensitivity(
            "update_lead_status",
            {"lead_id": "LEAD-101", "status": "WON"},
        )
        == "HITL_REQUIRED"
    )

    # Pre-flight validation catches invalid arguments before touching HTTP
    val_err = executor.validate_args(
        "update_lead_status",
        {"lead_id": "LEAD-101", "status": "CLOSED_WON"},
    )
    assert val_err is not None
    assert "Allowed values" in val_err or "NEW" in val_err

    # Safe read executes and wraps output in <tool_output_data> delimiter
    exec_out = executor.execute_validated(
        session_id="sess_1",
        step_index=1,
        tool_name="get_inventory",
        tool_args={"sku": "SKU-GPU-H100"},
    )
    assert exec_out.ok is True
    assert "<tool_output_data tool=\"get_inventory\">" in exec_out.observation_text

    # Canonical SHA-256 hash is key-order independent
    h1 = canonical_args_sha256("reserve_inventory", {"sku": "SKU-GPU-H100", "quantity": 5, "lead_id": "LEAD-101"})
    h2 = canonical_args_sha256("reserve_inventory", {"lead_id": "LEAD-101", "quantity": 5, "sku": "SKU-GPU-H100"})
    assert h1 == h2
