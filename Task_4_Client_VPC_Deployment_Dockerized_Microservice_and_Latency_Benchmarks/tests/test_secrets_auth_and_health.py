from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.vpc_service.auth import (
    VPCAuthenticationError,
    VPCAuthorizationError,
    authenticate_vpc_principal,
    require_vpc_supervisor,
)
from src.vpc_service.config import load_vpc_settings
from src.vpc_service.health import check_liveness, evaluate_readiness
from src.vpc_service.secrets import (
    SecretConfigurationError,
    init_dev_secret_files,
    load_secret_bundle,
)
from scripts.seed_vpc import ensure_vpc_seeded


def test_production_secrets_fail_closed_on_missing_or_dev_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FDE_ENV", "production")
    monkeypatch.delenv("FDE_API_KEYS_JSON", raising=False)
    monkeypatch.delenv("FDE_API_KEYS_JSON_FILE", raising=False)

    # 1. Missing secrets in production -> fails closed
    with pytest.raises(SecretConfigurationError, match="production"):
        load_secret_bundle(env="production")

    # 2. Using known static dev default key in production -> fails closed
    bad_prod_json = json.dumps(
        {
            "key-field-operator": {
                "principal_id": "acme_field_engineer",
                "tenant_id": "acme_corp",
                "clearance_tier": 2,
                "departments": ["engineering"],
                "agent_role": "operator",
            }
        }
    )
    monkeypatch.setenv("FDE_API_KEYS_JSON", bad_prod_json)
    with pytest.raises(SecretConfigurationError, match="development|short|weak"):
        load_secret_bundle(env="production")


def test_file_mounted_secrets_init_dev_secrets_and_redaction(tmp_path: Path) -> None:
    secrets_dir = tmp_path / "secrets"
    keys_file, metrics_file = init_dev_secret_files(secrets_dir)

    assert keys_file.exists()
    assert metrics_file.exists()

    # Generated keys are cryptographically random and pass strict production validation
    bundle = load_secret_bundle(
        env="production",
        api_keys_file=keys_file,
        metrics_token_file=metrics_file,
    )
    assert len(bundle.api_key_map) >= 4
    assert bundle.metrics_token is not None and len(bundle.metrics_token) >= 24

    # Verify redaction in str(), repr(), and to_redacted_dict()
    raw_token = next(iter(bundle.api_key_map.keys()))
    assert raw_token not in repr(bundle)
    assert raw_token not in str(bundle)
    assert bundle.metrics_token not in repr(bundle)
    redacted = bundle.to_redacted_dict()
    assert redacted["metrics_token"] == "***REDACTED***"


def test_unified_vpc_principal_bridges_rag_and_agent_roles(tmp_path: Path) -> None:
    secrets_dir = tmp_path / "secrets"
    keys_file, _ = init_dev_secret_files(secrets_dir)
    bundle = load_secret_bundle(env="production", api_keys_file=keys_file)

    # Find the supervisor key and operator key generated in the file
    sup_key = next(k for k, p in bundle.api_key_map.items() if p.agent_role == "supervisor")
    op_key = next(
        k
        for k, p in bundle.api_key_map.items()
        if p.agent_role == "operator" and p.tenant_id == "acme_corp"
    )

    sup_principal = authenticate_vpc_principal(f"Bearer {sup_key}", bundle)
    assert sup_principal.tenant_id == "acme_corp"
    assert sup_principal.clearance_tier == 4
    assert sup_principal.to_rag_principal().clearance_tier == 4
    assert sup_principal.to_agent_operator().role == "supervisor"
    require_vpc_supervisor(sup_principal)

    op_principal = authenticate_vpc_principal(op_key, bundle)
    assert op_principal.to_agent_operator().role == "operator"
    with pytest.raises(VPCAuthorizationError):
        require_vpc_supervisor(op_principal)

    with pytest.raises(VPCAuthenticationError):
        authenticate_vpc_principal("invalid-token", bundle)


def test_idempotent_auto_seed_and_production_seed_guard_with_sanitized_readiness(tmp_path: Path) -> None:
    var_dir = tmp_path / "var"
    secrets_dir = tmp_path / "secrets"
    keys_file, metrics_file = init_dev_secret_files(secrets_dir)
    bundle = load_secret_bundle(env="production", api_keys_file=keys_file, metrics_token_file=metrics_file)

    # 1. Liveness is always 200 OK
    live = check_liveness()
    assert live["status"] == "alive"

    # 2. In production with allow_sample_seed=False on an empty dir, auto-seed refuses to inject sample data
    prod_seed = ensure_vpc_seeded(
        var_dir=var_dir,
        env="production",
        auto_seed=True,
        allow_sample_seed=False,
    )
    assert prod_seed["seeded_now"] is False

    # Readiness on unseeded dir returns 503 degraded with sanitized output (no file paths leaked)
    code_unseeded, ready_unseeded = evaluate_readiness(var_dir=var_dir, secret_bundle=bundle)
    assert code_unseeded == 503
    assert ready_unseeded["status"] == "degraded"
    assert str(var_dir) not in json.dumps(ready_unseeded)

    # 3. With allow_sample_seed=True (or development mode), first start seeds ChromaDB and SQLite
    first_seed = ensure_vpc_seeded(
        var_dir=var_dir,
        env="production",
        auto_seed=True,
        allow_sample_seed=True,
    )
    assert first_seed["seeded_now"] is True
    assert first_seed["chroma_chunks"] > 0
    assert first_seed["erp_skus"] >= 5

    code_ready, ready_ok = evaluate_readiness(var_dir=var_dir, secret_bundle=bundle)
    assert code_ready == 200
    assert ready_ok["status"] == "ready"
    assert ready_ok["checks"] == {
        "sqlite_db": "ok",
        "chroma_vector_store": "ok",
        "secrets_manager": "ok",
    }

    # 4. Mutate ERP state and verify second start preserves state without overwriting
    from src.erp_crm.database import ERPCRMDatabase
    from src.erp_crm.models import ReserveInventoryRequest

    erp_db = ERPCRMDatabase(var_dir / "acme_corp_erp.db")
    erp_db.reserve_inventory(
        "SKU-GPU-H100",
        ReserveInventoryRequest(quantity=4, lead_id="LEAD-101", requested_by="acme_field_engineer"),
    )
    assert erp_db.get_inventory("SKU-GPU-H100").available == 16

    second_seed = ensure_vpc_seeded(
        var_dir=var_dir,
        env="production",
        auto_seed=True,
        allow_sample_seed=True,
    )
    assert second_seed["seeded_now"] is False
    assert erp_db.get_inventory("SKU-GPU-H100").available == 16

