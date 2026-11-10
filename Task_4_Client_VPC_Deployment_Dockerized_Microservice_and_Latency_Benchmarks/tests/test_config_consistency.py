"""Guards against drift between Dockerfile/Compose and the Python settings loader."""
from __future__ import annotations

import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _env_names_read_by_code() -> set:
    names = set()
    for rel in ("src/vpc_service/config.py", "src/vpc_service/secrets.py"):
        names |= set(re.findall(r'"(FDE_[A-Z_]+)"', (ROOT / rel).read_text(encoding="utf-8")))
    return names


def test_dockerfile_copy_sources_exist() -> None:
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("COPY") and "--from=" not in s:
            srcs = [t for t in s.split()[1:] if not t.startswith("--")][:-1]
            for src in srcs:
                assert (ROOT / src).exists(), f"Dockerfile COPY source missing: {src}"


def test_container_fde_env_vars_are_consumed_by_code() -> None:
    read = _env_names_read_by_code()
    docker_env = set(re.findall(r"\b(FDE_[A-Z_]+)=", (ROOT / "Dockerfile").read_text(encoding="utf-8")))
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    compose_env = set(compose["services"]["fde-vpc-api"]["environment"].keys())
    unused = (docker_env | compose_env) - read
    assert not unused, f"Env vars set in container but never read by code: {unused}"


def test_benchmark_and_smoke_skus_exist_in_seed_data() -> None:
    seed = json.loads((ROOT / "data" / "seed_enterprise_data.json").read_text(encoding="utf-8"))
    known = {item["sku"] for item in seed["inventory"]}
    for rel in ("src/vpc_service/benchmark.py", "scripts/smoke_test.ps1", "scripts/smoke_test.sh"):
        used = set(re.findall(r"SKU-[A-Z0-9\-]+[A-Z0-9]", (ROOT / rel).read_text(encoding="utf-8")))
        assert used <= known, f"{rel} references unseeded SKUs: {used - known}"


def test_requirements_docker_excludes_pytest_and_resilient_client_closes_cleanly(tmp_path: Path) -> None:
    req_docker = (ROOT / "requirements-docker.txt").read_text(encoding="utf-8").lower()
    assert "pytest" not in req_docker, "requirements-docker.txt must not install pytest in the runtime container"

    from src.agent.http_client import ResilientERPClient
    from src.erp_crm.app import create_erp_crm_app
    from src.erp_crm.database import ERPCRMDatabase

    erp_db = ERPCRMDatabase(tmp_path / "erp.db")
    client = ResilientERPClient(app=create_erp_crm_app(db=erp_db))
    res = client.request("GET", "/erp/v1/inventory")
    assert res.status_code == 200
    assert len(client._open_test_clients) == 1
    client.close()
    assert len(client._open_test_clients) == 0
    assert getattr(client._local, "client", None) is None

