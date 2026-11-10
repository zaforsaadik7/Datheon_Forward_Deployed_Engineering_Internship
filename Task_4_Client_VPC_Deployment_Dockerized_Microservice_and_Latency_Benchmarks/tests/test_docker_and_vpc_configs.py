"""Structural and cross-reference verification for Dockerfile, Compose, Prometheus, and smoke scripts."""
from __future__ import annotations

from pathlib import Path
import re

import yaml

from src.vpc_service.telemetry import VPCTelemetry


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_dockerfile_multistage_nonroot_offline_and_healthcheck() -> None:
    dockerfile_path = ROOT_DIR / "Dockerfile"
    assert dockerfile_path.is_file(), "Dockerfile must exist at repository root"
    content = dockerfile_path.read_text(encoding="utf-8")

    # 1. Multi-stage python:3.12-slim with digest-pinning guidance
    from_lines = [line.strip() for line in content.splitlines() if line.strip().upper().startswith("FROM ")]
    assert len(from_lines) >= 2, "Dockerfile must use a multi-stage build (at least 2 FROM stages)"
    assert any("python:3.12-slim" in line for line in from_lines)
    assert "sha256:" in content, "Dockerfile must include digest-pinning annotation/instructions for air-gapped VPC builds"

    # 2. Lean default dependencies (requirements-docker.txt) + optional PRELOAD_HF_MODELS build arg
    assert "requirements-docker.txt" in content
    assert "ARG PRELOAD_HF_MODELS=false" in content
    assert "requirements-models.txt" in content

    # 3. Non-root UID/GID 10001 and offline environment variables
    assert "10001" in content
    assert "USER 10001:10001" in content or "USER fdeuser" in content
    assert "FDE_AIRGAPPED_MODE=true" in content
    assert "HF_HUB_OFFLINE=1" in content
    assert "TRANSFORMERS_OFFLINE=1" in content

    # 4. Pure-Python urllib HEALTHCHECK (no curl dependency) and single-worker Uvicorn with graceful shutdown
    assert "HEALTHCHECK" in content
    assert "urllib.request.urlopen('http://127.0.0.1:8000/healthz'" in content
    assert "--workers" in content and "1" in content
    assert "--timeout-graceful-shutdown" in content and "15" in content


def test_docker_compose_hardening_secrets_and_prometheus_isolation() -> None:
    compose_path = ROOT_DIR / "docker-compose.yml"
    assert compose_path.is_file(), "docker-compose.yml must exist at repository root"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    services = compose.get("services", {})
    assert "fde-vpc-api" in services
    assert "prometheus" in services

    api = services["fde-vpc-api"]
    assert "127.0.0.1:8000:8000" in api.get("ports", [])
    assert api.get("user") == "10001:10001"
    assert api.get("read_only") is True
    assert "/tmp" in api.get("tmpfs", [])
    assert "ALL" in api.get("cap_drop", [])
    assert "no-new-privileges:true" in api.get("security_opt", [])

    env = api.get("environment", {})
    assert env.get("FDE_ENV") == "production"
    assert str(env.get("FDE_AIRGAPPED_MODE")).lower() == "true"
    assert str(env.get("FDE_ALLOW_SAMPLE_SEED")).lower() == "true"
    assert env.get("FDE_API_KEYS_FILE") == "/run/secrets/fde_api_keys_json"
    assert env.get("FDE_METRICS_TOKEN_FILE") == "/run/secrets/fde_metrics_token"

    api_secrets = api.get("secrets", [])
    assert "fde_api_keys_json" in api_secrets
    assert "fde_metrics_token" in api_secrets

    hc = api.get("healthcheck", {})
    hc_cmd = " ".join(hc.get("test", []))
    assert "urllib.request.urlopen('http://127.0.0.1:8000/healthz'" in hc_cmd
    assert hc.get("start_period") == "15s"

    # Prometheus hardening & internal-only exposure
    prom = services["prometheus"]
    assert "ports" not in prom, "Prometheus must NOT publish host ports; use expose on vpc_internal_net"
    assert "9090" in [str(p) for p in prom.get("expose", [])]
    assert prom.get("user") == "65534:65534"
    assert prom.get("read_only") is True
    assert "/tmp" in prom.get("tmpfs", [])
    assert "ALL" in prom.get("cap_drop", [])
    assert "no-new-privileges:true" in prom.get("security_opt", [])
    assert "fde_metrics_token" in prom.get("secrets", [])
    assert any("prometheus_data:/prometheus" in v for v in prom.get("volumes", []))

    top_secrets = compose.get("secrets", {})
    assert "fde_api_keys_json" in top_secrets
    assert "fde_metrics_token" in top_secrets


def test_prometheus_config_and_alert_rules_match_registered_metrics(tmp_path: Path) -> None:
    prom_path = ROOT_DIR / "deploy" / "prometheus.yml"
    rules_path = ROOT_DIR / "deploy" / "prometheus_rules.yml"
    assert prom_path.is_file()
    assert rules_path.is_file()

    prom_cfg = yaml.safe_load(prom_path.read_text(encoding="utf-8"))
    assert "/etc/prometheus/prometheus_rules.yml" in prom_cfg.get("rule_files", [])

    scrape_cfgs = prom_cfg.get("scrape_configs", [])
    assert len(scrape_cfgs) >= 1
    fde_job = scrape_cfgs[0]
    assert fde_job.get("metrics_path") == "/metrics"
    auth_cfg = fde_job.get("authorization", {})
    assert auth_cfg.get("type") == "Bearer"
    assert auth_cfg.get("credentials_file") == "/run/secrets/fde_metrics_token"

    rules_cfg = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    groups = rules_cfg.get("groups", [])
    assert len(groups) >= 1
    rules = groups[0].get("rules", [])
    alert_names = {r["alert"] for r in rules}
    assert alert_names == {
        "FDEHighErrorRate",
        "FDEHighP95Latency",
        "FDEDependencyDown",
        "FDEHitlBacklog",
    }

    # Verify every metric identifier in alert expressions is exported by VPCTelemetry
    telemetry = VPCTelemetry()
    rendered_names = set(VPCTelemetry.METRIC_NAMES) | {
        f"{name}_bucket" for name in VPCTelemetry.METRIC_NAMES
    } | {
        f"{name}_count" for name in VPCTelemetry.METRIC_NAMES
    } | {
        f"{name}_sum" for name in VPCTelemetry.METRIC_NAMES
    }
    assert len(telemetry.METRIC_NAMES) == 8

    for rule in rules:
        expr = rule["expr"]
        referenced_metrics = set(re.findall(r"\bfde_[a-z0-9_]+\b", expr))
        assert referenced_metrics, f"Alert {rule['alert']} must reference at least one fde_* metric"
        unknown = referenced_metrics - rendered_names
        assert not unknown, f"Alert {rule['alert']} references unregistered metrics: {unknown}"


def test_gitignore_dockerignore_and_smoke_scripts_exist() -> None:
    gitignore = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (ROOT_DIR / ".dockerignore").read_text(encoding="utf-8")
    for secret_rel in ("deploy/secrets/fde_api_keys.json", "deploy/secrets/fde_metrics_token.txt"):
        assert secret_rel in gitignore
        assert secret_rel in dockerignore

    ps1_path = ROOT_DIR / "scripts" / "smoke_test.ps1"
    sh_path = ROOT_DIR / "scripts" / "smoke_test.sh"
    assert ps1_path.is_file()
    assert sh_path.is_file()

    for script_text in (ps1_path.read_text(encoding="utf-8"), sh_path.read_text(encoding="utf-8")):
        assert "init-dev-secrets" in script_text
        assert "docker compose up" in script_text
        assert "promtool check config" in script_text
        assert "promtool check rules" in script_text
        assert "/healthz" in script_text
        assert "/readyz" in script_text
        assert "/api/v1/rag/query" in script_text
        assert "/api/v1/agent/run" in script_text
        assert "/metrics" in script_text
        assert "docker compose down" in script_text
