"""Live-Uvicorn subprocess HTTP concurrency & latency benchmark suite across all 5 VPC workloads."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, Tuple

import httpx

from scripts.seed_vpc import ensure_vpc_seeded
from src.vpc_service.secrets import init_dev_secret_files, load_secret_bundle


WORKLOAD_NAMES: Tuple[str, ...] = (
    "health_and_readiness",
    "rag_authorized_query",
    "rag_rbac_denial",
    "agent_safe_read_workflow",
    "agent_hitl_approval_roundtrip",
)

SLO_P95_MS: Dict[str, float] = {
    "health_and_readiness": 50.0,
    "rag_authorized_query": 350.0,
    "rag_rbac_denial": 350.0,
    "agent_safe_read_workflow": 500.0,
    "agent_hitl_approval_roundtrip": 1500.0,
}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(sorted_vals[lower])
    weight = pos - lower
    return float(sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight)


def _sum_prometheus_metric(metrics_text: str, metric_name: str, label_filter: Optional[str] = None) -> float:
    total = 0.0
    pattern = re.compile(rf"^{re.escape(metric_name)}(\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$")
    for line in metrics_text.splitlines():
        line_s = line.strip()
        if not line_s or line_s.startswith("#"):
            continue
        m = pattern.match(line_s)
        if m:
            labels_part = m.group(1) or ""
            if label_filter and label_filter not in labels_part:
                continue
            total += float(m.group(2))
    return total


async def _run_concurrency_tier(
    *,
    concurrency: int,
    total_requests: int,
    worker_fn: Callable[[int], Coroutine[Any, Any, bool]],
) -> Dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    latencies_ms: List[float] = []
    errors = 0

    async def _bounded_call(idx: int) -> None:
        nonlocal errors
        async with sem:
            t0 = time.perf_counter()
            ok = False
            try:
                ok = await worker_fn(idx)
            except Exception:
                ok = False
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            if not ok:
                errors += 1

    wall_t0 = time.perf_counter()
    await asyncio.gather(*(_bounded_call(i) for i in range(total_requests)))
    wall_elapsed_s = max(1e-6, time.perf_counter() - wall_t0)

    sorted_lats = sorted(latencies_ms)
    mean_ms = sum(sorted_lats) / len(sorted_lats) if sorted_lats else 0.0
    error_rate = errors / float(total_requests) if total_requests > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "errors": errors,
        "error_rate": round(error_rate, 4),
        "min_ms": round(sorted_lats[0], 3) if sorted_lats else 0.0,
        "p50_ms": round(_percentile(sorted_lats, 0.50), 3),
        "p95_ms": round(_percentile(sorted_lats, 0.95), 3),
        "p99_ms": round(_percentile(sorted_lats, 0.99), 3),
        "max_ms": round(sorted_lats[-1], 3) if sorted_lats else 0.0,
        "mean_ms": round(mean_ms, 3),
        "rps": round(total_requests / wall_elapsed_s, 2),
    }


async def _execute_all_workloads(
    *,
    base_url: str,
    eng_key: str,
    sup_key: str,
    metrics_token: str,
    concurrency_levels: Sequence[int],
    requests_per_level: int,
    warmup_requests: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=50)
    async with httpx.AsyncClient(base_url=base_url, timeout=15.0, limits=limits) as client:
        eng_headers = {"X-API-Key": eng_key}
        sup_headers = {"X-API-Key": sup_key}

        async def wl_health(i: int) -> bool:
            if i % 2 == 0:
                r1 = await client.get("/healthz")
                return r1.status_code == 200 and r1.json().get("status") == "alive"
            r2 = await client.get("/readyz")
            return r2.status_code == 200 and r2.json().get("status") == "ready"

        async def wl_rag_authorized(_i: int) -> bool:
            r = await client.post(
                "/api/v1/rag/query",
                json={"query": "What is the minimum required password length?", "top_k": 3},
                headers=eng_headers,
            )
            if r.status_code != 200:
                return False
            body = r.json()
            return body.get("abstained") is False and "token_usage" in body

        async def wl_rag_rbac_denial(_i: int) -> bool:
            r = await client.post(
                "/api/v1/rag/query",
                json={"query": "What is the Project Helios acquisition valuation cap?", "top_k": 3},
                headers=eng_headers,
            )
            if r.status_code != 200:
                return False
            body = r.json()
            return body.get("abstained") is True

        async def wl_agent_safe_read(_i: int) -> bool:
            r = await client.post(
                "/api/v1/agent/run",
                json={"goal": "Check inventory for SKU-GPU-H100 and SKU-NET-SW400."},
                headers=eng_headers,
            )
            if r.status_code != 200:
                return False
            body = r.json()
            tool_steps = [s for s in body.get("steps", []) if s.get("action_type") == "TOOL_CALL"]
            return (
                body.get("status") == "COMPLETED"
                and len(tool_steps) >= 2
                and all(s.get("status") == "OK" for s in tool_steps)
            )

        async def wl_agent_hitl_roundtrip(i: int) -> bool:
            r_run = await client.post(
                "/api/v1/agent/run",
                json={"goal": "Reserve 1 unit of SKU-BENCH-NODE for LEAD-101."},
                headers=eng_headers,
            )
            if r_run.status_code != 200:
                return False
            run_body = r_run.json()
            if run_body.get("status") != "AWAITING_APPROVAL":
                return False
            approval_id = run_body["pending_approval"]["approval_id"]

            r_dec = await client.post(
                f"/api/v1/agent/approvals/{approval_id}/decide",
                json={"decision": "APPROVE", "reason": "Automated benchmark calibration approval"},
                headers=sup_headers,
            )
            if r_dec.status_code != 200:
                return False
            return r_dec.json().get("status") == "COMPLETED"

        workload_fns: Dict[str, Callable[[int], Coroutine[Any, Any, bool]]] = {
            "health_and_readiness": wl_health,
            "rag_authorized_query": wl_rag_authorized,
            "rag_rbac_denial": wl_rag_rbac_denial,
            "agent_safe_read_workflow": wl_agent_safe_read,
            "agent_hitl_approval_roundtrip": wl_agent_hitl_roundtrip,
        }

        # Warmup phase (discarded from latency distributions)
        for w_fn in workload_fns.values():
            for w_idx in range(warmup_requests):
                await w_fn(w_idx)

        workloads_report: Dict[str, Any] = {}
        for w_name in WORKLOAD_NAMES:
            slo_ms = SLO_P95_MS[w_name]
            w_fn = workload_fns[w_name]
            tier_results: List[Dict[str, Any]] = []
            for c_level in concurrency_levels:
                tier = await _run_concurrency_tier(
                    concurrency=c_level,
                    total_requests=requests_per_level,
                    worker_fn=w_fn,
                )
                tier["slo_p95_target_ms"] = slo_ms
                tier["passed_slo"] = bool(tier["error_rate"] == 0.0 and tier["p95_ms"] <= slo_ms)
                tier_results.append(tier)

            workloads_report[w_name] = {
                "slo_p95_target_ms": slo_ms,
                "passed_all_tiers": all(t["passed_slo"] for t in tier_results),
                "concurrency_results": tier_results,
            }

        # Scrape /metrics from the live Uvicorn subprocess
        m_resp = await client.get("/metrics", headers={"Authorization": f"Bearer {metrics_token}"})
        m_text = m_resp.text if m_resp.status_code == 200 else ""
        telemetry_verification = {
            "scraped_live_uvicorn_metrics": m_resp.status_code == 200,
            "http_requests_total": int(_sum_prometheus_metric(m_text, "fde_http_requests_total")),
            "http_errors_total": int(_sum_prometheus_metric(m_text, "fde_http_errors_total")),
            "rag_queries_total": int(_sum_prometheus_metric(m_text, "fde_rag_queries_total")),
            "agent_tool_calls_total": int(_sum_prometheus_metric(m_text, "fde_agent_tool_calls_total")),
            "llm_tokens_estimated_total": int(
                _sum_prometheus_metric(
                    m_text,
                    "fde_llm_tokens_total",
                    label_filter='token_type="total"',
                )
            ),
        }

        return workloads_report, telemetry_verification


def run_vpc_benchmarks(
    *,
    work_dir: Optional[Path] = None,
    concurrency_levels: Sequence[int] = (1, 5, 10, 20),
    requests_per_level: int = 12,
    warmup_requests: int = 3,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Launch a live Uvicorn HTTP server subprocess on an ephemeral loopback port,
    drive concurrent HTTP workloads across all 5 VPC workflows, verify Prometheus
    telemetry counters, and write the structured JSON benchmark report.
    """
    temp_ctx = None
    if work_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(
            prefix="fde_vpc_bench_",
            ignore_cleanup_errors=True,
        )
        base_dir = Path(temp_ctx.name)
    else:
        base_dir = Path(work_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    try:
        var_dir = base_dir / "var"
        secrets_dir = base_dir / "secrets"
        keys_file, metrics_file = init_dev_secret_files(secrets_dir, overwrite=True)
        bundle = load_secret_bundle(
            env="production",
            api_keys_file=keys_file,
            metrics_token_file=metrics_file,
        )
        raw_keys_map = json.loads(keys_file.read_text(encoding="utf-8"))
        eng_key = next(
            k for k, v in raw_keys_map.items() if v["principal_id"] == "acme_field_engineer"
        )
        sup_key = next(
            k for k, v in raw_keys_map.items() if v["principal_id"] == "acme_exec_supervisor"
        )

        port = _find_free_port()
        base_url = f"http://127.0.0.1:{port}"
        repo_root = Path(__file__).resolve().parents[2]

        proc_env = os.environ.copy()
        proc_env.update(
            {
                "FDE_ENV": "production",
                "FDE_VAR_DIR": str(var_dir),
                "FDE_AIRGAPPED_MODE": "true",                "FDE_AUTO_SEED": "true",
                "FDE_ALLOW_SAMPLE_SEED": "true",
                "FDE_API_KEYS_FILE": str(keys_file),
                "FDE_METRICS_TOKEN_FILE": str(metrics_file),
                "FDE_RATE_LIMIT_RPS": "10000.0",
                "FDE_RATE_LIMIT_BURST": "10000",
            }
        )

        stderr_log_path = base_dir / "uvicorn_stderr.log"
        stderr_fh = open(stderr_log_path, "w+", encoding="utf-8")

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.vpc_service.app:create_vpc_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
                "--log-level",
                "warning",
            ],
            cwd=str(repo_root),
            env=proc_env,
            stdout=subprocess.DEVNULL,
            stderr=stderr_fh,
        )

        try:
            # Wait for /readyz == 200 on the live Uvicorn subprocess
            ready = False
            deadline = time.perf_counter() + 30.0
            while time.perf_counter() < deadline:
                if proc.poll() is not None:
                    stderr_fh.flush()
                    stderr_fh.seek(0)
                    stderr_out = stderr_fh.read()
                    raise RuntimeError(f"Uvicorn subprocess exited prematurely: {stderr_out}")
                try:
                    r = httpx.get(f"{base_url}/readyz", timeout=1.0)
                    if r.status_code == 200 and r.json().get("status") == "ready":
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(0.15)

            if not ready:
                raise RuntimeError("Timed out waiting for live Uvicorn subprocess /readyz == 200")

            workloads_report, telemetry_verification = asyncio.run(
                _execute_all_workloads(
                    base_url=base_url,
                    eng_key=eng_key,
                    sup_key=sup_key,
                    metrics_token=bundle.metrics_token or "",
                    concurrency_levels=concurrency_levels,
                    requests_per_level=requests_per_level,
                    warmup_requests=warmup_requests,
                )
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            stderr_fh.close()

        overall_passed = all(w["passed_all_tiers"] for w in workloads_report.values()) and bool(
            telemetry_verification["scraped_live_uvicorn_metrics"]
        )

        report: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "service": "datheon-fde-vpc-gateway",
            "overall_passed": overall_passed,
            "environment": {
                "execution_mode": "local_uvicorn_subprocess",
                "containerized": False,
                "docker_cli_available": shutil.which("docker") is not None,
                "embedding_backend": "DeterministicHashEmbedding",
                "reranker_backend": "LexicalCrossEncoder",
                "planner_backend": "RuleBasedReActPlanner",
                "sqlite_journal_mode": "WAL",
                "sqlite_busy_timeout_ms": 5000,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count() or 1,
                "concurrency_levels": list(concurrency_levels),
                "requests_per_level": requests_per_level,
                "warmup_requests": warmup_requests,
            },
            "slo_targets_p95_ms": SLO_P95_MS,
            "workloads": workloads_report,
            "telemetry_verification": telemetry_verification,
        }

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        return report
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
