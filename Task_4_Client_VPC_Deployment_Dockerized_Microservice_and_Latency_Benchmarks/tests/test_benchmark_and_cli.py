"""Tests for the live-Uvicorn subprocess concurrency benchmark suite and Rich CLI."""
from __future__ import annotations

import json
from pathlib import Path

from src.vpc_service.benchmark import WORKLOAD_NAMES, run_vpc_benchmarks
from src.vpc_service.cli import main as cli_main


def test_live_uvicorn_subprocess_benchmark_all_5_workloads_and_environment_metadata(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "vpc_latency_benchmark.json"
    report = run_vpc_benchmarks(
        work_dir=tmp_path / "bench_var",
        concurrency_levels=(1, 4),
        requests_per_level=4,
        warmup_requests=2,
        output_path=report_path,
    )

    assert report_path.is_file()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["overall_passed"] is True
    assert report["overall_passed"] is True

    # Verify explicit environment metadata (honest reporting of local subprocess vs Docker)
    env_meta = loaded["environment"]
    assert env_meta["execution_mode"] == "local_uvicorn_subprocess"
    assert env_meta["containerized"] is False
    assert isinstance(env_meta["docker_cli_available"], bool)
    assert env_meta["embedding_backend"] == "DeterministicHashEmbedding"
    assert env_meta["reranker_backend"] == "LexicalCrossEncoder"
    assert "python_version" in env_meta
    assert "platform" in env_meta
    assert env_meta["cpu_count"] >= 1

    # Verify all 5 workloads ran across both concurrency levels with 0% unexpected errors
    workloads = loaded["workloads"]
    assert set(workloads.keys()) == set(WORKLOAD_NAMES)
    for w_name in WORKLOAD_NAMES:
        tiers = workloads[w_name]["concurrency_results"]
        assert [t["concurrency"] for t in tiers] == [1, 4]
        for tier in tiers:
            assert tier["total_requests"] == 4
            assert tier["error_rate"] == 0.0
            assert tier["p50_ms"] > 0.0
            assert tier["p95_ms"] >= tier["p50_ms"]
            assert tier["p99_ms"] >= tier["p95_ms"]
            assert tier["rps"] > 0.0
            assert tier["passed_slo"] is True

    # Verify Prometheus /metrics scrape verification from the live subprocess
    telem = loaded["telemetry_verification"]
    assert telem["scraped_live_uvicorn_metrics"] is True
    assert telem["http_requests_total"] >= 40
    assert telem["rag_queries_total"] >= 16
    assert telem["agent_tool_calls_total"] >= 16
    assert telem["llm_tokens_estimated_total"] > 0


def test_cli_subcommands_init_secrets_check_seed_verify_and_benchmark(tmp_path: Path) -> None:
    secrets_dir = tmp_path / "secrets"
    var_dir = tmp_path / "var"
    report_out = tmp_path / "cli_bench.json"

    # 1. init-dev-secrets
    rc_init = cli_main(["init-dev-secrets", "--out-dir", str(secrets_dir)])
    assert rc_init == 0
    keys_file = secrets_dir / "fde_api_keys.json"
    metrics_file = secrets_dir / "fde_metrics_token.txt"
    assert keys_file.is_file()
    assert metrics_file.is_file()

    # 2. check-secrets in production mode
    rc_check = cli_main(
        [
            "check-secrets",
            "--env",
            "production",
            "--api-keys-file",
            str(keys_file),
            "--metrics-token-file",
            str(metrics_file),
        ]
    )
    assert rc_check == 0

    # 3. seed
    rc_seed = cli_main(["seed", "--var-dir", str(var_dir)])
    assert rc_seed == 0
    assert (var_dir / "acme_corp_erp.db").is_file()
    assert (var_dir / "globex_inc_erp.db").is_file()

    # 4. verify-config
    rc_verify = cli_main(["verify-config"])
    assert rc_verify == 0

    # 5. benchmark via CLI
    rc_bench = cli_main(
        [
            "benchmark",
            "--out",
            str(report_out),
            "--concurrency",
            "1,2",
            "--requests-per-level",
            "2",
            "--warmup",
            "1",
        ]
    )
    assert rc_bench == 0
    assert report_out.is_file()
