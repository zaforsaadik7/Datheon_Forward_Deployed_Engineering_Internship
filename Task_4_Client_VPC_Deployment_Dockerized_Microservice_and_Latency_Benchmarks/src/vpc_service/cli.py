"""Command-line interface for Client VPC secret initialization, seeding, config verification, and benchmarking."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import List, Optional, Sequence

from rich.console import Console
from rich.table import Table
import yaml

from scripts.seed_vpc import ensure_vpc_seeded
from src.vpc_service.benchmark import WORKLOAD_NAMES, run_vpc_benchmarks
from src.vpc_service.secrets import init_dev_secret_files, load_secret_bundle
from src.vpc_service.telemetry import VPCTelemetry


def _verify_vpc_configs(root_dir: Path, console: Console) -> int:
    dockerfile = root_dir / "Dockerfile"
    compose = root_dir / "docker-compose.yml"
    prom = root_dir / "deploy" / "prometheus.yml"
    rules = root_dir / "deploy" / "prometheus_rules.yml"

    for p in (dockerfile, compose, prom, rules):
        if not p.is_file():
            console.print(f"[red]Missing required configuration file: {p}[/red]")
            return 1

    compose_cfg = yaml.safe_load(compose.read_text(encoding="utf-8"))
    prom_cfg = yaml.safe_load(prom.read_text(encoding="utf-8"))
    rules_cfg = yaml.safe_load(rules.read_text(encoding="utf-8"))

    allowed_metrics = set(VPCTelemetry.METRIC_NAMES) | {
        f"{name}_bucket" for name in VPCTelemetry.METRIC_NAMES
    } | {
        f"{name}_count" for name in VPCTelemetry.METRIC_NAMES
    } | {
        f"{name}_sum" for name in VPCTelemetry.METRIC_NAMES
    }

    alert_count = 0
    for grp in rules_cfg.get("groups", []):
        for rule in grp.get("rules", []):
            alert_count += 1
            refs = set(re.findall(r"\bfde_[a-z0-9_]+\b", rule.get("expr", "")))
            unknown = refs - allowed_metrics
            if unknown:
                console.print(f"[red]Alert {rule.get('alert')} references unknown metrics: {unknown}[/red]")
                return 1

    console.print(
        f"[green]Verified Dockerfile, docker-compose.yml ({len(compose_cfg.get('services', {}))} services), "
        f"prometheus.yml ({len(prom_cfg.get('scrape_configs', []))} scrape job), and "
        f"prometheus_rules.yml ({alert_count} alert rules).[/green]"
    )
    return 0


def _print_benchmark_summary(report: dict, console: Console) -> None:
    env_meta = report["environment"]
    console.print(
        f"[bold cyan]Datheon Client VPC Microservice Benchmark[/bold cyan] "
        f"(mode={env_meta['execution_mode']}, containerized={env_meta['containerized']}, "
        f"embedding={env_meta['embedding_backend']}, reranker={env_meta['reranker_backend']})"
    )

    table = Table(title="VPC Concurrency & Latency Benchmark Results (Live Uvicorn Subprocess)")
    table.add_column("Workload", style="cyan")
    table.add_column("Concurrency", justify="right")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("p99 (ms)", justify="right")
    table.add_column("SLO p95 (ms)", justify="right")
    table.add_column("RPS", justify="right")
    table.add_column("Error Rate", justify="right")
    table.add_column("Status", justify="center")

    workloads = report["workloads"]
    for w_name in WORKLOAD_NAMES:
        w_data = workloads[w_name]
        for tier in w_data["concurrency_results"]:
            status_str = "[green]PASS[/green]" if tier["passed_slo"] else "[red]FAIL[/red]"
            table.add_row(
                w_name,
                str(tier["concurrency"]),
                f"{tier['p50_ms']:.2f}",
                f"{tier['p95_ms']:.2f}",
                f"{tier['p99_ms']:.2f}",
                f"{tier['slo_p95_target_ms']:.1f}",
                f"{tier['rps']:.1f}",
                f"{tier['error_rate'] * 100.0:.1f}%",
                status_str,
            )

    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.vpc_service.cli",
        description="Datheon Client VPC AI Microservice Gateway CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-dev-secrets", help="Generate random local secret files (chmod 0644)")
    p_init.add_argument("--out-dir", type=Path, default=Path("deploy/secrets"))

    p_seed = sub.add_parser("seed", help="Idempotently seed multi-tenant ChromaDB and SQLite ERP databases")
    p_seed.add_argument("--var-dir", type=Path, default=Path("var"))
    p_seed.add_argument("--env", type=str, default="development")

    p_check = sub.add_parser("check-secrets", help="Validate secret files and print redacted summary")
    p_check.add_argument("--env", type=str, default="production")
    p_check.add_argument("--api-keys-file", type=Path, default=Path("deploy/secrets/fde_api_keys.json"))
    p_check.add_argument("--metrics-token-file", type=Path, default=Path("deploy/secrets/fde_metrics_token.txt"))

    p_verify = sub.add_parser("verify-config", help="Validate Dockerfile, Compose, and Prometheus alert rules")
    p_verify.add_argument("--root-dir", type=Path, default=Path(__file__).resolve().parents[2])

    p_bench = sub.add_parser("benchmark", help="Run live-Uvicorn subprocess HTTP concurrency benchmarks")
    p_bench.add_argument("--out", type=Path, default=Path("reports/vpc_latency_benchmark.json"))
    p_bench.add_argument("--concurrency", type=str, default="1,5,10,20")
    p_bench.add_argument("--requests-per-level", type=int, default=12)
    p_bench.add_argument("--warmup", type=int, default=3)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.command == "init-dev-secrets":
        info = init_dev_secret_files(args.out_dir, overwrite=True)
        console.print(f"[green]Initialized secret files in {args.out_dir}:[/green] {info}")
        return 0

    if args.command == "seed":
        summary = ensure_vpc_seeded(
            var_dir=args.var_dir,
            env=args.env,
            auto_seed=True,
            allow_sample_seed=True,
        )
        console.print(f"[green]Seeded VPC data stores in {args.var_dir}:[/green] {summary}")
        return 0

    if args.command == "check-secrets":
        bundle = load_secret_bundle(
            env=args.env,
            api_keys_file=args.api_keys_file,
            metrics_token_file=args.metrics_token_file,
        )
        console.print(f"[green]Validated SecretBundle:[/green] {bundle.to_redacted_dict()}")
        return 0

    if args.command == "verify-config":
        return _verify_vpc_configs(args.root_dir, console)

    if args.command == "benchmark":
        levels: List[int] = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
        report = run_vpc_benchmarks(
            concurrency_levels=levels,
            requests_per_level=args.requests_per_level,
            warmup_requests=args.warmup,
            output_path=args.out,
        )
        _print_benchmark_summary(report, console)
        return 0 if report["overall_passed"] else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
