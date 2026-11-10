"""Rich CLI for the Autonomous Field Agent (`seed`, `run`, `approve`, `reject`, `sessions`, `eval`)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scripts.seed_db import seed_database
from src.agent.auth import OPERATORS
from src.agent.eval import run_agent_evaluation
from src.agent.http_client import ResilientERPClient
from src.agent.models import AgentRunResult
from src.agent.orchestrator import AgentOrchestrator
from src.agent.planner import OpenAIFunctionPlanner, RuleBasedReActPlanner
from src.agent.session_store import SessionStore
from src.agent.tools import ToolExecutor
from src.config import get_settings
from src.erp_crm.app import create_erp_crm_app
from src.erp_crm.database import ERPCRMDatabase

console = Console()


def _render_session(result: AgentRunResult) -> None:
    table = Table(title=f"Agent Session {result.session_id} [{result.status}]")
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("Action / Tool", style="magenta")
    table.add_column("Sensitivity", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Retries", justify="right")
    table.add_column("Thought / Observation", overflow="fold")

    for step in result.steps:
        label = step.tool_name if step.action_type == "TOOL_CALL" else "FINAL_ANSWER"
        detail = f"Thought: {step.thought}\nObs: {(step.observation or '')[:220]}"
        table.add_row(
            str(step.step_index),
            label or "-",
            step.sensitivity or "-",
            step.status,
            str(step.retries_used),
            detail,
        )
    console.print(table)
    if result.pending_approval is not None:
        pa = result.pending_approval
        console.print(
            Panel(
                f"[bold yellow]AWAITING SUPERVISOR APPROVAL[/bold yellow]\n"
                f"Approval ID : [bold]{pa.approval_id}[/bold]\n"
                f"Tool        : {pa.tool_name}({json.dumps(pa.tool_args)})\n"
                f"Args SHA256 : {pa.args_sha256}\n"
                f"Expires At  : {pa.expires_at}",
                title="HITL Gate",
            )
        )
    elif result.final_answer:
        console.print(Panel(result.final_answer, title="Final Answer", border_style="green"))


def _build_orchestrator(erp_db_path: Path, session_db_path: Path, use_openai: bool = False) -> AgentOrchestrator:
    if not erp_db_path.exists():
        seed_database(erp_db_path)
    erp_db = ERPCRMDatabase(erp_db_path)
    erp_app = create_erp_crm_app(db=erp_db)
    erp_client = ResilientERPClient(app=erp_app, max_retries=3, base_backoff_s=0.01)
    executor = ToolExecutor(erp_client)
    store = SessionStore(session_db_path)
    planner = OpenAIFunctionPlanner() if use_openai else RuleBasedReActPlanner()
    return AgentOrchestrator(executor=executor, session_store=store, planner=planner)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m src.agent.cli",
        description="Datheon Task 3: Autonomous Field Agent CLI (ERP/CRM Tool Orchestration & HITL)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # seed
    p_seed = sub.add_parser("seed", help="Seed or reset the Enterprise ERP/CRM/ITSM SQLite database")
    p_seed.add_argument("--erp-db", type=Path, default=settings.erp_db_path)

    # run
    p_run = sub.add_parser("run", help="Run an autonomous field agent goal")
    p_run.add_argument("--goal", type=str, required=True, help="Natural-language field engineering goal")
    p_run.add_argument("--operator", type=str, default="field_operator", choices=list(OPERATORS.keys()))
    p_run.add_argument("--erp-db", type=Path, default=settings.erp_db_path)
    p_run.add_argument("--session-db", type=Path, default=settings.session_db_path)
    p_run.add_argument("--openai", action="store_true", help="Use OpenAIFunctionPlanner instead of offline RuleBasedReActPlanner")

    # approve
    p_app = sub.add_parser("approve", help="Approve a pending HITL ticket as ops_supervisor and resume session")
    p_app.add_argument("--approval-id", type=str, required=True)
    p_app.add_argument("--notes", type=str, default="Approved by supervisor via CLI")
    p_app.add_argument("--supervisor", type=str, default="ops_supervisor", choices=list(OPERATORS.keys()))
    p_app.add_argument("--erp-db", type=Path, default=settings.erp_db_path)
    p_app.add_argument("--session-db", type=Path, default=settings.session_db_path)

    # reject
    p_rej = sub.add_parser("reject", help="Reject a pending HITL ticket as ops_supervisor and abort mutation")
    p_rej.add_argument("--approval-id", type=str, required=True)
    p_rej.add_argument("--notes", type=str, default="Rejected by supervisor via CLI")
    p_rej.add_argument("--supervisor", type=str, default="ops_supervisor", choices=list(OPERATORS.keys()))
    p_rej.add_argument("--erp-db", type=Path, default=settings.erp_db_path)
    p_rej.add_argument("--session-db", type=Path, default=settings.session_db_path)

    # sessions
    p_sess = sub.add_parser("sessions", help="Inspect a persisted agent session or list pending HITL approvals")
    p_sess.add_argument("--session-id", type=str, default=None)
    p_sess.add_argument("--session-db", type=Path, default=settings.session_db_path)

    # eval
    p_eval = sub.add_parser("eval", help="Run the 10-scenario operational and security evaluation suite")
    p_eval.add_argument(
        "--out",
        type=Path,
        default=Path("reports/agent_eval.json"),
        help="Output path for JSON evaluation report",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "seed":
        summary = seed_database(args.erp_db)
        console.print(f"[green]Seeded ERP/CRM database at {args.erp_db}:[/green] {summary}")
        return 0

    if args.command == "run":
        orch = _build_orchestrator(args.erp_db, args.session_db, use_openai=args.openai)
        operator = OPERATORS[args.operator]
        res = orch.run_goal(operator=operator, goal=args.goal)
        _render_session(res)
        return 0

    if args.command in {"approve", "reject"}:
        orch = _build_orchestrator(args.erp_db, args.session_db)
        supervisor = OPERATORS[args.supervisor]
        decision = "APPROVE" if args.command == "approve" else "REJECT"
        res = orch.decide_and_resume(
            approval_id=args.approval_id,
            supervisor=supervisor,
            decision=decision,
            notes=args.notes,
        )
        _render_session(res)
        return 0

    if args.command == "sessions":
        store = SessionStore(args.session_db)
        if args.session_id:
            res = store.get_session_result(args.session_id)
            _render_session(res)
        else:
            approvals = store.list_approvals()
            table = Table(title="Persisted HITL Approvals")
            table.add_column("Approval ID", style="cyan")
            table.add_column("Session ID")
            table.add_column("Tool")
            table.add_column("Status", style="yellow")
            table.add_column("Requested By")
            table.add_column("Decided By")
            for a in approvals:
                table.add_row(
                    a.approval_id,
                    a.session_id,
                    a.tool_name,
                    a.status,
                    a.requested_by,
                    a.decided_by or "-",
                )
            console.print(table)
        return 0

    if args.command == "eval":
        report = run_agent_evaluation()
        out_path: Path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        table = Table(title="Datheon Task 3 — 10-Scenario Autonomous Field Agent Evaluation")
        table.add_column("Scenario ID", style="cyan")
        table.add_column("Description")
        table.add_column("Status", style="magenta")
        table.add_column("Steps", justify="right")
        table.add_column("Retries", justify="right")
        table.add_column("Repairs", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Result", style="bold")

        for sc in report["scenarios"]:
            badge = "[green]PASS[/green]" if sc["passed"] else "[red]FAIL[/red]"
            table.add_row(
                sc["scenario_id"],
                sc["title"],
                str(sc["status"]),
                str(sc["steps"]),
                str(sc["retries"]),
                str(sc["validation_repairs"]),
                f"{sc['latency_ms']:.2f}",
                badge,
            )
        console.print(table)
        console.print(
            f"[bold green]Pass Rate:[/bold green] {report['scenario_pass_rate']}% "
            f"({report['passed_scenarios']}/{report['total_scenarios']}) | "
            f"[bold]Unauthorized Mutations:[/bold] {report['unauthorized_mutations']} | "
            f"[bold]Saved:[/bold] {out_path}"
        )
        return 0 if report["passed_scenarios"] == report["total_scenarios"] else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
