from __future__ import annotations

import json
from pathlib import Path

from src.agent.cli import main as cli_main
from src.agent.eval import run_agent_evaluation


def test_10_scenario_evaluation_suite_achieves_100_percent_and_zero_unauthorized_mutations(
    tmp_path: Path,
) -> None:
    report = run_agent_evaluation(work_dir=tmp_path / "eval_run")
    assert report["total_scenarios"] == 10
    assert report["passed_scenarios"] == 10
    assert report["scenario_pass_rate"] == 100.0
    assert report["unauthorized_mutations"] == 0
    assert len(report["scenarios"]) == 10
    for item in report["scenarios"]:
        assert item["passed"] is True, f"Scenario {item['scenario_id']} failed: {item}"


def test_cli_subcommands_seed_run_approve_sessions_and_eval(tmp_path: Path) -> None:
    erp_db = tmp_path / "cli_erp.db"
    sess_db = tmp_path / "cli_sessions.db"
    eval_out = tmp_path / "agent_eval.json"

    # 1. seed
    rc_seed = cli_main(["seed", "--erp-db", str(erp_db)])
    assert rc_seed == 0
    assert erp_db.exists()

    # 2. run (read-only)
    rc_read = cli_main(
        [
            "run",
            "--erp-db",
            str(erp_db),
            "--session-db",
            str(sess_db),
            "--goal",
            "Check inventory for SKU-GPU-H100 and inspect LEAD-101",
        ]
    )
    assert rc_read == 0

    # 3. run (HITL pause) -> approve -> sessions
    rc_write = cli_main(
        [
            "run",
            "--erp-db",
            str(erp_db),
            "--session-db",
            str(sess_db),
            "--goal",
            "Reserve 2 units of SKU-GPU-H100 for LEAD-101",
        ]
    )
    assert rc_write == 0

    from src.agent.session_store import SessionStore

    store = SessionStore(sess_db)
    pending_list = [a for a in store.list_approvals() if a.status == "PENDING"]
    assert len(pending_list) == 1
    approval_id = pending_list[0].approval_id
    session_id = pending_list[0].session_id

    rc_approve = cli_main(
        [
            "approve",
            "--erp-db",
            str(erp_db),
            "--session-db",
            str(sess_db),
            "--approval-id",
            approval_id,
            "--notes",
            "Approved via CLI test",
        ]
    )
    assert rc_approve == 0

    rc_sessions = cli_main(
        [
            "sessions",
            "--session-db",
            str(sess_db),
            "--session-id",
            session_id,
        ]
    )
    assert rc_sessions == 0

    # 4. eval
    rc_eval = cli_main(["eval", "--out", str(eval_out)])
    assert rc_eval == 0
    assert eval_out.exists()
    saved = json.loads(eval_out.read_text(encoding="utf-8"))
    assert saved["scenario_pass_rate"] == 100.0
    assert saved["unauthorized_mutations"] == 0
