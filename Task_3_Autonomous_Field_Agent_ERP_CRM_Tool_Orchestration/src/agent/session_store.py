"""SQLite persistence for AgentSession state, ReActStep traces, and single-use expiring HITL approvals."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterator, List, Literal, Optional
import uuid

from .auth import require_supervisor
from .models import (
    AgentRunResult,
    OperatorPrincipal,
    PendingApproval,
    ReActStep,
    SessionStatus,
)
from .tools import canonical_args_sha256


class SessionNotFoundError(LookupError):
    """Raised when a session_id does not exist in SQLite (HTTP 404)."""
    pass


class ApprovalNotFoundError(LookupError):
    """Raised when an approval_id does not exist in SQLite (HTTP 404)."""
    pass


class ApprovalMismatchError(RuntimeError):
    """Raised when tool_name or tool_args do not match the approved args_sha256 (HTTP 409)."""
    pass


class ApprovalReplayError(RuntimeError):
    """Raised when an approval ticket is already decided or already consumed (HTTP 409)."""
    pass


class ApprovalExpiredError(RuntimeError):
    """Raised when an approval ticket has passed its expires_at timestamp (HTTP 409)."""
    pass


class SessionStore:
    """
    Transactional SQLite store for agent sessions, step traces, and HITL approvals.
    Ensures paused sessions survive across HTTP requests and worker processes, and enforces
    single-use consumption and TTL expiry on all PendingApproval tickets.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    operator_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    final_answer TEXT,
                    total_retries INTEGER NOT NULL DEFAULT 0,
                    validation_repairs INTEGER NOT NULL DEFAULT 0,
                    total_ms REAL NOT NULL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS react_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pending_approvals (
                    approval_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    requested_by TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_args_json TEXT NOT NULL,
                    args_sha256 TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    decided_by TEXT,
                    decision_notes TEXT,
                    decided_at TEXT
                );
                """
            )

    def create_session(self, operator_id: str, goal: str) -> str:
        session_id = f"SES-{uuid.uuid4().hex[:10].upper()}"
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_sessions (
                    session_id, operator_id, goal, status, final_answer,
                    total_retries, validation_repairs, total_ms, created_at, updated_at
                ) VALUES (?, ?, ?, 'RUNNING', NULL, 0, 0, 0.0, ?, ?)
                """,
                (session_id, operator_id, goal, now, now),
            )
        return session_id

    def append_step(self, session_id: str, step: ReActStep) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO react_steps (session_id, step_index, step_json) VALUES (?, ?, ?)",
                (session_id, step.step_index, step.model_dump_json()),
            )

    def update_last_step(self, session_id: str, step: ReActStep) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM react_steps WHERE session_id = ? AND step_index = ? ORDER BY id DESC LIMIT 1",
                (session_id, step.step_index),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE react_steps SET step_json = ? WHERE id = ?",
                    (step.model_dump_json(), int(row["id"])),
                )
            else:
                conn.execute(
                    "INSERT INTO react_steps (session_id, step_index, step_json) VALUES (?, ?, ?)",
                    (session_id, step.step_index, step.model_dump_json()),
                )

    def update_session_state(
        self,
        session_id: str,
        *,
        status: SessionStatus,
        final_answer: Optional[str] = None,
        total_retries: int = 0,
        validation_repairs: int = 0,
        total_ms: float = 0.0,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_sessions
                SET status = ?, final_answer = ?, total_retries = ?,
                    validation_repairs = ?, total_ms = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (status, final_answer, total_retries, validation_repairs, round(total_ms, 3), now, session_id),
            )

    def get_steps(self, session_id: str) -> List[ReActStep]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT step_json FROM react_steps WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [ReActStep(**json.loads(r["step_json"])) for r in rows]

    def create_pending_approval(
        self,
        *,
        session_id: str,
        step_index: int,
        requested_by: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        reason: str,
        ttl_seconds: int = 900,
        now_override: Optional[datetime] = None,
    ) -> PendingApproval:
        now_dt = now_override if now_override is not None else datetime.now(timezone.utc)
        exp_dt = now_dt + timedelta(seconds=ttl_seconds)
        approval_id = f"APR-{uuid.uuid4().hex[:8].upper()}"
        args_hash = canonical_args_sha256(tool_name, tool_args)

        approval = PendingApproval(
            approval_id=approval_id,
            session_id=session_id,
            step_index=step_index,
            requested_by=requested_by,
            tool_name=tool_name,
            tool_args=tool_args,
            args_sha256=args_hash,
            reason=reason,
            status="PENDING",
            created_at=now_dt.isoformat(),
            expires_at=exp_dt.isoformat(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pending_approvals (
                    approval_id, session_id, step_index, requested_by, tool_name,
                    tool_args_json, args_sha256, reason, status, created_at, expires_at,
                    decided_by, decision_notes, decided_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (
                    approval.approval_id,
                    approval.session_id,
                    approval.step_index,
                    approval.requested_by,
                    approval.tool_name,
                    json.dumps(approval.tool_args, sort_keys=True),
                    approval.args_sha256,
                    approval.reason,
                    approval.status,
                    approval.created_at,
                    approval.expires_at,
                ),
            )
        return approval

    @staticmethod
    def _row_to_approval(row: sqlite3.Row) -> PendingApproval:
        return PendingApproval(
            approval_id=row["approval_id"],
            session_id=row["session_id"],
            step_index=int(row["step_index"]),
            requested_by=row["requested_by"],
            tool_name=row["tool_name"],
            tool_args=json.loads(row["tool_args_json"]),
            args_sha256=row["args_sha256"],
            reason=row["reason"],
            status=row["status"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            decided_by=row["decided_by"],
            decision_notes=row["decision_notes"],
            decided_at=row["decided_at"],
        )

    def get_approval(self, approval_id: str) -> PendingApproval:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pending_approvals WHERE approval_id = ?",
                (approval_id.strip().upper(),),
            ).fetchone()
            if row is None:
                raise ApprovalNotFoundError(f"Approval ticket '{approval_id}' not found.")
            return self._row_to_approval(row)

    def list_approvals(
        self,
        session_id: Optional[str] = None,
        requested_by: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[PendingApproval]:
        clauses: List[str] = []
        params: List[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id.strip().upper())
        if requested_by:
            clauses.append("requested_by = ?")
            params.append(requested_by.strip())
        if status:
            clauses.append("status = ?")
            params.append(status.strip().upper())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = "ORDER BY created_at ASC" if session_id else "ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM pending_approvals {where_sql} {order_sql}",
                tuple(params),
            ).fetchall()
            return [self._row_to_approval(r) for r in rows]

    def _check_not_expired(self, conn: sqlite3.Connection, approval: PendingApproval) -> None:
        now_dt = datetime.now(timezone.utc)
        exp_dt = datetime.fromisoformat(approval.expires_at)
        if now_dt >= exp_dt:
            conn.execute(
                "UPDATE pending_approvals SET status = 'EXPIRED' WHERE approval_id = ?",
                (approval.approval_id,),
            )
            conn.commit()
            raise ApprovalExpiredError(
                f"Approval ticket '{approval.approval_id}' expired at {approval.expires_at}."
            )

    def decide_approval(
        self,
        *,
        approval_id: str,
        supervisor: OperatorPrincipal,
        decision: Literal["APPROVE", "REJECT"],
        notes: str = "",
    ) -> PendingApproval:
        require_supervisor(supervisor)
        norm_id = approval_id.strip().upper()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pending_approvals WHERE approval_id = ?",
                (norm_id,),
            ).fetchone()
            if row is None:
                raise ApprovalNotFoundError(f"Approval ticket '{norm_id}' not found.")

            current = self._row_to_approval(row)
            if current.status != "PENDING":
                raise ApprovalReplayError(
                    f"Approval ticket '{norm_id}' has already been processed (status='{current.status}')."
                )

            self._check_not_expired(conn, current)

            new_status = "APPROVED" if decision == "APPROVE" else "REJECTED"
            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                UPDATE pending_approvals
                SET status = ?, decided_by = ?, decision_notes = ?, decided_at = ?
                WHERE approval_id = ?
                """,
                (new_status, supervisor.operator_id, notes.strip(), now_iso, norm_id),
            )
            updated_row = conn.execute(
                "SELECT * FROM pending_approvals WHERE approval_id = ?",
                (norm_id,),
            ).fetchone()
            return self._row_to_approval(updated_row)

    def consume_approved_ticket(
        self,
        *,
        approval_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> PendingApproval:
        """
        Verify and atomically consume an APPROVED ticket for a single tool execution:
        - Rejects if status != 'APPROVED' (e.g. 'CONSUMED', 'PENDING', 'REJECTED', 'EXPIRED')
        - Rejects if expired
        - Rejects if canonical_args_sha256(tool_name, tool_args) != ticket.args_sha256
        - Marks status = 'CONSUMED' so it can never be replayed
        """
        norm_id = approval_id.strip().upper()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pending_approvals WHERE approval_id = ?",
                (norm_id,),
            ).fetchone()
            if row is None:
                raise ApprovalNotFoundError(f"Approval ticket '{norm_id}' not found.")

            current = self._row_to_approval(row)
            if current.status == "CONSUMED":
                raise ApprovalReplayError(
                    f"Approval ticket '{norm_id}' has already been consumed and cannot be replayed."
                )
            if current.status != "APPROVED":
                raise ApprovalReplayError(
                    f"Approval ticket '{norm_id}' is not in APPROVED state (status='{current.status}')."
                )

            self._check_not_expired(conn, current)

            actual_hash = canonical_args_sha256(tool_name, tool_args)
            if actual_hash != current.args_sha256 or tool_name != current.tool_name:
                raise ApprovalMismatchError(
                    f"Approval hash mismatch for ticket '{norm_id}': approved args_sha256='{current.args_sha256[:12]}...', got '{actual_hash[:12]}...'."
                )

            conn.execute(
                "UPDATE pending_approvals SET status = 'CONSUMED' WHERE approval_id = ?",
                (norm_id,),
            )
            updated_row = conn.execute(
                "SELECT * FROM pending_approvals WHERE approval_id = ?",
                (norm_id,),
            ).fetchone()
            return self._row_to_approval(updated_row)

    def get_session_result(self, session_id: str) -> AgentRunResult:
        norm_id = session_id.strip().upper()
        with self._connect() as conn:
            sess_row = conn.execute(
                "SELECT * FROM agent_sessions WHERE session_id = ?",
                (norm_id,),
            ).fetchone()
            if sess_row is None:
                raise SessionNotFoundError(f"Agent session '{session_id}' not found.")

            steps = self.get_steps(norm_id)
            pending_row = conn.execute(
                "SELECT * FROM pending_approvals WHERE session_id = ? AND status = 'PENDING' ORDER BY created_at DESC LIMIT 1",
                (norm_id,),
            ).fetchone()
            pending = self._row_to_approval(pending_row) if pending_row is not None else None

            return AgentRunResult(
                session_id=sess_row["session_id"],
                operator_id=sess_row["operator_id"],
                goal=sess_row["goal"],
                status=sess_row["status"],
                final_answer=sess_row["final_answer"],
                steps=tuple(steps),
                pending_approval=pending,
                total_retries=int(sess_row["total_retries"]),
                validation_repairs=int(sess_row["validation_repairs"]),
                total_ms=float(sess_row["total_ms"]),
            )
