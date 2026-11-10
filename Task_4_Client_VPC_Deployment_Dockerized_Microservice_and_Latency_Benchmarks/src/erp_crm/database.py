"""SQLite persistence layer for Mock Enterprise ERP, CRM, and ITSM with idempotency and state transitions."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterator, List, Optional
import uuid

from .models import (
    ALLOWED_LEAD_TRANSITIONS,
    CreateTicketRequest,
    InventoryItem,
    LeadRecord,
    LeadTransitionReceipt,
    ReservationReceipt,
    ReserveInventoryRequest,
    TicketRecord,
    UpdateLeadStatusRequest,
)


class NotFoundError(LookupError):
    """Raised when an SKU, Lead, or Ticket ID does not exist (HTTP 404)."""
    pass


class StockConflictError(RuntimeError):
    """Raised when requested reservation quantity exceeds available stock (HTTP 409)."""
    pass


class IllegalStateTransitionError(RuntimeError):
    """Raised when a CRM lead status update violates ALLOWED_LEAD_TRANSITIONS (HTTP 409)."""
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ERPCRMDatabase:
    """Transactional SQLite repository for ERP inventory, CRM leads, ITSM tickets, and idempotency keys."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS inventory (
                    sku TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    stock_on_hand INTEGER NOT NULL,
                    reserved INTEGER NOT NULL,
                    available INTEGER NOT NULL,
                    unit_price_usd REAL NOT NULL,
                    warehouse TEXT NOT NULL,
                    reorder_threshold INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reservations (
                    reservation_id TEXT PRIMARY KEY,
                    sku TEXT NOT NULL,
                    lead_id TEXT NOT NULL,
                    quantity_reserved INTEGER NOT NULL,
                    remaining_available INTEGER NOT NULL,
                    total_reserved_for_sku INTEGER NOT NULL,
                    unit_price_usd REAL NOT NULL,
                    total_hold_value_usd REAL NOT NULL,
                    warehouse TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS leads (
                    lead_id TEXT PRIMARY KEY,
                    company TEXT NOT NULL,
                    contact_email TEXT NOT NULL,
                    status TEXT NOT NULL,
                    deal_value_usd REAL NOT NULL,
                    owner TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    related_sku TEXT,
                    related_lead_id TEXT,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    idem_key TEXT PRIMARY KEY,
                    endpoint TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def reset_and_seed(self, seed_data: Dict[str, Any]) -> None:
        now = _now_iso()
        with self._connect() as conn:
            conn.executescript(
                """
                DELETE FROM inventory;
                DELETE FROM reservations;
                DELETE FROM leads;
                DELETE FROM tickets;
                DELETE FROM idempotency_keys;
                """
            )
            for item in seed_data.get("inventory", []):
                conn.execute(
                    """
                    INSERT INTO inventory (
                        sku, name, stock_on_hand, reserved, available,
                        unit_price_usd, warehouse, reorder_threshold
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["sku"],
                        item["name"],
                        int(item["stock_on_hand"]),
                        int(item["reserved"]),
                        int(item["available"]),
                        float(item["unit_price_usd"]),
                        item["warehouse"],
                        int(item["reorder_threshold"]),
                    ),
                )
            for lead in seed_data.get("leads", []):
                conn.execute(
                    """
                    INSERT INTO leads (
                        lead_id, company, contact_email, status,
                        deal_value_usd, owner, notes, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lead["lead_id"],
                        lead["company"],
                        lead["contact_email"],
                        lead["status"],
                        float(lead["deal_value_usd"]),
                        lead["owner"],
                        lead.get("notes", ""),
                        now,
                    ),
                )
            for tck in seed_data.get("tickets", []):
                conn.execute(
                    """
                    INSERT INTO tickets (
                        ticket_id, title, description, priority, status,
                        related_sku, related_lead_id, created_by, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tck["ticket_id"],
                        tck["title"],
                        tck["description"],
                        tck["priority"],
                        tck["status"],
                        tck.get("related_sku"),
                        tck.get("related_lead_id"),
                        tck.get("created_by", "system"),
                        now,
                    ),
                )

    def get_inventory(self, sku: str) -> InventoryItem:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM inventory WHERE sku = ?", (sku.strip().upper(),)).fetchone()
            if row is None:
                raise NotFoundError(f"Inventory SKU '{sku}' not found.")
            return InventoryItem(**dict(row))

    def search_inventory(self, query: Optional[str] = None) -> List[InventoryItem]:
        with self._connect() as conn:
            if query and query.strip():
                q = f"%{query.strip().lower()}%"
                rows = conn.execute(
                    "SELECT * FROM inventory WHERE lower(sku) LIKE ? OR lower(name) LIKE ? OR lower(warehouse) LIKE ? ORDER BY sku",
                    (q, q, q),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM inventory ORDER BY sku").fetchall()
            return [InventoryItem(**dict(r)) for r in rows]

    def reserve_inventory(
        self,
        sku: str,
        req: ReserveInventoryRequest,
        idempotency_key: Optional[str] = None,
    ) -> ReservationReceipt:
        norm_sku = sku.strip().upper()
        endpoint_tag = f"POST:/erp/v1/inventory/{norm_sku}/reserve"
        with self._connect() as conn:
            if idempotency_key and idempotency_key.strip():
                cached = conn.execute(
                    "SELECT response_json FROM idempotency_keys WHERE idem_key = ? AND endpoint = ?",
                    (idempotency_key.strip(), endpoint_tag),
                ).fetchone()
                if cached is not None:
                    return ReservationReceipt(**json.loads(cached["response_json"]))

            row = conn.execute("SELECT * FROM inventory WHERE sku = ?", (norm_sku,)).fetchone()
            if row is None:
                raise NotFoundError(f"Inventory SKU '{norm_sku}' not found.")

            available = int(row["available"])
            reserved = int(row["reserved"])
            if req.quantity > available:
                raise StockConflictError(
                    f"Insufficient available stock for SKU '{norm_sku}': requested {req.quantity}, available {available}."
                )

            new_available = available - req.quantity
            new_reserved = reserved + req.quantity
            unit_price = float(row["unit_price_usd"])
            hold_val = round(unit_price * req.quantity, 2)
            res_id = f"RES-{uuid.uuid4().hex[:8].upper()}"
            now = _now_iso()

            conn.execute(
                "UPDATE inventory SET available = ?, reserved = ? WHERE sku = ?",
                (new_available, new_reserved, norm_sku),
            )

            receipt = ReservationReceipt(
                reservation_id=res_id,
                sku=norm_sku,
                lead_id=req.lead_id,
                quantity_reserved=req.quantity,
                remaining_available=new_available,
                total_reserved_for_sku=new_reserved,
                unit_price_usd=unit_price,
                total_hold_value_usd=hold_val,
                warehouse=str(row["warehouse"]),
                created_at=now,
            )

            conn.execute(
                """
                INSERT INTO reservations (
                    reservation_id, sku, lead_id, quantity_reserved, remaining_available,
                    total_reserved_for_sku, unit_price_usd, total_hold_value_usd, warehouse, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.reservation_id,
                    receipt.sku,
                    receipt.lead_id,
                    receipt.quantity_reserved,
                    receipt.remaining_available,
                    receipt.total_reserved_for_sku,
                    receipt.unit_price_usd,
                    receipt.total_hold_value_usd,
                    receipt.warehouse,
                    receipt.created_at,
                ),
            )

            if idempotency_key and idempotency_key.strip():
                conn.execute(
                    "INSERT INTO idempotency_keys (idem_key, endpoint, response_json, created_at) VALUES (?, ?, ?, ?)",
                    (idempotency_key.strip(), endpoint_tag, receipt.model_dump_json(), now),
                )

            return receipt

    def get_lead(self, lead_id: str) -> LeadRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM leads WHERE lead_id = ?", (lead_id.strip().upper(),)).fetchone()
            if row is None:
                raise NotFoundError(f"CRM Lead '{lead_id}' not found.")
            return LeadRecord(**dict(row))

    def search_leads(self, query: Optional[str] = None) -> List[LeadRecord]:
        with self._connect() as conn:
            if query and query.strip():
                q = f"%{query.strip().lower()}%"
                rows = conn.execute(
                    "SELECT * FROM leads WHERE lower(lead_id) LIKE ? OR lower(company) LIKE ? OR lower(status) LIKE ? ORDER BY lead_id",
                    (q, q, q),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM leads ORDER BY lead_id").fetchall()
            return [LeadRecord(**dict(r)) for r in rows]

    def update_lead_status(
        self,
        lead_id: str,
        req: UpdateLeadStatusRequest,
        idempotency_key: Optional[str] = None,
    ) -> LeadTransitionReceipt:
        norm_id = lead_id.strip().upper()
        endpoint_tag = f"PATCH:/crm/v1/leads/{norm_id}/status"
        with self._connect() as conn:
            if idempotency_key and idempotency_key.strip():
                cached = conn.execute(
                    "SELECT response_json FROM idempotency_keys WHERE idem_key = ? AND endpoint = ?",
                    (idempotency_key.strip(), endpoint_tag),
                ).fetchone()
                if cached is not None:
                    return LeadTransitionReceipt(**json.loads(cached["response_json"]))

            row = conn.execute("SELECT * FROM leads WHERE lead_id = ?", (norm_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"CRM Lead '{norm_id}' not found.")

            current_status = str(row["status"])
            target_status = req.status
            allowed_next = ALLOWED_LEAD_TRANSITIONS.get(current_status, frozenset())

            if target_status not in allowed_next:
                allowed_list = sorted(allowed_next) if allowed_next else ["NONE (terminal state)"]
                raise IllegalStateTransitionError(
                    f"Illegal lead status transition for '{norm_id}': cannot transition from '{current_status}' to '{target_status}'. Allowed next states from '{current_status}': {allowed_list}."
                )

            new_deal_val = float(req.deal_value_usd) if req.deal_value_usd is not None else float(row["deal_value_usd"])
            existing_notes = str(row["notes"])
            combined_notes = (
                f"{existing_notes} | {req.notes.strip()}".strip(" |")
                if req.notes.strip()
                else existing_notes
            )
            now = _now_iso()
            trans_id = f"TRN-{uuid.uuid4().hex[:8].upper()}"

            conn.execute(
                "UPDATE leads SET status = ?, deal_value_usd = ?, notes = ?, updated_at = ? WHERE lead_id = ?",
                (target_status, new_deal_val, combined_notes, now, norm_id),
            )

            receipt = LeadTransitionReceipt(
                transition_id=trans_id,
                lead_id=norm_id,
                company=str(row["company"]),
                previous_status=current_status,  # type: ignore[arg-type]
                status=target_status,
                deal_value_usd=new_deal_val,
                notes=combined_notes,
                updated_at=now,
            )

            if idempotency_key and idempotency_key.strip():
                conn.execute(
                    "INSERT INTO idempotency_keys (idem_key, endpoint, response_json, created_at) VALUES (?, ?, ?, ?)",
                    (idempotency_key.strip(), endpoint_tag, receipt.model_dump_json(), now),
                )

            return receipt

    def create_ticket(
        self,
        req: CreateTicketRequest,
        idempotency_key: Optional[str] = None,
    ) -> TicketRecord:
        endpoint_tag = "POST:/itsm/v1/tickets"
        with self._connect() as conn:
            if idempotency_key and idempotency_key.strip():
                cached = conn.execute(
                    "SELECT response_json FROM idempotency_keys WHERE idem_key = ? AND endpoint = ?",
                    (idempotency_key.strip(), endpoint_tag),
                ).fetchone()
                if cached is not None:
                    return TicketRecord(**json.loads(cached["response_json"]))

            tck_id = f"TCK-{uuid.uuid4().hex[:6].upper()}"
            now = _now_iso()
            record = TicketRecord(
                ticket_id=tck_id,
                title=req.title,
                description=req.description,
                priority=req.priority,
                status="OPEN",
                related_sku=req.related_sku.strip().upper() if req.related_sku else None,
                related_lead_id=req.related_lead_id.strip().upper() if req.related_lead_id else None,
                created_by=req.created_by,
                created_at=now,
            )

            conn.execute(
                """
                INSERT INTO tickets (
                    ticket_id, title, description, priority, status,
                    related_sku, related_lead_id, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.ticket_id,
                    record.title,
                    record.description,
                    record.priority,
                    record.status,
                    record.related_sku,
                    record.related_lead_id,
                    record.created_by,
                    record.created_at,
                ),
            )

            if idempotency_key and idempotency_key.strip():
                conn.execute(
                    "INSERT INTO idempotency_keys (idem_key, endpoint, response_json, created_at) VALUES (?, ?, ?, ?)",
                    (idempotency_key.strip(), endpoint_tag, record.model_dump_json(), now),
                )

            return record

    def get_ticket(self, ticket_id: str) -> TicketRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id.strip().upper(),)).fetchone()
            if row is None:
                raise NotFoundError(f"Ticket '{ticket_id}' not found.")
            return TicketRecord(**dict(row))

    def list_tickets(self) -> List[TicketRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tickets ORDER BY created_at DESC").fetchall()
            return [TicketRecord(**dict(r)) for r in rows]
