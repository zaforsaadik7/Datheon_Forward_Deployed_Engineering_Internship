"""Mock Enterprise ERP/CRM/ITSM FastAPI server with fault injection and idempotency headers."""
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ..config import get_settings
from .database import (
    ERPCRMDatabase,
    IllegalStateTransitionError,
    NotFoundError,
    StockConflictError,
)
from .models import (
    CreateTicketRequest,
    FaultInjectionRule,
    InventoryItem,
    LeadRecord,
    LeadTransitionReceipt,
    ReservationReceipt,
    ReserveInventoryRequest,
    TicketRecord,
    UpdateLeadStatusRequest,
)


def create_erp_crm_app(db: Optional[ERPCRMDatabase] = None) -> FastAPI:
    """Create the Mock Enterprise ERP/CRM/ITSM REST FastAPI app."""
    settings = get_settings()
    erp_db = db if db is not None else ERPCRMDatabase(settings.erp_db_path)

    # Fault-injection table keyed by (METHOD, path) -> [remaining_failures, status_code]
    fault_rules: Dict[Tuple[str, str], List[int]] = {}

    app = FastAPI(
        title="Mock Enterprise ERP / CRM / ITSM Transactional API",
        version="1.0.0",
    )

    @app.middleware("http")
    async def fault_injection_middleware(request: Request, call_next):
        key = (request.method.upper(), request.url.path)
        rule = fault_rules.get(key)
        if rule and rule[0] > 0:
            rule[0] -= 1
            status_code = rule[1]
            return JSONResponse(
                status_code=status_code,
                content={
                    "detail": f"Simulated transient upstream error ({status_code}) on {key[0]} {key[1]}.",
                    "retryable": True,
                },
                headers={"Retry-After": "1"},
            )
        return await call_next(request)

    @app.post("/admin/faults")
    def configure_fault(rule: FaultInjectionRule) -> Dict[str, object]:
        key = (rule.method.strip().upper(), rule.path.strip())
        fault_rules[key] = [rule.remaining_failures, rule.status_code]
        return {
            "status": "configured",
            "method": key[0],
            "path": key[1],
            "remaining_failures": rule.remaining_failures,
            "status_code": rule.status_code,
        }

    @app.delete("/admin/faults")
    def clear_faults() -> Dict[str, str]:
        fault_rules.clear()
        return {"status": "cleared"}

    # --- ERP Inventory Endpoints ---
    @app.get("/erp/v1/inventory", response_model=List[InventoryItem])
    def list_or_search_inventory(query: Optional[str] = Query(default=None)) -> List[InventoryItem]:
        return erp_db.search_inventory(query=query)

    @app.get("/erp/v1/inventory/{sku}", response_model=InventoryItem)
    def get_inventory_sku(sku: str) -> InventoryItem:
        try:
            return erp_db.get_inventory(sku)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/erp/v1/inventory/{sku}/reserve", response_model=ReservationReceipt)
    def reserve_inventory_sku(
        sku: str,
        payload: ReserveInventoryRequest,
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    ) -> ReservationReceipt:
        try:
            return erp_db.reserve_inventory(sku, payload, idempotency_key=idempotency_key)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StockConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # --- CRM Lead Endpoints ---
    @app.get("/crm/v1/leads", response_model=List[LeadRecord])
    def list_or_search_leads(query: Optional[str] = Query(default=None)) -> List[LeadRecord]:
        return erp_db.search_leads(query=query)

    @app.get("/crm/v1/leads/{lead_id}", response_model=LeadRecord)
    def get_lead_by_id(lead_id: str) -> LeadRecord:
        try:
            return erp_db.get_lead(lead_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch("/crm/v1/leads/{lead_id}/status", response_model=LeadTransitionReceipt)
    def patch_lead_status(
        lead_id: str,
        payload: UpdateLeadStatusRequest,
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    ) -> LeadTransitionReceipt:
        try:
            return erp_db.update_lead_status(lead_id, payload, idempotency_key=idempotency_key)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IllegalStateTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # --- ITSM Ticket Endpoints ---
    @app.get("/itsm/v1/tickets", response_model=List[TicketRecord])
    def list_all_tickets() -> List[TicketRecord]:
        return erp_db.list_tickets()

    @app.get("/itsm/v1/tickets/{ticket_id}", response_model=TicketRecord)
    def get_ticket_by_id(ticket_id: str) -> TicketRecord:
        try:
            return erp_db.get_ticket(ticket_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/itsm/v1/tickets", response_model=TicketRecord)
    def create_new_ticket(
        payload: CreateTicketRequest,
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    ) -> TicketRecord:
        return erp_db.create_ticket(payload, idempotency_key=idempotency_key)

    return app
