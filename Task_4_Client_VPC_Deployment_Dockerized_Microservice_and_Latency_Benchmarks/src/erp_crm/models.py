"""Pydantic schemas and CRM lead state-transition rules for the Mock Enterprise ERP/CRM/ITSM API."""
from typing import Dict, FrozenSet, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

LeadStatus = Literal["NEW", "CONTACTED", "QUALIFIED", "PROPOSAL_SENT", "WON", "LOST"]
TicketPriority = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
TicketStatus = Literal["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"]

ALLOWED_LEAD_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    "NEW": frozenset({"CONTACTED", "QUALIFIED", "LOST"}),
    "CONTACTED": frozenset({"QUALIFIED", "PROPOSAL_SENT", "LOST"}),
    "QUALIFIED": frozenset({"PROPOSAL_SENT", "WON", "LOST"}),
    "PROPOSAL_SENT": frozenset({"WON", "LOST"}),
    "WON": frozenset(),
    "LOST": frozenset(),
}


class InventoryItem(BaseModel):
    """Warehouse SKU inventory record."""
    model_config = ConfigDict(frozen=True)

    sku: str
    name: str
    stock_on_hand: int
    reserved: int
    available: int
    unit_price_usd: float
    warehouse: str
    reorder_threshold: int


class ReserveInventoryRequest(BaseModel):
    """Payload for POST /erp/v1/inventory/{sku}/reserve."""
    model_config = ConfigDict(extra="forbid")

    quantity: int = Field(..., gt=0, le=10000)
    lead_id: str = Field(..., min_length=1)
    requested_by: str = Field(default="field_operator", min_length=1)

    @field_validator("lead_id", "requested_by")
    @classmethod
    def validate_non_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field must not be blank.")
        return cleaned


class ReservationReceipt(BaseModel):
    """Transactional receipt returned when stock is reserved."""
    model_config = ConfigDict(frozen=True)

    reservation_id: str
    sku: str
    lead_id: str
    quantity_reserved: int
    remaining_available: int
    total_reserved_for_sku: int
    unit_price_usd: float
    total_hold_value_usd: float
    warehouse: str
    created_at: str


class LeadRecord(BaseModel):
    """CRM commercial lead record."""
    model_config = ConfigDict(frozen=True)

    lead_id: str
    company: str
    contact_email: str
    status: LeadStatus
    deal_value_usd: float
    owner: str
    notes: str
    updated_at: str


class UpdateLeadStatusRequest(BaseModel):
    """Payload for PATCH /crm/v1/leads/{lead_id}/status."""
    model_config = ConfigDict(extra="forbid")

    status: LeadStatus
    deal_value_usd: Optional[float] = Field(default=None, ge=0.0)
    notes: str = Field(default="", max_length=2000)


class LeadTransitionReceipt(BaseModel):
    """Transactional receipt returned when a CRM lead status is updated."""
    model_config = ConfigDict(frozen=True)

    transition_id: str
    lead_id: str
    company: str
    previous_status: LeadStatus
    status: LeadStatus
    deal_value_usd: float
    notes: str
    updated_at: str


class TicketRecord(BaseModel):
    """ITSM support/escalation ticket record."""
    model_config = ConfigDict(frozen=True)

    ticket_id: str
    title: str
    description: str
    priority: TicketPriority
    status: TicketStatus
    related_sku: Optional[str] = None
    related_lead_id: Optional[str] = None
    created_by: str
    created_at: str


class CreateTicketRequest(BaseModel):
    """Payload for POST /itsm/v1/tickets."""
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=3, max_length=4000)
    priority: TicketPriority = "MEDIUM"
    related_sku: Optional[str] = None
    related_lead_id: Optional[str] = None
    created_by: str = Field(default="field_operator", min_length=1)

    @field_validator("title", "description", "created_by")
    @classmethod
    def validate_non_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field must not be blank.")
        return cleaned


class FaultInjectionRule(BaseModel):
    """Rule for simulating transient 503/429 failures on a specific ERP/CRM route."""
    model_config = ConfigDict(extra="forbid")

    method: str = Field(..., min_length=3)
    path: str = Field(..., min_length=1)
    remaining_failures: int = Field(..., ge=1, le=20)
    status_code: int = Field(default=503)

