"""Tool registry, Pydantic pre-flight validation, sensitivity classification, and HTTP execution."""
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Optional, Tuple, Type
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..erp_crm.models import LeadStatus, TicketPriority
from .http_client import ResilientERPClient
from .models import ToolSensitivity, ToolSpec


class GetInventoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku: str = Field(..., min_length=3, description="Warehouse SKU identifier, e.g. SKU-GPU-H100")

    @field_validator("sku")
    @classmethod
    def check_sku(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned.startswith("SKU-"):
            raise ValueError("Invalid SKU format: must start with 'SKU-' (e.g. 'SKU-GPU-H100').")
        return cleaned


class SearchInventoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(default="", description="Search filter for SKU, item name, or warehouse")


class ReserveInventoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku: str = Field(..., min_length=3)
    quantity: int = Field(..., gt=0, le=10000)
    lead_id: str = Field(..., min_length=3)

    @field_validator("sku")
    @classmethod
    def check_sku(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned.startswith("SKU-"):
            raise ValueError("Invalid SKU format: must start with 'SKU-'.")
        return cleaned

    @field_validator("lead_id")
    @classmethod
    def check_lead(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned.startswith("LEAD-"):
            raise ValueError("Invalid lead_id format: must start with 'LEAD-'.")
        return cleaned


class GetLeadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_id: str = Field(..., min_length=3)

    @field_validator("lead_id")
    @classmethod
    def check_lead(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned.startswith("LEAD-"):
            raise ValueError("Invalid lead_id format: must start with 'LEAD-' (e.g. 'LEAD-101').")
        return cleaned


class UpdateLeadStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_id: str = Field(..., min_length=3)
    status: LeadStatus = Field(
        ...,
        description="Target CRM status. Allowed values: NEW, CONTACTED, QUALIFIED, PROPOSAL_SENT, WON, LOST",
    )
    deal_value_usd: Optional[float] = Field(default=None, ge=0.0)
    notes: str = Field(default="", max_length=2000)

    @field_validator("lead_id")
    @classmethod
    def check_lead(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned.startswith("LEAD-"):
            raise ValueError("Invalid lead_id format: must start with 'LEAD-'.")
        return cleaned


class CreateTicketArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=3, max_length=4000)
    priority: TicketPriority = Field(
        default="MEDIUM",
        description="Allowed priorities: LOW, MEDIUM, HIGH, CRITICAL",
    )
    related_sku: Optional[str] = None
    related_lead_id: Optional[str] = None


TOOL_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "get_inventory": GetInventoryArgs,
    "search_inventory": SearchInventoryArgs,
    "reserve_inventory": ReserveInventoryArgs,
    "get_lead": GetLeadArgs,
    "update_lead_status": UpdateLeadStatusArgs,
    "create_ticket": CreateTicketArgs,
}

TOOL_SPECS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_inventory",
        description="Look up warehouse stock on hand, reserved units, available units, and unit price for a specific SKU.",
        sensitivity="SAFE_READ",
        parameters_schema=GetInventoryArgs.model_json_schema(),
    ),
    ToolSpec(
        name="search_inventory",
        description="Search warehouse inventory across SKUs, product names, or warehouse locations.",
        sensitivity="SAFE_READ",
        parameters_schema=SearchInventoryArgs.model_json_schema(),
    ),
    ToolSpec(
        name="reserve_inventory",
        description="Reserve a positive quantity of an inventory SKU for a CRM lead. Sensitive write requiring supervisor HITL approval.",
        sensitivity="HITL_REQUIRED",
        parameters_schema=ReserveInventoryArgs.model_json_schema(),
    ),
    ToolSpec(
        name="get_lead",
        description="Fetch CRM lead profile, pipeline stage, deal value USD, owner, and notes by lead_id.",
        sensitivity="SAFE_READ",
        parameters_schema=GetLeadArgs.model_json_schema(),
    ),
    ToolSpec(
        name="update_lead_status",
        description="Update a CRM lead's pipeline status (NEW, CONTACTED, QUALIFIED, PROPOSAL_SENT, WON, LOST), deal value, and notes. Sensitive write requiring supervisor HITL approval.",
        sensitivity="HITL_REQUIRED",
        parameters_schema=UpdateLeadStatusArgs.model_json_schema(),
    ),
    ToolSpec(
        name="create_ticket",
        description="Create an ITSM fulfillment or escalation ticket. LOW/MEDIUM execute automatically; HIGH/CRITICAL require supervisor HITL approval.",
        sensitivity="AUTO_WRITE",
        parameters_schema=CreateTicketArgs.model_json_schema(),
    ),
)


def classify_tool_sensitivity(tool_name: str, tool_args: Dict[str, Any]) -> ToolSensitivity:
    """
    Code-enforced sensitivity policy (never delegated to the planner):
    - get_inventory, search_inventory, get_lead -> SAFE_READ
    - create_ticket with priority in {LOW, MEDIUM} -> AUTO_WRITE
    - create_ticket with priority in {HIGH, CRITICAL} -> HITL_REQUIRED
    - reserve_inventory, update_lead_status -> HITL_REQUIRED
    """
    if tool_name in {"get_inventory", "search_inventory", "get_lead"}:
        return "SAFE_READ"
    if tool_name == "create_ticket":
        priority = str(tool_args.get("priority", "MEDIUM")).strip().upper()
        if priority in {"LOW", "MEDIUM"}:
            return "AUTO_WRITE"
        return "HITL_REQUIRED"
    return "HITL_REQUIRED"


def canonical_args_sha256(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 digest over (tool_name, canonical_sorted_json(tool_args))."""
    canonical_payload = json.dumps(
        {"tool_name": tool_name, "tool_args": tool_args},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def build_idempotency_key(
    session_id: str,
    step_index: int,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> str:
    """Deterministic Idempotency-Key for write retries within a session step."""
    arg_hash = canonical_args_sha256(tool_name, tool_args)[:16]
    return f"idem-{session_id}-{step_index}-{tool_name}-{arg_hash}"


def wrap_untrusted_tool_output(tool_name: str, payload: Any) -> str:
    """
    Wrap external ERP/CRM tool output in <tool_output_data> delimiters so planners treat
    returned fields (such as CRM notes) strictly as untrusted data, never instructions.
    """
    serialized = json.dumps(payload, sort_keys=True)
    return f'<tool_output_data tool="{tool_name}">{serialized}</tool_output_data>'


@dataclass(frozen=True)
class ToolExecutionResult:
    """Structured outcome of invoking a validated tool against the ERP/CRM REST service."""
    ok: bool
    status_label: str
    http_status: int
    retries_used: int
    latency_ms: float
    observation_text: str
    raw_data: Any


class ToolExecutor:
    """Validates tool arguments and dispatches REST calls through ResilientERPClient."""

    def __init__(self, client: ResilientERPClient) -> None:
        self.client = client

    def validate_args(
        self,
        tool_name: str,
        raw_args: Dict[str, Any],
    ) -> Optional[str]:
        """
        Validate raw_args against the tool's Pydantic schema.
        Returns None if valid, or a descriptive validation error message for self-healing.
        """
        if tool_name not in TOOL_SCHEMAS:
            return f"Unknown tool '{tool_name}'. Allowed tools: {sorted(TOOL_SCHEMAS.keys())}."

        schema_cls = TOOL_SCHEMAS[tool_name]
        try:
            schema_cls(**raw_args)
            return None
        except ValidationError as exc:
            details = []
            for err in exc.errors():
                loc = ".".join(str(x) for x in err.get("loc", ()))
                msg = err.get("msg", "invalid value")
                details.append(f"{loc}: {msg}")
            hint = ""
            if tool_name == "update_lead_status":
                hint = " Allowed values for 'status': ['NEW', 'CONTACTED', 'QUALIFIED', 'PROPOSAL_SENT', 'WON', 'LOST']."
            elif tool_name == "create_ticket":
                hint = " Allowed values for 'priority': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']."
            return f"Schema validation failed for tool '{tool_name}': {'; '.join(details)}.{hint}"

    def normalize_args(self, tool_name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        schema_cls = TOOL_SCHEMAS[tool_name]
        validated = schema_cls(**raw_args)
        return validated.model_dump(exclude_none=True)

    def execute_validated(
        self,
        *,
        session_id: str,
        step_index: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        operator_id: str = "field_operator",
    ) -> ToolExecutionResult:
        norm_args = self.normalize_args(tool_name, tool_args)
        idem_key = build_idempotency_key(session_id, step_index, tool_name, norm_args)

        if tool_name == "get_inventory":
            res = self.client.request("GET", f"/erp/v1/inventory/{norm_args['sku']}")
        elif tool_name == "search_inventory":
            res = self.client.request("GET", "/erp/v1/inventory", params={"query": norm_args.get("query", "")})
        elif tool_name == "reserve_inventory":
            res = self.client.request(
                "POST",
                f"/erp/v1/inventory/{norm_args['sku']}/reserve",
                json_body={
                    "quantity": norm_args["quantity"],
                    "lead_id": norm_args["lead_id"],
                    "requested_by": operator_id,
                },
                idempotency_key=idem_key,
            )
        elif tool_name == "get_lead":
            res = self.client.request("GET", f"/crm/v1/leads/{norm_args['lead_id']}")
        elif tool_name == "update_lead_status":
            body: Dict[str, Any] = {
                "status": norm_args["status"],
                "notes": norm_args.get("notes", ""),
            }
            if "deal_value_usd" in norm_args:
                body["deal_value_usd"] = norm_args["deal_value_usd"]
            res = self.client.request(
                "PATCH",
                f"/crm/v1/leads/{norm_args['lead_id']}/status",
                json_body=body,
                idempotency_key=idem_key,
            )
        elif tool_name == "create_ticket":
            res = self.client.request(
                "POST",
                "/itsm/v1/tickets",
                json_body={
                    "title": norm_args["title"],
                    "description": norm_args["description"],
                    "priority": norm_args.get("priority", "MEDIUM"),
                    "related_sku": norm_args.get("related_sku"),
                    "related_lead_id": norm_args.get("related_lead_id"),
                    "created_by": operator_id,
                },
                idempotency_key=idem_key,
            )
        else:
            raise ValueError(f"Unsupported tool '{tool_name}'")

        ok = 200 <= res.status_code < 300
        status_label = "OK" if ok else ("VALIDATION_ERROR" if res.status_code == 422 else "API_ERROR")
        obs_text = wrap_untrusted_tool_output(
            tool_name,
            {
                "http_status": res.status_code,
                "ok": ok,
                "result": res.data,
            },
        )
        return ToolExecutionResult(
            ok=ok,
            status_label=status_label,
            http_status=res.status_code,
            retries_used=res.retries_used,
            latency_ms=res.latency_ms,
            observation_text=obs_text,
            raw_data=res.data,
        )
