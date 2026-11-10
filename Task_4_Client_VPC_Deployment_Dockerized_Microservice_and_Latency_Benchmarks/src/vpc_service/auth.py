"""Unified VPCPrincipal authentication model bridging Multi-Tenant RBAC RAG and Agent HITL roles."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.agent.models import OperatorPrincipal
from src.rag.models import UserPrincipal as RAGPrincipal

if TYPE_CHECKING:
    from src.vpc_service.secrets import SecretBundle


class VPCAuthenticationError(PermissionError):
    """Raised when an API key is missing or invalid (HTTP 401)."""


class VPCAuthorizationError(PermissionError):
    """Raised when an authenticated principal lacks required role or tenant permissions (HTTP 403)."""


class VPCPrincipal(BaseModel):
    """Unified enterprise principal carrying both RAG clearance attributes and Agent operator/supervisor role."""
    model_config = ConfigDict(frozen=True)

    principal_id: str
    tenant_id: str
    clearance_tier: int = Field(..., ge=1, le=4)
    departments: Tuple[str, ...]
    agent_role: Literal["operator", "supervisor"] = "operator"

    def to_rag_principal(self) -> RAGPrincipal:
        """Convert to Task 2 Multi-Tenant RAG UserPrincipal."""
        return RAGPrincipal(
            user_id=self.principal_id,
            tenant_id=self.tenant_id,
            clearance_tier=self.clearance_tier,
            departments=frozenset(self.departments),
            role_title=f"{self.principal_id} (Tier {self.clearance_tier})",
        )

    def to_agent_operator(self) -> OperatorPrincipal:
        """Convert to Task 3 Autonomous Field Agent OperatorPrincipal (preserving tenant_id)."""
        return OperatorPrincipal(
            operator_id=self.principal_id,
            role=self.agent_role,
            tenant_id=self.tenant_id,
            display_name=f"{self.principal_id} ({self.tenant_id})",
        )


DEFAULT_DEV_VPC_PRINCIPALS: Dict[str, VPCPrincipal] = {
    "acme_exec_supervisor": VPCPrincipal(
        principal_id="acme_exec_supervisor",
        tenant_id="acme_corp",
        clearance_tier=4,
        departments=("engineering", "finance", "legal", "hr", "executive", "operations"),
        agent_role="supervisor",
    ),
    "acme_field_engineer": VPCPrincipal(
        principal_id="acme_field_engineer",
        tenant_id="acme_corp",
        clearance_tier=2,
        departments=("engineering", "operations"),
        agent_role="operator",
    ),
    "acme_peer_engineer": VPCPrincipal(
        principal_id="acme_peer_engineer",
        tenant_id="acme_corp",
        clearance_tier=2,
        departments=("engineering",),
        agent_role="operator",
    ),
    "acme_finance_analyst": VPCPrincipal(
        principal_id="acme_finance_analyst",
        tenant_id="acme_corp",
        clearance_tier=3,
        departments=("finance",),
        agent_role="operator",
    ),
    "acme_ops_associate": VPCPrincipal(
        principal_id="acme_ops_associate",
        tenant_id="acme_corp",
        clearance_tier=1,
        departments=("operations",),
        agent_role="operator",
    ),
    "globex_operator": VPCPrincipal(
        principal_id="globex_operator",
        tenant_id="globex_inc",
        clearance_tier=4,
        departments=("engineering", "finance", "legal", "hr", "executive", "operations"),
        agent_role="operator",
    ),
    "globex_supervisor": VPCPrincipal(
        principal_id="globex_supervisor",
        tenant_id="globex_inc",
        clearance_tier=4,
        departments=("engineering", "finance", "legal", "hr", "executive", "operations"),
        agent_role="supervisor",
    ),
}

DEFAULT_DEV_VPC_KEYS: Dict[str, str] = {
    "dev-key-acme-supervisor": "acme_exec_supervisor",
    "dev-key-acme-engineer": "acme_field_engineer",
    "dev-key-acme-peer": "acme_peer_engineer",
    "dev-key-acme-finance": "acme_finance_analyst",
    "dev-key-acme-ops": "acme_ops_associate",
    "dev-key-globex-operator": "globex_operator",
    "dev-key-globex-supervisor": "globex_supervisor",
}


def authenticate_vpc_principal(
    raw_credential: Optional[str],
    secret_bundle: "SecretBundle",
) -> VPCPrincipal:
    """Resolve X-API-Key or Bearer token to a VPCPrincipal. Raises VPCAuthenticationError (401) if invalid."""
    if raw_credential is None or not raw_credential.strip():
        raise VPCAuthenticationError("Missing authentication credentials. Provide X-API-Key header.")

    token = raw_credential.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    principal = secret_bundle.api_key_map.get(token)
    if principal is None:
        raise VPCAuthenticationError("Invalid or unrecognized API key.")
    return principal


def require_vpc_supervisor(principal: VPCPrincipal) -> None:
    """Enforce supervisor role for HITL approval decisions. Raises VPCAuthorizationError (403) otherwise."""
    if principal.agent_role != "supervisor":
        raise VPCAuthorizationError(
            f"Principal '{principal.principal_id}' (agent_role='{principal.agent_role}') is not authorized "
            f"to decide HITL approvals. Supervisor role required."
        )
