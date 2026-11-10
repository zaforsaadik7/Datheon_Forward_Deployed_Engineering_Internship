"""Operator and Supervisor API key authentication and role-separation enforcement."""
import json
import os
from typing import Dict, Optional

from .models import OperatorPrincipal


class AuthenticationError(PermissionError):
    """Raised when an API key is missing or unrecognized (HTTP 401)."""
    pass


class AccessDeniedError(PermissionError):
    """Raised when an authenticated operator lacks required role permissions (HTTP 403)."""
    pass


OPERATORS: Dict[str, OperatorPrincipal] = {
    "field_operator": OperatorPrincipal(
        operator_id="field_operator",
        role="operator",
        display_name="Field Deployment Operator",
    ),
    "peer_operator": OperatorPrincipal(
        operator_id="peer_operator",
        role="operator",
        display_name="Secondary Field Operator",
    ),
    "ops_supervisor": OperatorPrincipal(
        operator_id="ops_supervisor",
        role="supervisor",
        display_name="Enterprise Operations Supervisor",
    ),
}

DEFAULT_DEV_OPERATOR_KEYS: Dict[str, str] = {
    "field_operator": "key-field-operator",
    "peer_operator": "key-peer-operator",
    "ops_supervisor": "key-ops-supervisor",
}
_DEFAULT_DEV_KEYS = DEFAULT_DEV_OPERATOR_KEYS


def load_operator_key_registry(allow_dev_keys: Optional[bool] = None) -> Dict[str, OperatorPrincipal]:
    """
    Load API keys from environment variables:
    - AGENT_API_KEYS_JSON: JSON mapping of {"<api_key>": "<operator_id>"}
    - AGENT_API_KEY_FIELD_OPERATOR / AGENT_API_KEY_PEER_OPERATOR / AGENT_API_KEY_OPS_SUPERVISOR
    When AGENT_ENV=production (or allow_dev_keys=False), dev fallback keys are disabled and only
    explicitly configured environment keys are accepted.
    """
    if allow_dev_keys is None:
        allow_dev_keys = os.environ.get("AGENT_ENV", "development").strip().lower() != "production"

    registry: Dict[str, OperatorPrincipal] = {}
    raw_json = os.environ.get("AGENT_API_KEYS_JSON", "").strip()
    if raw_json:
        parsed = json.loads(raw_json)
        for token, op_id in parsed.items():
            if op_id in OPERATORS and str(token).strip():
                registry[str(token).strip()] = OPERATORS[op_id]
        return registry

    for op_id, principal in OPERATORS.items():
        env_var = f"AGENT_API_KEY_{op_id.upper()}"
        default_token = DEFAULT_DEV_OPERATOR_KEYS[op_id] if allow_dev_keys else ""
        token = os.environ.get(env_var, default_token).strip()
        if token:
            registry[token] = principal
    return registry


def authenticate_operator(
    api_key: Optional[str],
    dev_operator: Optional[str] = None,
    allow_dev_header: bool = False,
) -> OperatorPrincipal:
    """
    Resolve X-API-Key or Bearer token to an OperatorPrincipal.
    Raises AuthenticationError (HTTP 401) if missing or invalid.
    """
    if api_key is not None and api_key.strip():
        token = api_key.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        registry = load_operator_key_registry()
        if token in registry:
            return registry[token]
        raise AuthenticationError("Invalid API key.")

    if allow_dev_header and dev_operator is not None and dev_operator.strip():
        op_id = dev_operator.strip()
        if op_id in OPERATORS:
            return OPERATORS[op_id]
        raise AuthenticationError(f"Unknown dev operator '{op_id}'.")

    raise AuthenticationError("Missing authentication credentials. Provide X-API-Key header.")


def require_supervisor(principal: OperatorPrincipal) -> None:
    """
    Enforce separation of duties: only principals with role == 'supervisor' may approve or reject
    Human-in-the-Loop gates. Raises AccessDeniedError (HTTP 403) otherwise.
    """
    if principal.role != "supervisor":
        raise AccessDeniedError(
            f"Operator '{principal.operator_id}' (role='{principal.role}') is not authorized to decide HITL approvals. Supervisor role required."
        )

