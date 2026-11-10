"""RBAC principals, API key authentication, defense-in-depth post-retrieval verification, and audit logging."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from ..config import DEPARTMENTS, DEFAULT_TENANT
from ..etl.vector_store import QueryHit
from .models import AuditEvent, Citation, UserPrincipal


class AuthenticationError(PermissionError):
    """Raised when an API key or credential is missing or unrecognized (HTTP 401)."""
    pass


class AccessDeniedError(PermissionError):
    """Raised when an authenticated principal has no authorized scope (HTTP 403)."""
    pass


class RBACViolationError(PermissionError):
    """Raised if a post-retrieval chunk violates the principal's tenant, tier, or department scope."""
    pass


PERSONAS: Dict[str, UserPrincipal] = {
    "ops_associate": UserPrincipal(
        user_id="ops_associate",
        clearance_tier=1,
        departments=frozenset({"operations"}),
        tenant_id=DEFAULT_TENANT,
        role_title="Operations Associate (Tier 1)",
    ),
    "cloud_engineer": UserPrincipal(
        user_id="cloud_engineer",
        clearance_tier=2,
        departments=frozenset({"engineering"}),
        tenant_id=DEFAULT_TENANT,
        role_title="Cloud Infrastructure Engineer (Tier 2)",
    ),
    "hr_specialist": UserPrincipal(
        user_id="hr_specialist",
        clearance_tier=2,
        departments=frozenset({"hr"}),
        tenant_id=DEFAULT_TENANT,
        role_title="HR Compensation Specialist (Tier 2)",
    ),
    "finance_director": UserPrincipal(
        user_id="finance_director",
        clearance_tier=3,
        departments=frozenset({"finance", "executive"}),
        tenant_id=DEFAULT_TENANT,
        role_title="VP of Finance & Strategy (Tier 3)",
    ),
    "general_counsel": UserPrincipal(
        user_id="general_counsel",
        clearance_tier=4,
        departments=DEPARTMENTS,
        tenant_id=DEFAULT_TENANT,
        role_title="Chief Legal Officer & General Counsel (Tier 4)",
    ),
    "external_tenant": UserPrincipal(
        user_id="external_tenant",
        clearance_tier=4,
        departments=DEPARTMENTS,
        tenant_id="globex_inc",
        role_title="Globex Inc. External General Counsel (Tier 4)",
    ),
}

API_KEY_REGISTRY: Dict[str, UserPrincipal] = {
    "key-ops-tier1": PERSONAS["ops_associate"],
    "key-eng-tier2": PERSONAS["cloud_engineer"],
    "key-hr-tier2": PERSONAS["hr_specialist"],
    "key-fin-tier3": PERSONAS["finance_director"],
    "key-leg-tier4": PERSONAS["general_counsel"],
    "key-globex-tier4": PERSONAS["external_tenant"],
}


def authenticate_api_key(
    api_key: Optional[str],
    dev_persona: Optional[str] = None,
    allow_dev_header: bool = False,
) -> UserPrincipal:
    """
    Resolve an API key (or dev persona header when explicitly enabled) to a server-side UserPrincipal.
    Raises AuthenticationError (HTTP 401) if credentials are missing or invalid.
    """
    if api_key is not None and api_key.strip():
        token = api_key.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if token in API_KEY_REGISTRY:
            return API_KEY_REGISTRY[token]
        raise AuthenticationError("Invalid API key.")

    if allow_dev_header and dev_persona is not None and dev_persona.strip():
        persona_key = dev_persona.strip()
        if persona_key in PERSONAS:
            return PERSONAS[persona_key]
        raise AuthenticationError(f"Unknown dev persona '{persona_key}'.")

    raise AuthenticationError("Missing authentication credentials. Provide X-API-Key header.")


def authorize_principal(principal: UserPrincipal) -> None:
    """
    Verify that an authenticated principal has non-empty department scope and valid tenant_id.
    Raises AccessDeniedError (HTTP 403) if the principal is not permitted to query.
    """
    if not principal.tenant_id or not principal.tenant_id.strip():
        raise AccessDeniedError(f"Principal '{principal.user_id}' has no assigned tenant.")
    if len(principal.departments) == 0:
        raise AccessDeniedError(f"Principal '{principal.user_id}' has no authorized departments.")


def is_chunk_authorized(metadata: Dict[str, Any], principal: UserPrincipal) -> bool:
    """
    Check whether a single chunk's metadata satisfies the principal's RBAC & tenant policy:
    1. chunk_tenant == principal.tenant_id
    2. Tier-1 Public Rule: within the same tenant, clearance_tier == 1 is readable by any
       authenticated user with non-empty departments; Tiers 2-4 require both
       clearance_tier <= principal.clearance_tier and department in principal.departments.
    """
    if len(principal.departments) == 0:
        return False

    chunk_tenant = str(metadata.get("tenant_id", DEFAULT_TENANT)).strip()
    if chunk_tenant != principal.tenant_id:
        return False

    raw_tier = metadata.get("clearance_tier")
    if not isinstance(raw_tier, int) or isinstance(raw_tier, bool):
        return False

    if raw_tier == 1:
        return principal.clearance_tier >= 1

    chunk_dept = str(metadata.get("department", "")).strip().lower()
    return raw_tier <= principal.clearance_tier and chunk_dept in principal.departments


def assert_hits_authorized(hits: Sequence[QueryHit], principal: UserPrincipal) -> List[QueryHit]:
    """
    Defense-in-depth post-retrieval assertion.
    Verifies every retrieved QueryHit against is_chunk_authorized(hit.metadata, principal).
    Raises RBACViolationError immediately if any hit is unauthorized.
    """
    verified: List[QueryHit] = []
    for hit in hits:
        if not is_chunk_authorized(hit.metadata, principal):
            raise RBACViolationError(
                f"Post-retrieval RBAC violation on chunk '{hit.chunk_id}' for principal '{principal.user_id}'."
            )
        verified.append(hit)
    return verified


def build_citation_from_metadata(
    citation_index: int,
    chunk_id: str,
    metadata: Dict[str, Any],
    rerank_score: float,
) -> Citation:
    """
    Build a Citation with the deterministic 4-level locator hierarchy:
    1. page (PDF 1-based physical page_number) -> [source_file, p. N]
    2. row (CSV/JSON record with row_id) -> [source_file, row N]
    3. section (Markdown with section_title) -> [source_file, §section_title]
    4. chunk (fallback when page_number, row_id, and section_title are None) -> [source_file, chunk N]
       where N = int(chunk_index) + 1
    """
    page_number = metadata.get("page_number")
    row_id = metadata.get("row_id")
    section_title = metadata.get("section_title")
    chunk_index = int(metadata.get("chunk_index", 0))

    if page_number is not None:
        locator_type = "page"
        locator_value = str(int(page_number))
    elif row_id is not None:
        locator_type = "row"
        locator_value = str(int(row_id))
    elif section_title is not None and str(section_title).strip():
        locator_type = "section"
        locator_value = str(section_title).strip()
    else:
        locator_type = "chunk"
        locator_value = str(chunk_index + 1)

    return Citation(
        citation_index=citation_index,
        chunk_id=chunk_id,
        source_file=str(metadata.get("source_file", "unknown")),
        department=str(metadata.get("department", "operations")).lower(),
        clearance_tier=int(metadata.get("clearance_tier", 4)),
        document_date=str(metadata.get("document_date", "1970-01-01")),
        tenant_id=str(metadata.get("tenant_id", DEFAULT_TENANT)),
        page_number=int(page_number) if page_number is not None else None,
        row_id=int(row_id) if row_id is not None else None,
        section_title=str(section_title) if section_title is not None else None,
        chunk_index=chunk_index,
        locator_type=locator_type,
        locator_value=locator_value,
        rerank_score=float(rerank_score),
    )


class AuditLogger:
    """In-memory structured audit logger that never leaks restricted document existence."""

    def __init__(self) -> None:
        self._events: List[AuditEvent] = []

    @property
    def events(self) -> Tuple[AuditEvent, ...]:
        return tuple(self._events)

    def record(
        self,
        *,
        principal: UserPrincipal,
        query: str,
        rewritten_query: str,
        authorized_candidates: int,
        cited_sources: Sequence[str],
        abstained: bool,
        injection_detected: bool,
        rbac_violation_blocked: bool,
        total_ms: float,
    ) -> AuditEvent:
        event = AuditEvent(
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
            clearance_tier=principal.clearance_tier,
            departments=tuple(sorted(principal.departments)),
            query=query,
            rewritten_query=rewritten_query,
            authorized_candidates=authorized_candidates,
            cited_sources=tuple(cited_sources),
            abstained=abstained,
            injection_detected=injection_detected,
            rbac_violation_blocked=rbac_violation_blocked,
            total_ms=round(float(total_ms), 3),
        )
        self._events.append(event)
        return event
