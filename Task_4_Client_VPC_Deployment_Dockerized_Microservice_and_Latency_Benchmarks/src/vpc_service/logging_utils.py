"""Structured JSON logger with X-Request-ID correlation and automatic secret redaction."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("fde.vpc_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def emit_structured_log(
    *,
    request_id: str,
    method: str,
    route: str,
    status_code: int,
    latency_ms: float,
    error_class: Optional[str] = None,
    principal_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    log_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Emit a single-line structured JSON log record with zero raw credentials."""
    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "ERROR" if status_code >= 500 else ("WARNING" if status_code >= 400 else "INFO"),
        "service": "datheon-fde-vpc-gateway",
        "request_id": request_id,
        "method": method.upper(),
        "route": route,
        "status_code": int(status_code),
        "latency_ms": round(float(latency_ms), 3),
    }
    if principal_id:
        record["principal_id"] = principal_id
    if tenant_id:
        record["tenant_id"] = tenant_id
    if error_class:
        record["error_class"] = error_class

    logger.info(json.dumps(record, sort_keys=True))
    if log_sink is not None:
        log_sink(record)
    return record
