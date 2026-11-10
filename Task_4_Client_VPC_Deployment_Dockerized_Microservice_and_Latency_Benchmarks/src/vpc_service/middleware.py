"""Pure ASGI middlewares for streaming body-size enforcement (413), bounded LRU rate limiting (429), and telemetry."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import time
from typing import Any, Callable, Dict, Optional, Tuple
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.vpc_service.logging_utils import emit_structured_log
from src.vpc_service.secrets import SecretBundle
from src.vpc_service.telemetry import VPCTelemetry


class _BodyTooLargeError(BaseException):
    """Internal signal raised when streamed request chunks exceed max_body_bytes."""


def _extract_header(scope: Scope, name_lower: bytes) -> Optional[str]:
    for k, v in scope.get("headers") or ():
        if k.lower() == name_lower:
            return v.decode("latin1")
    return None


def _match_route_template(scope: Scope, known_templates: Tuple[str, ...]) -> str:
    route_obj = scope.get("route")
    if route_obj is not None:
        route_path = getattr(route_obj, "path", None)
        if isinstance(route_path, str) and route_path:
            return route_path

    raw_path = scope.get("path") or "/"
    if raw_path in known_templates:
        return raw_path
    if raw_path.startswith("/api/v1/agent/sessions/"):
        return "/api/v1/agent/sessions/{session_id}"
    if raw_path.startswith("/api/v1/agent/approvals/") and raw_path.endswith("/decide"):
        return "/api/v1/agent/approvals/{approval_id}/decide"
    return "UNMATCHED"


async def _send_json_response(
    send: Send,
    status_code: int,
    payload: Dict[str, Any],
    extra_headers: Optional[Dict[str, str]] = None,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if extra_headers:
        for k, v in extra_headers.items():
            headers.append((k.encode("latin1"), v.encode("latin1")))
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": headers,
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": body,
        }
    )


class BoundedTokenBucketLimiter:
    """
    Bounded LRU token-bucket rate limiter (max_buckets=1024).
    Keys on `key:<sha256[:16]>` when the caller presents a recognized API key in SecretBundle,
    or falls back to `ip:<client_host>` for unauthenticated/unrecognized keys so an attacker
    rotating random fake X-API-Key strings cannot bypass rate limits or exhaust memory.
    """

    def __init__(
        self,
        *,
        rate_rps: float,
        burst: int,
        max_buckets: int = 1024,
    ) -> None:
        self.rate_rps = float(rate_rps)
        self.burst = float(burst)
        self.max_buckets = int(max_buckets)
        self._buckets: "OrderedDict[str, Tuple[float, float]]" = OrderedDict()

    def resolve_bucket_key(self, scope: Scope, secret_bundle: SecretBundle) -> str:
        raw_key = _extract_header(scope, b"x-api-key") or _extract_header(scope, b"authorization") or ""
        token = raw_key.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()

        if token and token in secret_bundle.api_key_map:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
            return f"key:{digest}"

        client = scope.get("client")
        client_host = client[0] if (isinstance(client, (tuple, list)) and client) else "127.0.0.1"
        return f"ip:{client_host}"

    def allow(self, bucket_key: str) -> bool:
        if self.rate_rps <= 0 or self.burst <= 0:
            return True

        now = time.perf_counter()
        if bucket_key in self._buckets:
            tokens, last_ts = self._buckets.pop(bucket_key)
            elapsed = max(0.0, now - last_ts)
            tokens = min(self.burst, tokens + elapsed * self.rate_rps)
        else:
            tokens = self.burst

        if tokens < 1.0:
            self._buckets[bucket_key] = (tokens, now)
            if len(self._buckets) > self.max_buckets:
                self._buckets.popitem(last=False)
            return False

        self._buckets[bucket_key] = (tokens - 1.0, now)
        if len(self._buckets) > self.max_buckets:
            self._buckets.popitem(last=False)
        return True


class VPCGatewayMiddleware:
    """
    Unified outer ASGI middleware enforcing:
    1. X-Request-ID correlation ID extraction / generation
    2. Bounded LRU token-bucket rate limiting (429 Too Many Requests, bypassing /healthz and /metrics)
    3. Streaming + Content-Length request body size limit (413 Payload Too Large)
    4. Post-routing Prometheus HTTP metrics (route template normalization, latency histogram, error class)
    5. Structured JSON access logging with secret redaction
    """

    KNOWN_TEMPLATES: Tuple[str, ...] = (
        "/healthz",
        "/readyz",
        "/metrics",
        "/api/v1/rag/query",
        "/api/v1/agent/run",
        "/api/v1/agent/approvals",
    )

    def __init__(
        self,
        app: ASGIApp,
        *,
        telemetry: VPCTelemetry,
        secret_bundle: SecretBundle,
        max_body_bytes: int = 1_048_576,
        rate_limiter: Optional[BoundedTokenBucketLimiter] = None,
        log_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.app = app
        self.telemetry = telemetry
        self.secret_bundle = secret_bundle
        self.max_body_bytes = max_body_bytes
        self.rate_limiter = rate_limiter
        self.log_sink = log_sink

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        t0 = time.perf_counter()
        method = (scope.get("method") or "GET").upper()
        raw_path = scope.get("path") or "/"

        incoming_req_id = _extract_header(scope, b"x-request-id")
        request_id = (
            incoming_req_id.strip()
            if (incoming_req_id and incoming_req_id.strip())
            else f"req-{uuid.uuid4().hex[:12]}"
        )
        scope["fde_request_id"] = request_id

        status_code_holder = {"code": 200, "started": False}

        async def wrapped_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_code_holder["code"] = int(message.get("status", 200))
                status_code_holder["started"] = True
                headers = list(message.get("headers") or [])
                if not any(k.lower() == b"x-request-id" for k, _ in headers):
                    headers.append((b"x-request-id", request_id.encode("latin1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            # 1. Rate-limit check (skip /healthz, /readyz, /metrics)
            if self.rate_limiter is not None and raw_path not in {"/healthz", "/readyz", "/metrics"}:
                bucket_key = self.rate_limiter.resolve_bucket_key(scope, self.secret_bundle)
                if not self.rate_limiter.allow(bucket_key):
                    await _send_json_response(
                        wrapped_send,
                        429,
                        {"detail": "Rate limit exceeded. Retry after 1 second."},
                        extra_headers={"Retry-After": "1"},
                    )
                    return

            # 2. Fast-path Content-Length header check
            cl_header = _extract_header(scope, b"content-length")
            if cl_header is not None:
                try:
                    if int(cl_header) > self.max_body_bytes:
                        await _send_json_response(
                            wrapped_send,
                            413,
                            {"detail": f"Request payload exceeds maximum limit of {self.max_body_bytes} bytes."},
                        )
                        return
                except ValueError:
                    pass

            # 3. Streaming chunk byte counter on `receive`
            bytes_seen = {"total": 0}

            async def wrapped_receive() -> Message:
                msg = await receive()
                if msg["type"] == "http.request":
                    chunk = msg.get("body", b"") or b""
                    bytes_seen["total"] += len(chunk)
                    if bytes_seen["total"] > self.max_body_bytes:
                        raise _BodyTooLargeError()
                return msg

            try:
                await self.app(scope, wrapped_receive, wrapped_send)
            except _BodyTooLargeError:
                if not status_code_holder["started"]:
                    await _send_json_response(
                        wrapped_send,
                        413,
                        {"detail": f"Streamed request payload exceeds maximum limit of {self.max_body_bytes} bytes."},
                    )
        finally:
            elapsed_s = max(0.0, time.perf_counter() - t0)
            route_label = _match_route_template(scope, self.KNOWN_TEMPLATES)
            final_code = status_code_holder["code"]
            err_cls = self.telemetry.record_http(method, route_label, final_code, elapsed_s)
            emit_structured_log(
                request_id=request_id,
                method=method,
                route=route_label,
                status_code=final_code,
                latency_ms=elapsed_s * 1000.0,
                error_class=err_cls,
                principal_id=scope.get("fde_principal_id"),
                tenant_id=scope.get("fde_tenant_id"),
                log_sink=self.log_sink,
            )
