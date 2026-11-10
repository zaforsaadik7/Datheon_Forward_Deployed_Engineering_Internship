"""Resilient HTTP client for ERP/CRM/ITSM endpoints with exponential backoff and Idempotency-Key support."""
from dataclasses import dataclass
import threading
import time
from typing import Any, Dict, Optional
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


@dataclass(frozen=True)
class HTTPResult:
    """Result of an HTTP tool request including retry count and elapsed milliseconds."""
    status_code: int
    data: Any
    retries_used: int
    latency_ms: float


class ResilientERPClient:
    """
    HTTP client wrapper supporting either an in-process FastAPI app (via TestClient)
    or a remote base_url (via httpx.Client).
    Retries transient 502/503/504/429 and transport errors with exponential backoff,
    and attaches Idempotency-Key on write operations. Never retries 400/404/409/422.
    """

    def __init__(
        self,
        *,
        app: Optional[FastAPI] = None,
        base_url: str = "http://127.0.0.1:8001",
        max_retries: int = 3,
        base_backoff_s: float = 0.02,
        timeout_s: float = 10.0,
    ) -> None:
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self._app = app
        self._local = threading.local()
        if app is not None:
            self._client: Any = None
        else:
            self._client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def _get_active_client(self) -> Any:
        if self._app is None:
            return self._client
        client = getattr(self._local, "client", None)
        if client is None:
            tc = TestClient(self._app, raise_server_exceptions=False)
            tc.__enter__()
            self._local.client = tc
            return tc
        return client

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> HTTPResult:
        headers: Dict[str, str] = {}
        if idempotency_key and idempotency_key.strip():
            headers["Idempotency-Key"] = idempotency_key.strip()

        t0 = time.perf_counter()
        attempt = 0

        while True:
            try:
                active_client = self._get_active_client()
                resp = active_client.request(
                    method=method.upper(),
                    url=path,
                    params=params,
                    json=json_body,
                    headers=headers,
                )
                status = int(resp.status_code)
                try:
                    payload = resp.json()
                except Exception:
                    payload = {"raw": resp.text}

                if status in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    sleep_s = self.base_backoff_s * (2 ** attempt)
                    time.sleep(sleep_s)
                    attempt += 1
                    continue

                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                return HTTPResult(
                    status_code=status,
                    data=payload,
                    retries_used=attempt,
                    latency_ms=elapsed_ms,
                )
            except httpx.TransportError as exc:
                if attempt < self.max_retries:
                    sleep_s = self.base_backoff_s * (2 ** attempt)
                    time.sleep(sleep_s)
                    attempt += 1
                    continue
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                return HTTPResult(
                    status_code=503,
                    data={"detail": f"Transport error after {attempt} retries: {exc}"},
                    retries_used=attempt,
                    latency_ms=elapsed_ms,
                )

