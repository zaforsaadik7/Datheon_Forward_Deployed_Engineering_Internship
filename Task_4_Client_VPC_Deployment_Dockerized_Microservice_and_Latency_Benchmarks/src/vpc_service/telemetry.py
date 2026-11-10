"""Prometheus CollectorRegistry, 8-metric catalog, dynamic scrape gauges, and token accounting."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict

from src.agent.session_store import SessionStore
from src.etl.vector_store import VectorStore
from src.vpc_service.health import evaluate_readiness
from src.vpc_service.secrets import SecretBundle

AccountingMode = Literal["estimated", "provider_reported"]
ErrorClass = Literal[
    "auth",
    "forbidden",
    "validation",
    "conflict",
    "payload_too_large",
    "rate_limited",
    "not_found",
    "server_error",
]

LATENCY_BUCKETS = (0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class TokenUsage(BaseModel):
    """Structured LLM/extractive token accounting attached to RAG and Agent responses."""
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    accounting_mode: AccountingMode


def estimate_tokens(text: str) -> int:
    """Deterministic offline token estimator: ceil(len(stripped_text) / 4)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    return max(1, math.ceil(len(cleaned) / 4))


def classify_http_error(status_code: int) -> Optional[ErrorClass]:
    """Map HTTP 4xx/5xx status code to a bounded Prometheus error_class label."""
    if status_code < 400:
        return None
    if status_code == 401:
        return "auth"
    if status_code == 403:
        return "forbidden"
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        return "conflict"
    if status_code == 413:
        return "payload_too_large"
    if status_code == 422 or status_code == 400:
        return "validation"
    if status_code == 429:
        return "rate_limited"
    return "server_error"


class VPCTelemetry:
    """
    Isolated per-app Prometheus CollectorRegistry exposing the complete 8-metric catalog:
    1. fde_http_requests_total{method, route, status_code}
    2. fde_http_errors_total{method, route, status_code, error_class}
    3. fde_http_request_duration_seconds{method, route}
    4. fde_llm_tokens_total{operation, token_type, accounting_mode}
    5. fde_rag_queries_total{outcome, reranker}
    6. fde_agent_tool_calls_total{tool_name, sensitivity, status}
    7. fde_hitl_pending_approvals
    8. fde_dependency_up{component}
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)

        self.http_requests_total = Counter(
            "fde_http_requests_total",
            "Total HTTP requests processed by matched route template.",
            labelnames=("method", "route", "status_code"),
            registry=self.registry,
        )
        self.http_errors_total = Counter(
            "fde_http_errors_total",
            "Total HTTP 4xx and 5xx error responses by bounded error_class.",
            labelnames=("method", "route", "status_code", "error_class"),
            registry=self.registry,
        )
        self.http_request_duration_seconds = Histogram(
            "fde_http_request_duration_seconds",
            "End-to-end HTTP request latency in seconds.",
            labelnames=("method", "route"),
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.llm_tokens_total = Counter(
            "fde_llm_tokens_total",
            "Cumulative token usage across RAG and Agent operations.",
            labelnames=("operation", "token_type", "accounting_mode"),
            registry=self.registry,
        )
        self.rag_queries_total = Counter(
            "fde_rag_queries_total",
            "Total Multi-Tenant RAG queries by outcome and reranker backend.",
            labelnames=("outcome", "reranker"),
            registry=self.registry,
        )
        self.agent_tool_calls_total = Counter(
            "fde_agent_tool_calls_total",
            "Total Autonomous Field Agent tool calls by name, sensitivity, and status.",
            labelnames=("tool_name", "sensitivity", "status"),
            registry=self.registry,
        )
        self.hitl_pending_approvals = Gauge(
            "fde_hitl_pending_approvals",
            "Current count of PENDING supervisor approvals in SQLite.",
            registry=self.registry,
        )
        self.dependency_up = Gauge(
            "fde_dependency_up",
            "Live dependency health (1 = up, 0 = down) refreshed on scrape.",
            labelnames=("component",),
            registry=self.registry,
        )

        # Pre-initialize label series so all 8 metric families appear immediately on /metrics
        self.http_requests_total.labels(method="GET", route="/healthz", status_code="200").inc(0)
        self.http_errors_total.labels(
            method="POST", route="/api/v1/rag/query", status_code="401", error_class="auth"
        ).inc(0)
        self.llm_tokens_total.labels(
            operation="rag_query", token_type="total", accounting_mode="estimated"
        ).inc(0)
        self.rag_queries_total.labels(outcome="answered", reranker="lexical").inc(0)
        self.agent_tool_calls_total.labels(
            tool_name="get_inventory", sensitivity="SAFE_READ", status="OK"
        ).inc(0)
        self.hitl_pending_approvals.set(0)
        for comp in ("sqlite_db", "chroma_vector_store", "secrets_manager"):
            self.dependency_up.labels(component=comp).set(0)

    def record_http(self, method: str, route: str, status_code: int, duration_seconds: float) -> Optional[str]:
        m = method.upper()
        sc = str(status_code)
        self.http_requests_total.labels(method=m, route=route, status_code=sc).inc()
        self.http_request_duration_seconds.labels(method=m, route=route).observe(max(0.0, duration_seconds))
        err_cls = classify_http_error(status_code)
        if err_cls is not None:
            self.http_errors_total.labels(
                method=m,
                route=route,
                status_code=sc,
                error_class=err_cls,
            ).inc()
        return err_cls

    def record_tokens(
        self,
        operation: Literal["rag_query", "agent_run"],
        usage: TokenUsage,
    ) -> None:
        mode = usage.accounting_mode
        self.llm_tokens_total.labels(
            operation=operation, token_type="prompt", accounting_mode=mode
        ).inc(usage.prompt_tokens)
        self.llm_tokens_total.labels(
            operation=operation, token_type="completion", accounting_mode=mode
        ).inc(usage.completion_tokens)
        self.llm_tokens_total.labels(
            operation=operation, token_type="total", accounting_mode=mode
        ).inc(usage.total_tokens)

    def refresh_dynamic_gauges(
        self,
        *,
        var_dir: Path,
        secret_bundle: Optional[SecretBundle],
        vector_store: Optional[VectorStore],
        session_store: Optional[SessionStore],
    ) -> None:
        _, readiness = evaluate_readiness(
            var_dir=var_dir,
            secret_bundle=secret_bundle,
            vector_store=vector_store,
        )
        checks = readiness.get("checks", {})
        for comp in ("sqlite_db", "chroma_vector_store", "secrets_manager"):
            val = 1.0 if checks.get(comp) == "ok" else 0.0
            self.dependency_up.labels(component=comp).set(val)

        if session_store is not None:
            try:
                pending_count = len(session_store.list_approvals(status="PENDING"))
                self.hitl_pending_approvals.set(float(pending_count))
            except Exception:
                pass

    def render_prometheus_text(
        self,
        *,
        var_dir: Path,
        secret_bundle: Optional[SecretBundle],
        vector_store: Optional[VectorStore],
        session_store: Optional[SessionStore],
    ) -> bytes:
        self.refresh_dynamic_gauges(
            var_dir=var_dir,
            secret_bundle=secret_bundle,
            vector_store=vector_store,
            session_store=session_store,
        )
        return generate_latest(self.registry)
