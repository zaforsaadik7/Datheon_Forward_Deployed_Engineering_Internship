from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from fastapi.testclient import TestClient
import httpx
import pytest

from src.vpc_service.app import create_vpc_app
from src.vpc_service.secrets import init_dev_secret_files, load_secret_bundle


def _setup_test_env(tmp_path: Path):
    var_dir = tmp_path / "var"
    secrets_dir = tmp_path / "secrets"
    keys_file, metrics_file = init_dev_secret_files(secrets_dir)
    bundle = load_secret_bundle(
        env="production",
        api_keys_file=keys_file,
        metrics_token_file=metrics_file,
    )
    # Map principal_id -> api_key for convenient test lookups
    keys_by_principal = {p.principal_id: k for k, p in bundle.api_key_map.items()}
    return var_dir, keys_file, metrics_file, bundle, keys_by_principal


def test_8_metric_catalog_scrape_freshness_route_templates_and_metrics_auth(tmp_path: Path) -> None:
    var_dir, keys_file, metrics_file, bundle, keys = _setup_test_env(tmp_path)
    app = create_vpc_app(
        env="production",
        var_dir=var_dir,
        api_keys_file=keys_file,
        metrics_token_file=metrics_file,
        auto_seed_on_startup=True,
        allow_sample_seed_in_prod=True,
    )

    with TestClient(app) as client:
        # 1. /metrics requires FDE_METRICS_TOKEN when configured
        assert client.get("/metrics").status_code == 401
        metrics_headers = {"Authorization": f"Bearer {bundle.metrics_token}"}
        m_resp = client.get("/metrics", headers=metrics_headers)
        assert m_resp.status_code == 200
        text0 = m_resp.text

        # All 8 metric families exist and fde_dependency_up is refreshed on scrape even before /readyz
        for metric_name in (
            "fde_http_requests_total",
            "fde_http_errors_total",
            "fde_http_request_duration_seconds",
            "fde_llm_tokens_total",
            "fde_rag_queries_total",
            "fde_agent_tool_calls_total",
            "fde_hitl_pending_approvals",
            "fde_dependency_up",
        ):
            assert metric_name in text0, f"Missing metric family {metric_name}"

        assert 'fde_dependency_up{component="sqlite_db"} 1.0' in text0
        assert 'fde_dependency_up{component="chroma_vector_store"} 1.0' in text0
        assert 'fde_dependency_up{component="secrets_manager"} 1.0' in text0

        # 2. Route-template normalization & UNMATCHED 404 cardinality protection
        eng_headers = {"X-API-Key": keys["acme_field_engineer"]}
        client.get("/api/v1/agent/sessions/SES-NONEXISTENT-1", headers=eng_headers)
        client.get("/api/v1/agent/sessions/SES-NONEXISTENT-2", headers=eng_headers)
        client.get("/api/v1/random-unmatched-url-12345")
        client.get("/api/v1/random-unmatched-url-67890")

        text1 = client.get("/metrics", headers=metrics_headers).text
        assert 'route="/api/v1/agent/sessions/{session_id}"' in text1
        assert "SES-NONEXISTENT-1" not in text1
        assert 'route="UNMATCHED"' in text1
        assert "random-unmatched-url" not in text1


def test_token_accounting_modes_estimated_vs_provider_reported_and_json_logging(tmp_path: Path) -> None:
    var_dir, keys_file, metrics_file, bundle, keys = _setup_test_env(tmp_path)
    log_records: List[dict] = []

    app = create_vpc_app(
        env="production",
        var_dir=var_dir,
        api_keys_file=keys_file,
        metrics_token_file=metrics_file,
        auto_seed_on_startup=True,
        allow_sample_seed_in_prod=True,
        log_sink=log_records.append,
    )

    eng_key = keys["acme_field_engineer"]
    eng_headers = {"X-API-Key": eng_key, "X-Request-ID": "req-trace-777"}
    metrics_headers = {"Authorization": f"Bearer {bundle.metrics_token}"}

    with TestClient(app) as client:
        # 1. Offline RAG query emits estimated token_usage and echoes X-Request-ID
        rag_resp = client.post(
            "/api/v1/rag/query",
            json={"query": "What is the minimum required password length?"},
            headers=eng_headers,
        )
        assert rag_resp.status_code == 200
        assert rag_resp.headers["X-Request-ID"] == "req-trace-777"
        rag_data = rag_resp.json()
        assert rag_data["abstained"] is False
        assert rag_data["token_usage"]["accounting_mode"] == "estimated"
        assert rag_data["token_usage"]["total_tokens"] == (
            rag_data["token_usage"]["prompt_tokens"] + rag_data["token_usage"]["completion_tokens"]
        )

        # Verify structured JSON log record contains request_id and masks the raw API key
        matching_logs = [r for r in log_records if r.get("request_id") == "req-trace-777"]
        assert len(matching_logs) == 1
        log_entry = matching_logs[0]
        assert log_entry["route"] == "/api/v1/rag/query"
        assert log_entry["status_code"] == 200
        assert log_entry["latency_ms"] >= 0.0
        assert eng_key not in json.dumps(log_entry)

        # 2. Offline RuleBasedReActPlanner agent run emits estimated token_usage
        agent_resp = client.post(
            "/api/v1/agent/run",
            json={"goal": "Check inventory for SKU-GPU-H100"},
            headers=eng_headers,
        )
        assert agent_resp.status_code == 200
        agent_data = agent_resp.json()
        assert agent_data["token_usage"]["accounting_mode"] == "estimated"
        assert agent_data["token_usage"]["total_tokens"] > 0

        m_text = client.get("/metrics", headers=metrics_headers).text
        assert 'fde_llm_tokens_total{accounting_mode="estimated",operation="rag_query",token_type="total"}' in m_text
        assert 'fde_llm_tokens_total{accounting_mode="estimated",operation="agent_run",token_type="total"}' in m_text

    # 3. Now test FDE_PLANNER=openai (airgapped_mode=False) with an OpenAI usage payload -> provider_reported
    def mock_openai_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "SKU-GPU-H100 has 20 units available.",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 140,
                    "completion_tokens": 35,
                    "total_tokens": 175,
                },
            },
        )

    mock_llm_client = httpx.Client(
        transport=httpx.MockTransport(mock_openai_handler),
        base_url="https://api.openai.com/v1",
    )
    app_openai = create_vpc_app(
        env="production",
        var_dir=var_dir,
        api_keys_file=keys_file,
        metrics_token_file=metrics_file,
        airgapped_mode=False,
        planner_mode="openai",
        openai_http_client=mock_llm_client,
        auto_seed_on_startup=False,
    )
    with TestClient(app_openai) as client_oa:
        oa_resp = client_oa.post(
            "/api/v1/agent/run",
            json={"goal": "Summarize SKU-GPU-H100 availability"},
            headers=eng_headers,
        )
        assert oa_resp.status_code == 200
        oa_data = oa_resp.json()
        assert oa_data["token_usage"] == {
            "prompt_tokens": 140,
            "completion_tokens": 35,
            "total_tokens": 175,
            "accounting_mode": "provider_reported",
        }
        oa_metrics = client_oa.get("/metrics", headers=metrics_headers).text
        assert (
            'fde_llm_tokens_total{accounting_mode="provider_reported",operation="agent_run",token_type="total"} 175.0'
            in oa_metrics
        )


def test_streaming_body_size_limit_413_and_bounded_rate_limiter_429(tmp_path: Path) -> None:
    var_dir, keys_file, metrics_file, bundle, keys = _setup_test_env(tmp_path)
    app = create_vpc_app(
        env="production",
        var_dir=var_dir,
        api_keys_file=keys_file,
        metrics_token_file=metrics_file,
        auto_seed_on_startup=True,
        allow_sample_seed_in_prod=True,
        max_body_bytes=256,
        rate_limit_rps=1.0,
        rate_limit_burst=2,
    )
    eng_headers = {"X-API-Key": keys["acme_field_engineer"]}
    metrics_headers = {"Authorization": f"Bearer {bundle.metrics_token}"}

    with TestClient(app) as client:
        # 1. Fixed Content-Length > 256 bytes -> 413 Payload Too Large
        big_payload = {"query": "A" * 400}
        r_fixed = client.post("/api/v1/rag/query", json=big_payload, headers=eng_headers)
        assert r_fixed.status_code == 413

        # 2. Streamed/chunked request body exceeding 256 bytes -> 413 Payload Too Large
        def chunk_generator() -> Iterator[bytes]:
            yield b'{"query": "'
            for _ in range(10):
                yield b"X" * 50
            yield b'"}'

        r_chunked = client.post(
            "/api/v1/rag/query",
            content=chunk_generator(),
            headers={**eng_headers, "Content-Type": "application/json"},
        )
        assert r_chunked.status_code == 413

        # 3. Rate limiter: rotating fake X-API-Key headers still hits the per-IP fallback bucket after burst=2
        s1 = client.post(
            "/api/v1/rag/query",
            json={"query": "hi"},
            headers={"X-API-Key": "fake-random-key-1"},
        ).status_code
        s2 = client.post(
            "/api/v1/rag/query",
            json={"query": "hi"},
            headers={"X-API-Key": "fake-random-key-2"},
        ).status_code
        s3 = client.post(
            "/api/v1/rag/query",
            json={"query": "hi"},
            headers={"X-API-Key": "fake-random-key-3"},
        ).status_code
        assert s1 == 401
        assert s2 == 401
        assert s3 == 429

        m_text = client.get("/metrics", headers=metrics_headers).text
        assert 'error_class="payload_too_large"' in m_text
        assert 'error_class="rate_limited"' in m_text


def test_multi_tenant_isolation_across_rag_and_agent_endpoints(tmp_path: Path) -> None:
    var_dir, keys_file, metrics_file, bundle, keys = _setup_test_env(tmp_path)
    app = create_vpc_app(
        env="production",
        var_dir=var_dir,
        api_keys_file=keys_file,
        metrics_token_file=metrics_file,
        auto_seed_on_startup=True,
        allow_sample_seed_in_prod=True,
    )

    acme_op = {"X-API-Key": keys["acme_field_engineer"]}
    acme_sup = {"X-API-Key": keys["acme_exec_supervisor"]}
    globex_op = {"X-API-Key": keys["globex_operator"]}
    globex_sup = {"X-API-Key": keys["globex_supervisor"]}

    with TestClient(app) as client:
        # 1. acme_field_engineer starts a sensitive reservation in acme_corp
        run_acme = client.post(
            "/api/v1/agent/run",
            json={"goal": "Reserve 3 units of SKU-EDGE-NPU1 for LEAD-101"},
            headers=acme_op,
        )
        assert run_acme.status_code == 200
        acme_sess_id = run_acme.json()["session_id"]
        acme_appr_id = run_acme.json()["pending_approval"]["approval_id"]
        assert run_acme.json()["tenant_id"] == "acme_corp"

        # 2. globex_operator and globex_supervisor (tenant=globex_inc) are blocked (403) from
        #    reading or approving acme_corp's session or approval!
        assert client.get(f"/api/v1/agent/sessions/{acme_sess_id}", headers=globex_op).status_code == 403
        assert client.get(f"/api/v1/agent/sessions/{acme_sess_id}", headers=globex_sup).status_code == 403
        assert (
            client.post(
                f"/api/v1/agent/approvals/{acme_appr_id}/decide",
                json={"decision": "APPROVE", "reason": "Cross-tenant hijack attempt"},
                headers=globex_sup,
            ).status_code
            == 403
        )

        # 3. acme_exec_supervisor approves acme_corp's ticket -> SKU-EDGE-NPU1 in acme_corp drops from 3 to 0
        decide_acme = client.post(
            f"/api/v1/agent/approvals/{acme_appr_id}/decide",
            json={"decision": "APPROVE", "reason": "Authorized by Acme supervisor"},
            headers=acme_sup,
        )
        assert decide_acme.status_code == 200
        assert decide_acme.json()["status"] == "COMPLETED"

        # 4. Verify globex_inc's SKU-EDGE-NPU1 stock is still 3 (complete ERP database tenant isolation)
        globex_check = client.post(
            "/api/v1/agent/run",
            json={"goal": "Check inventory for SKU-EDGE-NPU1"},
            headers=globex_op,
        )
        assert globex_check.status_code == 200
        assert "3 available" in globex_check.json()["final_answer"]
