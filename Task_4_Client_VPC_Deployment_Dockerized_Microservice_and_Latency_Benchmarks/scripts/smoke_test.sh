#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Initializing local development secrets (0644 permissions for non-root containers)..."
python3 -m src.vpc_service.cli init-dev-secrets --out-dir deploy/secrets

ENG_KEY=$(python3 -c "import json; d=json.load(open('deploy/secrets/fde_api_keys.json')); print(next(k for k,v in d.items() if v['principal_id']=='acme_field_engineer'))")
METRICS_TOKEN=$(tr -d '\r\n' < deploy/secrets/fde_metrics_token.txt)

cleanup() {
  echo "Tearing down VPC stack..."
  docker compose down -v
}
trap cleanup EXIT

echo "[2/6] Building and starting hardened VPC stack via docker compose..."
docker compose up -d --build

echo "[3/6] Validating Prometheus configuration and alert rules via promtool inside prometheus container..."
docker compose exec -T prometheus promtool check config /etc/prometheus/prometheus.yml
docker compose exec -T prometheus promtool check rules /etc/prometheus/prometheus_rules.yml

echo "[4/6] Waiting for /healthz and /readyz on http://127.0.0.1:8000..."
READY=0
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/healthz >/dev/null && curl -fsS http://127.0.0.1:8000/readyz >/dev/null; then
    READY=1
    break
  fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: VPC service failed to reach ready state within 30s" >&2
  exit 1
fi

echo "[5/6] Executing authenticated smoke requests against /api/v1/rag/query, /api/v1/agent/run, and /metrics..."
curl -fsS -X POST http://127.0.0.1:8000/api/v1/rag/query \
  -H "X-API-Key: ${ENG_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the minimum required password length?", "top_k": 3}' | grep -q '"token_usage"'

curl -fsS -X POST http://127.0.0.1:8000/api/v1/agent/run \
  -H "X-API-Key: ${ENG_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"goal": "Look up customer CUST-001 and check inventory for SKU-COOL-100."}' | grep -q '"COMPLETED"'

curl -fsS http://127.0.0.1:8000/metrics \
  -H "Authorization: Bearer ${METRICS_TOKEN}" | grep -q 'fde_http_requests_total'

echo "[6/6] Smoke verification succeeded!"
