$ErrorActionPreference = "Stop"

Write-Host "[1/6] Initializing local development secrets (0644 permissions for non-root containers)..."
python -m src.vpc_service.cli init-dev-secrets --out-dir deploy/secrets

$keysJson = Get-Content "deploy/secrets/fde_api_keys.json" | ConvertFrom-Json
$engKey = ($keysJson.PSObject.Properties | Where-Object { $_.Value.principal_id -eq "acme_field_engineer" } | Select-Object -First 1).Name
$metricsToken = (Get-Content "deploy/secrets/fde_metrics_token.txt" -Raw).Trim()

try {
    Write-Host "[2/6] Building and starting hardened VPC stack via docker compose..."
    docker compose up -d --build

    Write-Host "[3/6] Validating Prometheus configuration and alert rules via promtool inside prometheus container..."
    docker compose exec -T prometheus promtool check config /etc/prometheus/prometheus.yml
    docker compose exec -T prometheus promtool check rules /etc/prometheus/prometheus_rules.yml

    Write-Host "[4/6] Waiting for /healthz and /readyz on http://127.0.0.1:8000..."
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $hz = Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -Method Get -TimeoutSec 2
            $rz = Invoke-RestMethod -Uri "http://127.0.0.1:8000/readyz" -Method Get -TimeoutSec 2
            if ($hz.status -eq "alive" -and $rz.status -eq "ready") {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $ready) {
        throw "VPC service failed to reach ready state within 30s."
    }

    Write-Host "[5/6] Executing authenticated smoke requests against /api/v1/rag/query, /api/v1/agent/run, and /metrics..."
    $headers = @{ "X-API-Key" = $engKey; "Content-Type" = "application/json" }
    $ragBody = '{"query": "What is the minimum required password length?", "top_k": 3}'
    $ragResp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/rag/query" -Method Post -Headers $headers -Body $ragBody
    if (-not $ragResp.token_usage) { throw "Missing token_usage in RAG response" }

    $agentBody = '{"goal": "Check inventory for SKU-GPU-H100 and SKU-NET-SW400."}'
    $agentResp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/agent/run" -Method Post -Headers $headers -Body $agentBody
    if ($agentResp.status -ne "COMPLETED") { throw "Unexpected agent status: $($agentResp.status)" }

    $metricsHeaders = @{ "Authorization" = "Bearer $metricsToken" }
    $metricsText = (Invoke-WebRequest -Uri "http://127.0.0.1:8000/metrics" -Headers $metricsHeaders -UseBasicParsing).Content
    if ($metricsText -notmatch "fde_http_requests_total") { throw "Missing fde_http_requests_total in /metrics" }

    Write-Host "[6/6] Smoke verification succeeded!"
} finally {
    Write-Host "Tearing down VPC stack..."
    docker compose down -v
}
