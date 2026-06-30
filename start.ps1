# start.ps1 - Startup script for GhostSig MVP

$ErrorActionPreference = "Stop"

# Step 1: Handle the case where Docker daemon is not yet running
$dockerRunning = $false
$timeout = 30
$elapsed = 0

Write-Host "Checking if Docker daemon is running..." -ForegroundColor Cyan
while (-not $dockerRunning -and $elapsed -lt $timeout) {
    & docker info >$null 2>&1
    if ($LASTEXITCODE -eq 0) {
        $dockerRunning = $true
    } else {
        Write-Host "Docker daemon is not running yet. Retrying in 2 seconds... ($elapsed/$timeout s)" -ForegroundColor Yellow
        Start-Sleep -Seconds 2
        $elapsed += 2
    }
}

if (-not $dockerRunning) {
    Write-Error "Docker daemon is not running. Please start Docker Desktop and try again."
    exit 1
}

Write-Host "Starting Docker containers..." -ForegroundColor Cyan
& docker compose up -d

# Wait for pgvector postgres container to be healthy
$postgresReady = $false
$elapsed = 0
while (-not $postgresReady -and $elapsed -lt $timeout) {
    & docker exec ghostsig-postgres-1 pg_isready -U ghostsig >$null 2>&1
    if ($LASTEXITCODE -eq 0) {
        $postgresReady = $true
    } else {
        Write-Host "Waiting for Postgres to be ready... ($elapsed/$timeout s)" -ForegroundColor Yellow
        Start-Sleep -Seconds 2
        $elapsed += 2
    }
}

if (-not $postgresReady) {
    Write-Error "Postgres container is not healthy/ready. Please check container logs."
    exit 1
}

Write-Host "Infrastructure is healthy!" -ForegroundColor Green

# Step 2 & 3: Launch API in a new PowerShell window
Write-Host "Launching API in a new window..." -ForegroundColor Cyan
$apiCommand = "Set-Location '$PSScriptRoot'; .\.venv\Scripts\Activate.ps1; python -m uvicorn api.main:app --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCommand

# Step 2 & 4: Launch Streamlit Dashboard in another new PowerShell window
Write-Host "Launching Streamlit Dashboard in a new window..." -ForegroundColor Cyan
$dashboardCommand = "Set-Location '$PSScriptRoot'; .\.venv\Scripts\Activate.ps1; streamlit run dashboard/app.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $dashboardCommand

# Step 5: Print confirmation message
Write-Host ""
Write-Host "GhostSig running — API: http://localhost:8000/docs | Dashboard: http://localhost:8501" -ForegroundColor Green
Write-Host ""
