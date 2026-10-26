param([int]$Port = 8000, [switch]$Workbench)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = (Get-Command python).Source
$logRoot = Join-Path $projectRoot '.odp-meta/web'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Test-Backend {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
        return $health.status -eq 'ok' -and $health.framework -eq 'django'
    } catch { return $false }
}

if (-not (Test-Backend)) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port is occupied by another service."
    }
    Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'od_web.app', '--port', "$Port") `
        -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot 'backend.out.log') `
        -RedirectStandardError (Join-Path $logRoot 'backend.err.log') | Out-Null
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        if (Test-Backend) { break }
        Start-Sleep -Milliseconds 300
    }
    if (-not (Test-Backend)) { throw 'Backend failed to start. Check .odp-meta/web/backend.err.log.' }
}
Write-Output "Dashboard: http://127.0.0.1:$Port"

if ($Workbench) {
    $env:ODP_API_URL = "http://127.0.0.1:$Port"
    if (-not (Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'streamlit', 'run', 'app.py', '--server.port', '8501') `
            -WorkingDirectory (Join-Path $projectRoot 'apps/streamlit') -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logRoot 'workbench.out.log') `
            -RedirectStandardError (Join-Path $logRoot 'workbench.err.log') | Out-Null
    }
    Write-Output 'Workbench: http://127.0.0.1:8501'
}
