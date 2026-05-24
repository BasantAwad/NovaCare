$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Join-Path $Root "services\asl"
Set-Location $WorkDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   NovaCare - Starting ASL Model API" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$VenvPath = Join-Path $Root "services\asl-model\venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "[!] No venv found!" -ForegroundColor Red
    Write-Host "[!] Creating virtual environment (venv)..." -ForegroundColor Yellow
    python -m venv $VenvPath
    & "$VenvPath\Scripts\Activate.ps1"
    Write-Host "[*] Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "[OK] Dependencies installed" -ForegroundColor Green
} else {
    & "$VenvPath\Scripts\Activate.ps1"
    Write-Host "[OK] ASL Model venv (from asl-model) activated" -ForegroundColor Green
}

Write-Host "[*] Starting FastAPI on port 8001..." -ForegroundColor Cyan
python -m api.main --port 8001
