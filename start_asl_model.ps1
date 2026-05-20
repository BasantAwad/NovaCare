$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Join-Path $Root "services\asl-model"
Set-Location $WorkDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   NovaCare - Starting ASL Model API" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path 'venv')) {
    Write-Host "[!] No venv found!" -ForegroundColor Red
    Write-Host "[!] Creating virtual environment (venv)..." -ForegroundColor Yellow
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    Write-Host "[*] Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "[OK] Dependencies installed" -ForegroundColor Green
} else {
    .\venv\Scripts\Activate.ps1
    Write-Host "[OK] ASL Model venv activated" -ForegroundColor Green
}

Write-Host "[*] Starting FastAPI on port 8001..." -ForegroundColor Cyan
python -m api.main --port 8001
